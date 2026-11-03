"""Paper memecoin trader + unbiased data collector.

Every new Solana pool seen on GeckoTerminal is recorded with its RugCheck report and then
tracked (DexScreener) for `meme_track_hours`, including the ones that die. That forward
dataset is what the backtester uses, so it has no survivorship bias.

Paper fills use real Jupiter quotes (USDC <-> token), so slippage and pulled liquidity are
priced in; open positions are valued at what a sell quote would actually return.
Live execution (signing Jupiter swap transactions) is deliberately not implemented.
"""
from __future__ import annotations

import json
import logging
import signal
import threading
import time
from pathlib import Path

from ..config import Config
from ..db import Database
from ..execution.portfolio import Portfolio
from ..risk.manager import PortfolioState, RiskManager
from .sources import USDC, MemeSources
from .strategy import MemeParams, TokenState, entry_signal, exit_signal

log = logging.getLogger("polyagent.meme")

SCHEMA = """
CREATE TABLE IF NOT EXISTS meme_tokens (
    mint TEXT PRIMARY KEY, symbol TEXT, name TEXT, pool TEXT, dex TEXT,
    discovered_ts REAL, created_ts REAL, rug_score REAL, lp_locked_pct REAL, risks TEXT,
    dead INTEGER DEFAULT 0, last_seen REAL, last_obs REAL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS meme_obs (
    mint TEXT, ts REAL, price_usd REAL, liquidity REAL, fdv REAL, vol_m5 REAL, vol_h1 REAL,
    buys_m5 INTEGER, sells_m5 INTEGER, buys_h1 INTEGER, sells_h1 INTEGER, chg_m5 REAL, chg_h1 REAL,
    PRIMARY KEY (mint, ts)
);
CREATE INDEX IF NOT EXISTS ix_meme_obs_ts ON meme_obs(ts);
CREATE TABLE IF NOT EXISTS meme_positions (
    mint TEXT PRIMARY KEY, symbol TEXT, entry_ts REAL, raw_amount INTEGER, cost_usd REAL,
    peak_mult REAL, took_profit INTEGER DEFAULT 0, last_value REAL, quote_failures INTEGER DEFAULT 0
);
"""


class MemeTrader:
    def __init__(self, cfg: Config, src: MemeSources | None = None, db: Database | None = None,
                 params: MemeParams | None = None):
        if cfg.mode != "paper":
            raise RuntimeError("meme trader is paper-only")
        self.cfg = cfg
        self.db = db or Database(cfg.db_path)
        self.db.conn.executescript(SCHEMA)
        self.src = src or MemeSources()
        self.p = params or MemeParams()
        self.portfolio = Portfolio(self.db, cfg.starting_capital)
        self.risk = RiskManager(cfg, self.db)
        self._stop = threading.Event()

    # --- lifecycle ------------------------------------------------------------
    def run_forever(self) -> None:
        signal.signal(signal.SIGTERM, lambda *_: self._stop.set())
        signal.signal(signal.SIGINT, lambda *_: self._stop.set())
        self.db.audit("meme_start", {})
        while not self._stop.is_set():
            t0 = time.time()
            try:
                self.cycle()
                self.db.set_state("last_error", None)
            except Exception as e:  # noqa: BLE001
                log.exception("meme cycle failed")
                self.db.set_state("last_error", {"ts": time.time(), "error": repr(e)[:500]})
                self.db.audit("meme_cycle_error", {"error": repr(e)[:1000]}, level="ERROR")
            self._stop.wait(max(1.0, self.cfg.meme_cycle_sec - (time.time() - t0)))

    def cycle(self, now: float | None = None) -> None:
        now = now or time.time()
        hb = self.db.get_state("heartbeat") or {}
        self.db.set_state("heartbeat", {"ts": now, "cycle": hb.get("cycle", 0) + 1, "mode": "paper-meme"})
        self.discover(now)
        states = self.observe(now)
        values = self.value_positions(now)
        equity = self.portfolio.cash + sum(values.values())
        st = self._snapshot(equity, now)
        kill = Path(self.cfg.kill_file).exists() or bool(self.db.get_state("kill"))
        breach = self.risk.check_limits(st)
        if kill or breach:
            reason = "kill switch" if kill else breach
            if not self.db.get_state("halt"):
                self.risk.halt(reason, until_next_day=bool(breach and "daily" in breach))
            self.manage_exits(now, values, force=True)  # flatten on risk events
            return
        if self.risk.is_halted()[0]:
            self.manage_exits(now, values)
            return
        self.manage_exits(now, values)
        self.enter(now, states, equity)

    # --- data -----------------------------------------------------------------
    def discover(self, now: float) -> None:
        known = {r["mint"] for r in self.db.query("SELECT mint FROM meme_tokens WHERE discovered_ts > ?",
                                                  (now - 7 * 86400,))}
        new = [p for p in self.src.new_pools(self.cfg.meme_discovery_pages) if p["mint"] not in known]
        seen = set()
        for p in new[: self.cfg.meme_max_new_per_cycle]:
            if p["mint"] in seen:
                continue
            seen.add(p["mint"])
            rep = self.src.rugcheck(p["mint"])
            created = None
            if p.get("created"):
                try:
                    created = time.mktime(time.strptime(p["created"][:19], "%Y-%m-%dT%H:%M:%S")) - time.timezone
                except ValueError:
                    pass
            self.db.execute(
                "INSERT OR IGNORE INTO meme_tokens(mint,pool,dex,discovered_ts,created_ts,rug_score,lp_locked_pct,risks,"
                "last_seen) VALUES(?,?,?,?,?,?,?,?,?)",
                (p["mint"], p["pool"], p["dex"], now, created or now, rep.score_normalised if rep else None,
                 rep.lp_locked_pct if rep else None,
                 json.dumps([r.get("name") for r in rep.risks]) if rep else None, now))

    def observe(self, now: float) -> dict[str, TokenState]:
        # polling schedule: young coins every cycle, older ones less often (thousands launch per day)
        rows = self.db.query(
            "SELECT * FROM meme_tokens WHERE dead=0 AND discovered_ts > ? AND ("
            " (? - created_ts < 3600) OR (? - created_ts < 21600 AND ? - last_obs >= 300) OR (? - last_obs >= 900))",
            (now - self.cfg.meme_track_hours * 3600, now, now, now, now))
        held = {r["mint"] for r in self.db.query("SELECT mint FROM meme_positions")}
        mints = list({r["mint"] for r in rows} | held)
        pairs = self.src.pairs(mints) if mints else {}
        meta = {r["mint"]: r for r in rows}
        obs, states = [], {}
        for mint in mints:
            pi = pairs.get(mint)
            if pi is None or pi.liquidity_usd < 100:
                m = meta.get(mint)
                if m and now - (m["last_seen"] or m["discovered_ts"]) > 1800:
                    self.db.execute("UPDATE meme_tokens SET dead=1 WHERE mint=?", (mint,))
                obs.append((mint, now, 0.0, pi.liquidity_usd if pi else 0.0, 0, 0, 0, 0, 0, 0, 0, 0, 0))
                continue
            self.db.execute("UPDATE meme_tokens SET symbol=?, name=?, last_seen=? WHERE mint=?",
                            (pi.symbol, pi.name, now, mint))
            obs.append((mint, now, pi.price_usd, pi.liquidity_usd, pi.fdv, pi.vol_m5, pi.vol_h1, pi.buys_m5,
                        pi.sells_m5, pi.buys_h1, pi.sells_h1, pi.chg_m5, pi.chg_h1))
            m = meta.get(mint)
            if m:
                created = (pi.created_ms / 1000) if pi.created_ms else m["created_ts"]
                states[mint] = TokenState(
                    age_min=(now - created) / 60, liquidity=pi.liquidity_usd, fdv=pi.fdv, vol_h1=pi.vol_h1,
                    buys_m5=pi.buys_m5, sells_m5=pi.sells_m5, chg_m5=pi.chg_m5, chg_h1=pi.chg_h1,
                    rug_score=m["rug_score"], risks=json.loads(m["risks"]) if m["risks"] else [])
        if obs:
            self.db.executemany("INSERT OR IGNORE INTO meme_obs VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)", obs)
            self.db.executemany("UPDATE meme_tokens SET last_obs=? WHERE mint=?", [(now, o[0]) for o in obs])
        return states

    # --- valuation & exits ------------------------------------------------------
    def value_positions(self, now: float) -> dict[str, float]:
        """Executable USD value of each position = Jupiter sell quote for the full amount."""
        values = {}
        for pos in self.db.query("SELECT * FROM meme_positions"):
            q = self.src.quote(pos["mint"], USDC, pos["raw_amount"])
            if q is None:
                fails = pos["quote_failures"] + 1
                self.db.execute("UPDATE meme_positions SET quote_failures=? WHERE mint=?", (fails, pos["mint"]))
                # no route for a while = liquidity pulled; value at zero
                values[pos["mint"]] = 0.0 if fails >= 3 else (pos["last_value"] or 0.0)
                continue
            v = q.out_amount / 1e6 - self.cfg.meme_tx_cost_usd
            v = max(v, 0.0)
            values[pos["mint"]] = v
            mult = v / pos["cost_usd"] if pos["cost_usd"] else 0
            self.db.execute("UPDATE meme_positions SET last_value=?, peak_mult=MAX(peak_mult, ?), quote_failures=0 "
                            "WHERE mint=?", (v, mult, pos["mint"]))
            unit = v / pos["raw_amount"] if pos["raw_amount"] else 0
            self.db.execute("INSERT INTO book_snapshots(token_id,ts,best_bid,best_ask) VALUES(?,?,?,?)",
                            (pos["mint"], now, unit, unit))
        return values

    def manage_exits(self, now: float, values: dict[str, float], force: bool = False) -> None:
        for pos in self.db.query("SELECT * FROM meme_positions"):
            v = values.get(pos["mint"])
            if v is None:
                continue
            mult = v / pos["cost_usd"] if pos["cost_usd"] else 0
            if pos["quote_failures"] >= 3:
                # no sell route for 3+ cycles: liquidity is gone. Write off after an hour of trying.
                if now - pos["entry_ts"] > 3600:
                    self.sell(pos, 1.0, "written_off", now)
                continue
            if force:
                reason, frac = "risk_flatten", 1.0
            else:
                reason, frac = exit_signal(mult, pos["peak_mult"] or mult, (now - pos["entry_ts"]) / 3600,
                                           bool(pos["took_profit"]), self.p)
            if reason:
                self.sell(pos, frac, reason, now)

    def sell(self, pos: dict, frac: float, reason: str, now: float) -> None:
        raw = int(pos["raw_amount"] * frac) if frac < 1 else int(pos["raw_amount"])
        q = self.src.quote(pos["mint"], USDC, raw)
        proceeds = (q.out_amount / 1e6 - self.cfg.meme_tx_cost_usd) if q else 0.0
        if q is None and reason != "written_off":
            return  # cannot sell right now; try again next cycle
        proceeds = max(proceeds, 0.0)
        cost_part = pos["cost_usd"] * raw / pos["raw_amount"]
        # Portfolio ledger works in "price per raw unit"
        self.portfolio.apply_fill(token_id=pos["mint"], condition_id=pos["mint"], event_id="meme", side="SELL",
                                  price=proceeds / raw if raw else 0.0, size=raw, fee=0.0)
        pnl = proceeds - cost_part
        self.db.execute(
            "INSERT INTO fills(order_id,exchange_trade_id,ts,token_id,condition_id,side,price,size,fee,realized_pnl,"
            "question,outcome,strategy) VALUES(NULL,?,?,?,?,'SELL',?,?,?,?,?,?,'meme')",
            (f"meme-sell:{pos['mint']}:{now}", now, pos["mint"], pos["mint"], proceeds / raw if raw else 0, raw,
             self.cfg.meme_tx_cost_usd, pnl, pos["symbol"] or pos["mint"][:8], reason))
        left = pos["raw_amount"] - raw
        if left <= 0 or frac >= 1:
            self.db.execute("DELETE FROM meme_positions WHERE mint=?", (pos["mint"],))
        else:
            self.db.execute("UPDATE meme_positions SET raw_amount=?, cost_usd=cost_usd-?, took_profit=1, "
                            "peak_mult=? WHERE mint=?", (left, cost_part, pos["peak_mult"], pos["mint"]))
        self.db.audit("meme_sell", {"mint": pos["mint"], "reason": reason, "proceeds": proceeds, "pnl": pnl})

    # --- entries ------------------------------------------------------------------
    def enter(self, now: float, states: dict[str, TokenState], equity: float) -> None:
        held = {r["mint"] for r in self.db.query("SELECT mint FROM meme_positions")}
        traded = {r["token_id"] for r in self.db.query("SELECT DISTINCT token_id FROM fills WHERE strategy='meme'")}
        slots = self.p.max_positions - len(held)
        for mint, st in sorted(states.items(), key=lambda kv: -kv[1].vol_h1):
            if slots <= 0:
                break
            if mint in held or mint in traded:
                continue  # one shot per coin: no revenge trading
            ok, why = entry_signal(st, self.p)
            if not ok:
                continue
            stake = min(self.p.stake_pct * equity, self.portfolio.cash - self.cfg.min_cash_reserve_pct * equity)
            if stake < self.cfg.min_order_usd:
                break
            probe = self.src.quote(USDC, mint, 1_000_000)  # $1
            full = self.src.quote(USDC, mint, int(stake * 1e6))
            if not probe or not full or probe.out_amount <= 0:
                continue
            impact = 1 - (full.out_amount / stake) / probe.out_amount
            if impact > self.p.max_entry_impact:
                self.db.audit("meme_skip_impact", {"mint": mint, "impact": impact})
                continue
            # can we get out? round-trip check on the fresh quote
            back = self.src.quote(mint, USDC, full.out_amount)
            if not back or back.out_amount / 1e6 < stake * 0.85:
                self.db.audit("meme_skip_roundtrip", {"mint": mint, "back": back.out_amount / 1e6 if back else None})
                continue
            cost = stake + self.cfg.meme_tx_cost_usd
            self.portfolio.apply_fill(token_id=mint, condition_id=mint, event_id="meme", side="BUY",
                                      price=stake / full.out_amount, size=full.out_amount,
                                      fee=self.cfg.meme_tx_cost_usd)
            sym = self.db.one("SELECT symbol FROM meme_tokens WHERE mint=?", (mint,))
            sym = (sym or {}).get("symbol") or mint[:8]
            self.db.execute("INSERT INTO meme_positions(mint,symbol,entry_ts,raw_amount,cost_usd,peak_mult,last_value) "
                            "VALUES(?,?,?,?,?,1.0,?)", (mint, sym, now, full.out_amount, cost, stake))
            self.db.execute(
                "INSERT INTO fills(order_id,exchange_trade_id,ts,token_id,condition_id,side,price,size,fee,realized_pnl,"
                "question,outcome,strategy) VALUES(NULL,?,?,?,?,'BUY',?,?,?,0,?,?,'meme')",
                (f"meme-buy:{mint}:{now}", now, mint, mint, stake / full.out_amount, full.out_amount,
                 self.cfg.meme_tx_cost_usd, sym, f"impact {impact:.1%}"))
            self.db.audit("meme_buy", {"mint": mint, "usd": stake, "impact": impact, "age_min": st.age_min,
                                       "liq": st.liquidity, "fdv": st.fdv, "rug": st.rug_score})
            slots -= 1

    def _snapshot(self, equity: float, now: float):
        """Risk state from *executable* equity (sell quotes), never from cost basis."""
        hw = max(float(self.db.get_state("high_water", equity)), equity)
        self.db.set_state("high_water", hw)
        today = time.strftime("%Y-%m-%d", time.gmtime(now))
        day = self.db.get_state("day_start", {}) or {}
        if day.get("date") != today:
            day = {"date": today, "equity": equity}
            self.db.set_state("day_start", day)
        st = PortfolioState(cash=self.portfolio.cash, equity=equity, high_water=hw,
                            day_start_equity=day["equity"], exposure_by_market={}, exposure_by_event={})
        start = self.portfolio.starting_capital
        self.db.execute("INSERT OR REPLACE INTO equity_snapshots VALUES(?,?,?,?,?,?,?,?)",
                        (now, self.portfolio.cash, equity - self.portfolio.cash, equity, equity - start,
                         (equity - start) / start if start else 0, hw, st.drawdown))
        return st
