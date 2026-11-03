"""Paper liquidity-reward market maker.

Each cycle:
  1. (every few minutes) pick the reward markets with the best estimated reward per $ quoted
  2. read the live YES book of every selected market
  3. replay new public trades against our resting bids -> fills (maker, no fee)
  4. merge YES+NO pairs into $1 (complete sets), settle resolved inventory
  5. accrue rewards with Polymarket's scoring formula; credit them at the UTC day roll
  6. risk checks (drawdown / daily loss / kill switch) -> pull everything
  7. re-quote: bid YES at mid - s and NO at (1 - mid) - s inside the reward band, unless
     the market is moving fast, near resolution, outside [0.10, 0.90], or inventory is too
     one-sided (then only the side that completes sets is quoted)

Profit = rewards + spread captured on completed sets - losses on inventory left unhedged
when price moves (adverse selection). Only running it tells you which term wins.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import math
import signal
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

from ..api.polymarket import Market, OrderBook, PolymarketData
from ..config import Config
from ..db import Database
from ..execution.portfolio import Portfolio
from ..market.scanner import MarketScanner, hours_to_close
from ..risk.manager import RiskManager
from .rewards import estimate_share
from .sim import FillSimulator, Quote, TapeTrade

log = logging.getLogger("polyagent.mm")


def _day(ts: float) -> str:
    return dt.datetime.fromtimestamp(ts, dt.timezone.utc).date().isoformat()


def reward_rate(m: Market) -> float:
    return sum(float(x.get("rewardsDailyRate") or 0) for x in (m.raw.get("clobRewards") or []))


@dataclass
class MMMarket:
    market: Market
    rate: float  # $/day pool
    max_spread: float  # price units
    min_size: float  # shares
    mids: deque = field(default_factory=lambda: deque(maxlen=200))
    last_accrual: float = 0.0
    paused: str = ""
    est_per_day: float = 0.0  # estimated reward at selection time (ranks collateral allocation)
    sim: FillSimulator = field(default_factory=FillSimulator)
    yes_q: Quote | None = None
    no_q: Quote | None = None


def target_quotes(mid: float, max_spread: float, tick: float, spread_frac: float,
                  best_bid: float | None, best_ask: float | None) -> tuple[float, float] | None:
    """Prices for (YES bid, NO bid): inside the reward band, on the tick grid, never crossing."""
    s = max(tick, round(spread_frac * max_spread / tick) * tick)
    yb = math.floor((mid - s) / tick + 1e-9) * tick
    nb = math.floor(((1 - mid) - s) / tick + 1e-9) * tick
    if mid - yb >= max_spread:
        yb += tick
    if (1 - mid) - nb >= max_spread:
        nb += tick
    if best_ask is not None and yb >= best_ask:
        yb = best_ask - tick
    if best_bid is not None and nb >= 1 - best_bid:  # NO ask = 1 - YES bid
        nb = (1 - best_bid) - tick
    yb, nb = round(yb, 4), round(nb, 4)
    if yb < tick or nb < tick or yb + nb >= 1:
        return None
    if mid - yb >= max_spread or (1 - mid) - nb >= max_spread:
        return None
    return yb, nb


class MarketMaker:
    def __init__(self, cfg: Config, api: PolymarketData | None = None, db: Database | None = None):
        if cfg.mode != "paper":
            raise RuntimeError("market maker is paper-only until its paper results justify live use")
        self.cfg = cfg
        self.db = db or Database(cfg.db_path)
        self.api = api or PolymarketData(cfg)
        self.portfolio = Portfolio(self.db, cfg.starting_capital)
        self.risk = RiskManager(cfg, self.db)
        self.scanner = MarketScanner(self.api, self.db, cfg)
        self.markets: dict[str, MMMarket] = {}
        self.last_select = 0.0
        self.last_settle = 0.0
        self._stop = threading.Event()

    # --- lifecycle ----------------------------------------------------------
    def run_forever(self) -> None:
        signal.signal(signal.SIGTERM, lambda *_: self._stop.set())
        signal.signal(signal.SIGINT, lambda *_: self._stop.set())
        self.db.audit("mm_start", {"mode": self.cfg.mode})
        log.info("market maker started (paper)")
        while not self._stop.is_set():
            t0 = time.time()
            try:
                self.cycle()
                self.db.set_state("last_error", None)
            except Exception as e:  # noqa: BLE001
                log.exception("mm cycle failed")
                self.db.set_state("last_error", {"ts": time.time(), "error": repr(e)[:500]})
                self.db.audit("mm_cycle_error", {"error": repr(e)[:1000]}, level="ERROR")
            self._stop.wait(max(1.0, self.cfg.mm_cycle_sec - (time.time() - t0)))
        self.pull_all("shutdown")
        self.db.audit("mm_stop", {})

    def cycle(self, now: float | None = None) -> None:
        now = now or time.time()
        hb = self.db.get_state("heartbeat") or {}
        self.db.set_state("heartbeat", {"ts": now, "cycle": hb.get("cycle", 0) + 1, "mode": "paper-mm"})
        kill = Path(self.cfg.kill_file).exists() or bool(self.db.get_state("kill"))
        if kill:
            self.pull_all("kill switch")
            if not self.db.get_state("kill_handled"):
                self.risk.halt("manual kill switch")
                self.db.set_state("kill_handled", True)
        if now - self.last_select > self.cfg.mm_reselect_minutes * 60 and not kill:
            self.select_markets(now)
        books = self._books()
        if books:  # lets the dashboard mark inventory
            self.db.executemany(
                "INSERT INTO book_snapshots(token_id,ts,best_bid,best_ask,bid_depth,ask_depth) VALUES(?,?,?,?,?,?)",
                [(t, now, b.best_bid, b.best_ask, sum(x for _, x in b.bids[:5]), sum(x for _, x in b.asks[:5]))
                 for t, b in books.items()]
                + [(mm.market.no_token, now, 1 - b.best_ask if b.best_ask is not None else None,
                    1 - b.best_bid if b.best_bid is not None else None, None, None)
                   for mm in self.markets.values() if (b := books.get(mm.market.yes_token))])
        self.process_tape(now)
        self.merge_sets()
        if now - self.last_settle > 600:
            self.settle_resolved()
            self.last_settle = now
        self.accrue_rewards(now, books)
        self.credit_rewards(now)
        marks = {t: b.mid for t, b in books.items() if b.mid is not None}
        for mm in self.markets.values():
            b = books.get(mm.market.yes_token)
            if b and b.mid is not None:
                marks[mm.market.no_token] = 1 - b.mid
        st = self.portfolio.state(marks)
        self.portfolio.snapshot(st)
        breach = self.risk.check_limits(st)
        if breach:
            self.pull_all(breach)
            self.risk.halt(breach, until_next_day="daily" in breach)
            return
        halted, why = self.risk.is_halted()
        if halted or kill:
            self.pull_all(why or "halted")
            return
        self.requote(now, books, st)
        self.persist_quotes(now)

    # --- market selection -----------------------------------------------------
    def candidates(self) -> list[Market]:
        raw: list[dict] = []
        for off in range(0, self.cfg.mm_universe, 100):
            raw += self.api.gamma.get("/markets", {"active": "true", "closed": "false", "limit": 100,
                                                   "offset": off, "order": "volume24hr", "ascending": "false"})
        out = []
        for d in raw:
            m = Market.from_gamma(d)
            if not d.get("enableOrderBook") or len(m.token_ids) != 2 or not m.accepting_orders:
                continue
            if reward_rate(m) <= 0 or not d.get("rewardsMaxSpread"):
                continue
            h = hours_to_close(m)
            if h is None or h < self.cfg.mm_min_hours_to_close:
                continue
            out.append(m)
        return out

    def select_markets(self, now: float) -> None:
        cands = self.candidates()
        self.scanner.upsert(cands)
        books = self.api.order_books([m.yes_token for m in cands]) if cands else {}
        scored = []
        for m in cands:
            b = books.get(m.yes_token)
            if not b or b.mid is None or not (self.cfg.mm_mid_min <= b.mid <= self.cfg.mm_mid_max):
                continue
            v = float(m.raw.get("rewardsMaxSpread")) / 100
            mn = float(m.raw.get("rewardsMinSize") or 0)
            tq = target_quotes(b.mid, v, m.tick_size, self.cfg.mm_spread_frac, b.best_bid, b.best_ask)
            if not tq:
                continue
            size = self.cfg.mm_capital_per_market / (tq[0] + tq[1])
            if size < mn:
                continue
            est = estimate_share(mid=b.mid, max_spread=v, min_size=mn, book_bids=b.bids, book_asks=b.asks,
                                 our_yes_bid=(tq[0], size), our_no_bid=(tq[1], size))
            scored.append((reward_rate(m) * est.share, m, v, mn))
        scored.sort(key=lambda x: -x[0])
        keep = {m.condition_id for _, m, _, _ in scored[: self.cfg.mm_max_markets]}
        for cid in list(self.markets):
            if cid not in keep:
                self._cancel(self.markets[cid], now)
                del self.markets[cid]
        for est, m, v, mn in scored[: self.cfg.mm_max_markets]:
            if m.condition_id in self.markets:
                self.markets[m.condition_id].market = m
                self.markets[m.condition_id].est_per_day = est
                continue
            mm = MMMarket(m, reward_rate(m), v, mn, last_accrual=now, est_per_day=est)
            mm.sim.tick = m.tick_size
            self.markets[m.condition_id] = mm
        self.last_select = now
        self.db.audit("mm_select", [{"q": m.question[:60], "est_per_day": round(e, 2), "pool": reward_rate(m)}
                                    for e, m, _, _ in scored[: self.cfg.mm_max_markets]])

    def _books(self) -> dict[str, OrderBook]:
        toks = [mm.market.yes_token for mm in self.markets.values()]
        return self.api.order_books(toks) if toks else {}

    # --- fills ----------------------------------------------------------------
    def process_tape(self, now: float) -> None:
        for mm in self.markets.values():
            if not (mm.yes_q or mm.no_q) and not any(q.cancelled_ts and q.cancelled_ts > now - 600
                                                      for q in mm.sim.quotes):
                continue
            try:
                raw = self.api.market_trades(mm.market.condition_id, max_items=200)
            except Exception as e:  # noqa: BLE001
                log.warning("tape %s failed: %s", mm.market.condition_id[:10], e)
                continue
            tape = []
            for d in raw:
                ts = float(d.get("timestamp") or 0)
                if ts < now - 900:
                    continue
                key = f'{d.get("transaction_hash")}|{d.get("token_id")}|{d.get("side")}|{d.get("size")}|{d.get("price")}'
                tape.append(TapeTrade(key, str(d.get("token_id")), (d.get("side") or "").upper(),
                                      float(d.get("price") or 0), float(d.get("size") or 0), ts))
            for t in sorted(tape, key=lambda x: x.ts):
                for f in mm.sim.on_trade(t):
                    self._record_fill(mm, f.token_id, f.price, f.size, f.ts, f.trade_key)
            mm.sim.prune(now - 900)

    def _record_fill(self, mm: MMMarket, token: str, price: float, size: float, ts: float, key: str) -> None:
        m = mm.market
        self.portfolio.apply_fill(token_id=token, condition_id=m.condition_id, event_id=m.event_id,
                                  side="BUY", price=price, size=size, fee=0.0)
        outcome = m.outcomes[0] if token == m.yes_token else (m.outcomes[1] if len(m.outcomes) > 1 else "No")
        self.db.execute(
            "INSERT OR IGNORE INTO fills(order_id,exchange_trade_id,ts,token_id,condition_id,side,price,size,fee,"
            "realized_pnl,question,outcome,strategy) VALUES(NULL,?,?,?,?,?,?,?,0,0,?,?,'mm')",
            ("mm:" + key + ":" + token, ts, token, m.condition_id, "BUY", price, size, m.question, outcome))
        self.db.audit("mm_fill", {"cid": m.condition_id, "token": token, "price": price, "size": size})

    def merge_sets(self) -> None:
        """YES + NO = $1 (CTF merge). Paper merges instantly; on-chain it costs a little gas."""
        for mm in self.markets.values():
            self._merge(mm.market)

    def _merge(self, m: Market) -> None:
        y = self.db.one("SELECT * FROM positions WHERE token_id=?", (m.yes_token,))
        n = self.db.one("SELECT * FROM positions WHERE token_id=?", (m.no_token,))
        if not y or not n:
            return
        k = min(y["size"], n["size"])
        if k < 1e-6:
            return
        pnl = k * (1.0 - y["avg_cost"] - n["avg_cost"])
        now = time.time()
        self.db.execute("UPDATE positions SET size=size-?, realized_pnl=realized_pnl+?, updated_at=? WHERE token_id=?",
                        (k, pnl, now, m.yes_token))
        self.db.execute("UPDATE positions SET size=size-?, updated_at=? WHERE token_id=?", (k, now, m.no_token))
        self.portfolio.set_cash(self.portfolio.cash + k)
        self.db.execute(
            "INSERT INTO fills(order_id,exchange_trade_id,ts,token_id,condition_id,side,price,size,fee,realized_pnl,"
            "question,outcome,strategy) VALUES(NULL,?,?,?,?,'MERGE',1.0,?,0,?,?,'YES+NO','mm')",
            (f"merge:{m.condition_id}:{now}", now, m.yes_token, m.condition_id, k, pnl, m.question))
        self.db.audit("mm_merge", {"cid": m.condition_id, "sets": k, "pnl": pnl})

    def settle_resolved(self) -> None:
        held = {p["condition_id"] for p in self.portfolio.positions()}
        for cid in held:
            try:
                m = self.api.market_by_condition(cid)
            except Exception:  # noqa: BLE001
                continue
            if m and m.closed and m.outcome_prices and set(m.outcome_prices) <= {0.0, 1.0}:
                for tok, payout in zip(m.token_ids, m.outcome_prices):
                    self.portfolio.settle(tok, payout)

    # --- rewards ----------------------------------------------------------------
    def accrue_rewards(self, now: float, books: dict[str, OrderBook]) -> None:
        for mm in self.markets.values():
            dt_s = min(now - (mm.last_accrual or now), 3 * self.cfg.mm_cycle_sec)  # no credit for downtime
            mm.last_accrual = now
            b = books.get(mm.market.yes_token)
            live_y = mm.yes_q if mm.yes_q and mm.yes_q.live_at(now) else None
            live_n = mm.no_q if mm.no_q and mm.no_q.live_at(now) else None
            if dt_s <= 0 or not b or b.mid is None or not (live_y or live_n):
                continue
            est = estimate_share(mid=b.mid, max_spread=mm.max_spread, min_size=mm.min_size,
                                 book_bids=b.bids, book_asks=b.asks,
                                 our_yes_bid=(live_y.price, live_y.size) if live_y else None,
                                 our_no_bid=(live_n.price, live_n.size) if live_n else None)
            amt = mm.rate * est.share * dt_s / 86400
            if amt <= 0:
                continue
            self.db.execute(
                "INSERT INTO mm_rewards(day,condition_id,accrued,seconds,share_sum) VALUES(?,?,?,?,?) "
                "ON CONFLICT(day,condition_id) DO UPDATE SET accrued=accrued+excluded.accrued,"
                "seconds=seconds+excluded.seconds, share_sum=share_sum+excluded.share_sum",
                (_day(now), mm.market.condition_id, amt, dt_s, est.share * dt_s))

    def credit_rewards(self, now: float) -> None:
        """Polymarket pays daily at 00:00 UTC, minimum $1 per address."""
        today = _day(now)
        for r in self.db.query("SELECT day, SUM(accrued) a FROM mm_rewards WHERE day<? AND paid=0 GROUP BY day",
                               (today,)):
            paid = r["a"] if r["a"] >= 1.0 else 0.0
            if paid:
                self.portfolio.set_cash(self.portfolio.cash + paid)
                self.db.execute(
                    "INSERT INTO fills(order_id,exchange_trade_id,ts,token_id,condition_id,side,price,size,fee,"
                    "realized_pnl,question,outcome,strategy) VALUES(NULL,?,?,'','','REWARD',1.0,?,0,?,?,'','mm')",
                    (f"reward:{r['day']}", now, paid, paid, f"Liquidity rewards for {r['day']}"))
            self.db.execute("UPDATE mm_rewards SET paid=1 WHERE day=?", (r["day"],))
            self.db.audit("mm_reward_credit", {"day": r["day"], "accrued": r["a"], "paid": paid})

    # --- quoting --------------------------------------------------------------
    def _cancel(self, mm: MMMarket, now: float, which: str = "both") -> None:
        if which in ("both", "yes") and mm.yes_q:
            mm.sim.cancel(mm.market.yes_token, now)
            mm.yes_q = None
        if which in ("both", "no") and mm.no_q:
            mm.sim.cancel(mm.market.no_token, now)
            mm.no_q = None

    def pull_all(self, reason: str) -> None:
        now = time.time()
        live = sum(1 for mm in self.markets.values() if mm.yes_q or mm.no_q)
        for mm in self.markets.values():
            self._cancel(mm, now)
            mm.paused = reason
        if live:
            self.db.audit("mm_pull_all", {"reason": reason}, level="WARNING")
        self.persist_quotes(now)

    def _guard(self, mm: MMMarket, b: OrderBook, now: float) -> str:
        c = self.cfg
        if b is None or b.mid is None:
            return "no book"
        mm.mids.append((now, b.mid))
        recent = [p for t, p in mm.mids if t >= now - c.mm_vol_window_sec]
        if recent and max(recent) - min(recent) > c.mm_vol_pull:
            return f"moving fast ({max(recent) - min(recent):.3f} in {c.mm_vol_window_sec}s)"
        if not (c.mm_mid_min <= b.mid <= c.mm_mid_max):
            return "mid outside band"
        h = hours_to_close(mm.market, now)
        if h is None or h < c.mm_min_hours_to_close:
            return "too close to resolution"
        if b.spread is None or b.spread > 2 * mm.max_spread:
            return "book too wide"
        return ""

    def requote(self, now: float, books: dict[str, OrderBook], st) -> None:
        c = self.cfg
        budget = st.cash - c.min_cash_reserve_pct * st.equity  # resting bids lock collateral
        for mm in sorted(self.markets.values(), key=lambda x: -x.est_per_day):
            m = mm.market
            b = books.get(m.yes_token)
            why = self._guard(mm, b, now)
            if why:
                if mm.yes_q or mm.no_q:
                    self.db.audit("mm_pause", {"cid": m.condition_id, "reason": why})
                self._cancel(mm, now)
                mm.paused = why
                continue
            mm.paused = ""
            tq = target_quotes(b.mid, mm.max_spread, m.tick_size, c.mm_spread_frac, b.best_bid, b.best_ask)
            if not tq:
                self._cancel(mm, now)
                mm.paused = "no valid quote"
                continue
            yb, nb = tq
            size = math.floor(min(c.mm_capital_per_market, max(budget, 0)) / (yb + nb))
            if size < max(mm.min_size, m.min_size):
                self._cancel(mm, now)
                mm.paused = "insufficient free collateral"
                continue
            y = self.db.one("SELECT size, avg_cost FROM positions WHERE token_id=?", (m.yes_token,))
            n = self.db.one("SELECT size, avg_cost FROM positions WHERE token_id=?", (m.no_token,))
            y_inv = (y["size"] if y else 0.0) * b.mid
            n_inv = (n["size"] if n else 0.0) * (1 - b.mid)
            want_yes = y_inv < c.mm_max_inventory_usd
            want_no = n_inv < c.mm_max_inventory_usd
            for side, want, price, tok, comp in (("yes", want_yes, yb, m.yes_token, m.no_token),
                                                 ("no", want_no, nb, m.no_token, m.yes_token)):
                cur = mm.yes_q if side == "yes" else mm.no_q
                if not want:
                    self._cancel(mm, now, side)
                    continue
                if cur and abs(cur.price - price) < m.tick_size / 2 and cur.size >= 0.5 * size:
                    continue  # keep queue position
                self._cancel(mm, now, side)
                if side == "yes":
                    queue = sum(s for p, s in b.bids if abs(p - price) < m.tick_size / 2)
                else:
                    queue = sum(s for p, s in b.asks if abs((1 - p) - price) < m.tick_size / 2)
                q = Quote(tok, comp, price, float(size), now, queue)
                mm.sim.place(q)
                if side == "yes":
                    mm.yes_q = q
                else:
                    mm.no_q = q
            budget -= sum(q.price * q.size for q in (mm.yes_q, mm.no_q) if q)

    def persist_quotes(self, now: float) -> None:
        rows = []
        for mm in self.markets.values():
            for q, side in ((mm.yes_q, "YES"), (mm.no_q, "NO")):
                rows.append((mm.market.condition_id, mm.market.question, side, q.price if q else None,
                             q.size if q else None, q.queue_ahead if q else None, mm.rate, mm.paused, now))
        with self.db.tx() as cx:
            cx.execute("DELETE FROM mm_quotes")
            cx.executemany("INSERT INTO mm_quotes VALUES(?,?,?,?,?,?,?,?,?)", rows)
