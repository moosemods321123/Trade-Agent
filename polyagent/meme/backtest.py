"""Walk-forward backtest over the forward-collected meme_obs dataset.

No survivorship bias: every coin the collector discovered is included, and a coin whose
liquidity vanishes is valued at zero from then on. Execution cost is modelled for a
constant-product pool: buying x USD into a pool with total liquidity L moves price by
about x / (L / 2); the same applies on exit against the liquidity at that time. Plus a fixed
per-swap transaction cost. Positions are entered at most once per coin.
"""
from __future__ import annotations

import heapq
import itertools
import json
from dataclasses import asdict, dataclass

import numpy as np

from ..analytics.stats import max_drawdown
from .strategy import MemeParams, TokenState, entry_signal, exit_signal


@dataclass
class Obs:
    ts: float
    price: float
    liq: float
    fdv: float
    vol_h1: float
    buys_m5: int
    sells_m5: int
    chg_m5: float
    chg_h1: float


@dataclass
class Coin:
    mint: str
    symbol: str
    discovered: float
    created: float
    rug_score: float | None
    risks: list[str]
    obs: list[Obs]


def load(db) -> list[Coin]:
    coins = []
    meta = {r["mint"]: r for r in db.query("SELECT * FROM meme_tokens")}
    by: dict[str, list[Obs]] = {}
    for r in db.query("SELECT * FROM meme_obs ORDER BY ts"):
        by.setdefault(r["mint"], []).append(Obs(r["ts"], r["price_usd"] or 0.0, r["liquidity"] or 0.0, r["fdv"] or 0.0,
                                                 r["vol_h1"] or 0.0, r["buys_m5"] or 0, r["sells_m5"] or 0,
                                                 r["chg_m5"] or 0.0, r["chg_h1"] or 0.0))
    for mint, obs in by.items():
        m = meta.get(mint)
        if not m:
            continue
        coins.append(Coin(mint, m["symbol"] or "", m["discovered_ts"], m["created_ts"] or m["discovered_ts"],
                          m["rug_score"], json.loads(m["risks"]) if m["risks"] else [], obs))
    return sorted(coins, key=lambda c: c.discovered)


def _impact(usd: float, liq: float) -> float:
    if liq <= 0:
        return 1.0
    return min(usd / (liq / 2), 1.0)


@dataclass
class Trade:
    mint: str
    symbol: str
    entry_ts: float
    exit_ts: float
    cost: float
    proceeds: float
    reason: str

    @property
    def mult(self) -> float:
        return self.proceeds / self.cost if self.cost else 0.0


def simulate(coins: list[Coin], p: MemeParams, start_equity: float = 1000.0, tx_cost: float = 0.15,
             reserve_pct: float = 0.2) -> dict:
    """Chronological portfolio simulation. Decisions at time t use only observations <= t."""
    events = []
    seq = itertools.count()
    for c in coins:
        for i, o in enumerate(c.obs):
            heapq.heappush(events, (o.ts, next(seq), c, i))
    cash = start_equity
    open_pos: dict[str, dict] = {}
    trades: list[Trade] = []
    curve = []
    entered: set[str] = set()
    last_val: dict[str, float] = {}
    while events:
        ts, _, c, i = heapq.heappop(events)
        o = c.obs[i]
        pos = open_pos.get(c.mint)
        if pos:
            dead = o.price <= 0 or o.liq < 100
            gross = 0.0 if dead else pos["tokens"] * o.price
            value = max(gross * (1 - _impact(gross, o.liq)) - tx_cost, 0.0)
            last_val[c.mint] = value
            mult = value / pos["cost"]
            pos["peak"] = max(pos["peak"], mult)
            reason, frac = exit_signal(mult, pos["peak"], (ts - pos["ts"]) / 3600, pos["tp"], p)
            if dead and not reason:
                reason, frac = "rugged", 1.0
            if reason:
                part_cost = pos["cost"] * frac
                proceeds = value * frac
                cash += proceeds
                trades.append(Trade(c.mint, c.symbol, pos["ts"], ts, part_cost, proceeds, reason))
                if frac >= 1:
                    del open_pos[c.mint]
                    last_val.pop(c.mint, None)
                else:
                    pos["tokens"] *= (1 - frac)
                    pos["cost"] -= part_cost
                    pos["tp"] = True
        elif c.mint not in entered and len(open_pos) < p.max_positions and o.price > 0:
            st = TokenState(age_min=(ts - c.created) / 60, liquidity=o.liq, fdv=o.fdv, vol_h1=o.vol_h1,
                            buys_m5=o.buys_m5, sells_m5=o.sells_m5, chg_m5=o.chg_m5, chg_h1=o.chg_h1,
                            rug_score=c.rug_score, risks=c.risks)
            ok, _ = entry_signal(st, p)
            if ok:
                equity = cash + sum(last_val.get(m, v["cost"]) for m, v in open_pos.items())
                stake = min(p.stake_pct * equity, cash - reserve_pct * equity)
                imp = _impact(stake, o.liq)
                if stake >= 1.0 and imp <= p.max_entry_impact:
                    entered.add(c.mint)
                    cash -= stake + tx_cost
                    tokens = stake * (1 - imp) / o.price
                    open_pos[c.mint] = {"ts": ts, "tokens": tokens, "cost": stake + tx_cost, "peak": 1.0, "tp": False}
        equity = cash + sum(last_val.get(m, v["cost"]) for m, v in open_pos.items())
        curve.append(equity)
    # anything still open is marked at its last executable value (or zero if never valued)
    end_equity = cash + sum(last_val.get(m, 0.0) for m in open_pos)
    mults = np.array([t.mult for t in trades]) if trades else np.array([])
    return {
        "coins": len(coins), "trades": len(trades), "open_at_end": len(open_pos),
        "total_return_pct": round(100 * (end_equity / start_equity - 1), 3),
        "end_equity": round(end_equity, 2), "max_drawdown_pct": round(100 * max_drawdown(curve), 2) if curve else 0,
        "win_rate": float((mults > 1).mean()) if len(mults) else None,
        "median_mult": float(np.median(mults)) if len(mults) else None,
        "best_mult": float(mults.max()) if len(mults) else None,
        "exit_reasons": {r: sum(1 for t in trades if t.reason == r) for r in {t.reason for t in trades}},
        "trades_detail": [asdict(t) for t in trades[:200]],
    }


GRID = {"min_chg_h1": [5.0, 25.0], "max_rug_score": [40.0, 70.0], "take_profit_mult": [1.5, 2.5],
        "stop_loss_mult": [0.5, 0.75]}


def walk_forward(coins: list[Coin], start_equity: float = 1000.0, train_frac: float = 0.6,
                 min_train_trades: int = 10) -> dict:
    cut = int(len(coins) * train_frac)
    train, test = coins[:cut], coins[cut:]
    res = []
    keys = list(GRID)
    for combo in itertools.product(*(GRID[k] for k in keys)):
        p = MemeParams(**dict(zip(keys, combo)))
        s = simulate(train, p, start_equity)
        res.append((p, s))
    ok = [(p, s) for p, s in res if s["trades"] >= min_train_trades]
    summary = {"coins_total": len(coins), "train_coins": len(train), "test_coins": len(test),
               "grid": [({k: getattr(p, k) for k in keys}, {k: s[k] for k in ("trades", "total_return_pct", "win_rate")})
                        for p, s in res]}
    if not ok:
        summary["note"] = "not enough training trades yet - keep collecting data"
        return summary
    best_p, best_s = max(ok, key=lambda ps: ps[1]["total_return_pct"])
    test_s = simulate(test, best_p, start_equity)
    summary.update(chosen={k: getattr(best_p, k) for k in keys},
                   train={k: v for k, v in best_s.items() if k != "trades_detail"},
                   test={k: v for k, v in test_s.items() if k != "trades_detail"})
    return summary
