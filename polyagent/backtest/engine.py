"""Event-driven, point-in-time backtester with walk-forward parameter selection.

Look-ahead protections:
  * Strategies receive only price points with t <= decision time.
  * Wallet rankings at time t use trades with ts <= t, scored ONLY on markets whose
    resolution timestamp <= t (outcomes not yet known are treated as unresolved).
  * Parameters are chosen on the earlier (train) slice of markets ordered by
    resolution time and reported on the later (test) slice untouched.

Known simplifications (see README): no historical order books exist publicly, so
execution cost = last trade/mid price + configurable half-spread + slippage + the
market's documented taker fee. Positions are held to resolution.
"""
from __future__ import annotations

import bisect
import datetime as dt
import heapq
import itertools
import json
import logging
import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from ..analytics.costs import taker_fee
from ..analytics.stats import brier_score, max_drawdown
from ..risk.manager import uncertainty_adjusted_kelly
from ..strategy.estimators import EstimatorParams, SmartMoneyEstimator, TrendEstimator, normal_prob_above
from ..wallets.intelligence import Trade, compute_metrics

log = logging.getLogger(__name__)


@dataclass
class MarketRecord:
    condition_id: str
    question: str
    event_id: str
    yes_token: str
    no_token: str
    start_ts: int
    resolve_ts: int
    outcome_yes: float  # 1.0 or 0.0
    fee_rate: float
    fee_exponent: float
    history: list[tuple[int, float]]  # YES price
    trades: list[Trade] = field(default_factory=list)

    def price_at(self, t: int) -> float | None:
        """Latest observed YES price at or before t, from the hourly history merged with the
        trade tape (the hourly series alone can lag fast moves by up to an hour)."""
        tape = self.__dict__.get("_tape")
        if tape is None:
            pts = dict(self.history)
            for tr in self.trades:
                px = tr.price if tr.token_id == self.yes_token else 1 - tr.price
                pts[tr.ts] = px
            tape = sorted(pts.items())
            self.__dict__["_tape"] = tape
        i = bisect.bisect_right(tape, (t, float("inf"))) - 1
        return tape[i][1] if i >= 0 else None

    def trades_between(self, lo: int, hi: int) -> list[Trade]:
        """Trades with lo <= ts <= hi (sorted index built lazily)."""
        idx = self.__dict__.get("_tidx")
        if idx is None:
            srt = sorted(self.trades, key=lambda tr: tr.ts)
            idx = ([tr.ts for tr in srt], srt)
            self.__dict__["_tidx"] = idx
        ts, srt = idx
        return srt[bisect.bisect_left(ts, lo):bisect.bisect_right(ts, hi)]

    def history_until(self, t: int) -> list[tuple[int, float]]:
        i = bisect.bisect_right(self.history, (t, float("inf")))
        return self.history[:i]

    def to_json(self) -> dict:
        d = {k: v for k, v in asdict(self).items() if not k.startswith("_")}
        d["trades"] = [asdict(t) for t in self.trades]
        return d

    @classmethod
    def from_json(cls, d: dict) -> "MarketRecord":
        d = dict(d)
        d["history"] = [tuple(x) for x in d["history"]]
        d["trades"] = [Trade(**t) for t in d.get("trades", [])]
        return cls(**d)


@dataclass
class BTParams:
    strategy: str = "smart_money"  # smart_money | trend
    min_edge: float = 0.03
    min_confidence: float = 0.8
    kelly_fraction: float = 0.25
    max_position_pct: float = 0.05
    min_cash_reserve_pct: float = 0.2
    half_spread: float = 0.01
    slippage: float = 0.005
    step_hours: int = 12
    min_hours_to_close: int = 6
    price_floor: float = 0.04
    price_ceiling: float = 0.96
    rank_every_hours: int = 72
    prior_concentration: float = 200.0
    per_wallet_pseudo: float = 6.0
    min_smart_wallets: int = 2
    wallet_min_trades: int = 10
    wallet_min_resolved: int = 3
    wallet_min_days: int = 7


@dataclass
class BTTrade:
    condition_id: str
    question: str
    entry_ts: int
    resolve_ts: int
    side: str  # YES/NO
    p_market: float
    p_est: float
    cost_per_share: float
    shares: float
    stake: float
    payout: float = 0.0
    pnl: float = 0.0


@dataclass
class BTResult:
    params: dict
    start_equity: float
    end_equity: float
    trades: list[BTTrade]
    equity_curve: list[tuple[int, float]]
    n_markets: int
    n_decisions: int

    @property
    def total_return(self) -> float:
        return self.end_equity / self.start_equity - 1

    def summary(self) -> dict:
        pnls = [t.pnl for t in self.trades]
        wins = sum(1 for p in pnls if p > 0)
        eq = [e for _, e in self.equity_curve] or [self.start_equity]
        ests = [t.p_est for t in self.trades]
        mkts = [t.p_market for t in self.trades]
        outs = [1.0 if t.payout > 0 else 0.0 for t in self.trades]
        # bootstrap CI on mean per-trade return, so a lucky handful of trades is visible as such
        rets = np.array([t.pnl / t.stake for t in self.trades if t.stake > 0])
        ci = (float("nan"), float("nan"))
        if len(rets) >= 5:
            rng = np.random.default_rng(0)
            boots = [rng.choice(rets, len(rets)).mean() for _ in range(2000)]
            ci = (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))
        return {
            "markets": self.n_markets, "decisions": self.n_decisions, "trades": len(self.trades),
            "win_rate": wins / len(pnls) if pnls else None,
            "total_return_pct": round(100 * self.total_return, 3),
            "pnl_usd": round(self.end_equity - self.start_equity, 2),
            "max_drawdown_pct": round(100 * max_drawdown(eq), 3),
            "mean_trade_return": float(rets.mean()) if len(rets) else None,
            "mean_trade_return_95ci": ci,
            "brier_ours": brier_score(ests, outs) if ests else None,
            "brier_market": brier_score(mkts, outs) if mkts else None,
            "avg_claimed_edge": float(np.mean([t.p_est - t.cost_per_share for t in self.trades])) if self.trades else None,
        }


class Backtester:
    def __init__(self, markets: list[MarketRecord], start_equity: float = 1000.0):
        self.markets = sorted(markets, key=lambda m: m.resolve_ts)
        self.start_equity = start_equity
        self._weights_cache: dict[tuple, dict[str, float]] = {}

    def weights_at(self, t: int, p: BTParams) -> dict[str, float]:
        """Ranking as of the start of t's ranking bucket (bucket start <= t, so still point-in-time)."""
        step = p.rank_every_hours * 3600
        tb = (t // step) * step
        key = (tb, p.wallet_min_trades, p.wallet_min_resolved, p.wallet_min_days)
        if key not in self._weights_cache:
            self._weights_cache[key] = self._wallet_weights(tb, p)
        return self._weights_cache[key]

    # --- point-in-time wallet ranking -------------------------------------
    def _wallet_weights(self, t: int, p: BTParams, exclude: str | None = None) -> dict[str, float]:
        resolved = [m for m in self.markets if m.resolve_ts <= t and m.condition_id != exclude]
        if not resolved:
            return {}
        resolutions, res_ts = {}, {}
        by_wallet: dict[str, list[Trade]] = {}
        for m in resolved:
            resolutions[m.yes_token] = m.outcome_yes
            resolutions[m.no_token] = 1 - m.outcome_yes
            res_ts[m.yes_token] = res_ts[m.no_token] = m.resolve_ts
            for tr in m.trades_between(-1, t):
                by_wallet.setdefault(tr.wallet, []).append(tr)
        scored = []
        for w, trs in by_wallet.items():
            if len(trs) < p.wallet_min_trades:
                continue
            met = compute_metrics(w, trs, resolutions, as_of=t, resolution_ts=res_ts,
                                  min_trades=p.wallet_min_trades, min_resolved=p.wallet_min_resolved,
                                  min_days=p.wallet_min_days)
            if met.reliable:
                scored.append((w, met.score))
        if not scored:
            return {}
        scored.sort(key=lambda x: -x[1])
        top = scored[0][1] or 1.0
        return {w: max(min(s / top, 1.0), 0.05) for w, s in scored[:50]}

    # --- simulation ---------------------------------------------------------
    def run(self, p: BTParams, markets: list[MarketRecord] | None = None) -> BTResult:
        markets = markets if markets is not None else self.markets
        ep = EstimatorParams(prior_concentration=p.prior_concentration, per_wallet_pseudo=p.per_wallet_pseudo,
                             min_smart_wallets=p.min_smart_wallets)
        smart, trend = SmartMoneyEstimator(ep), TrendEstimator(ep)
        # schedule decisions across all markets chronologically
        events: list[tuple[int, int, str, MarketRecord]] = []
        seq = itertools.count()
        for m in markets:
            if not m.history:
                continue
            t = m.history[0][0] + 24 * 3600
            while t < m.resolve_ts - p.min_hours_to_close * 3600:
                heapq.heappush(events, (t, next(seq), "decide", m))
                t += p.step_hours * 3600
        cash = self.start_equity
        open_pos: dict[str, BTTrade] = {}
        done: list[BTTrade] = []
        curve: list[tuple[int, float]] = []
        n_dec = 0
        while events:
            t, _, kind, m = heapq.heappop(events)
            if kind == "settle":
                tr = open_pos.pop(m.condition_id)
                won = (m.outcome_yes == 1.0) == (tr.side == "YES")
                tr.payout = tr.shares if won else 0.0
                tr.pnl = tr.payout - tr.stake
                cash += tr.payout
                done.append(tr)
                curve.append((t, cash + sum(o.stake for o in open_pos.values())))
                continue
            if m.condition_id in open_pos:
                continue
            n_dec += 1
            price = m.price_at(t)
            if price is None or not (p.price_floor <= price <= p.price_ceiling):
                continue
            if p.strategy == "smart_money":
                weights = self.weights_at(t, p)
                recent = m.trades_between(t - 24 * 3600, t)
                est = smart.estimate(price, m.yes_token, m.no_token, recent, weights) if weights else None
            else:
                est = trend.estimate(m.history_until(t), t)
            if est is None:
                continue
            best = None
            for side, p_true, px in (("YES", est.p, price), ("NO", 1 - est.p, 1 - price)):
                fill_px = min(px + p.half_spread + p.slippage, 0.99)
                cps = fill_px + taker_fee(1, fill_px, m.fee_rate, m.fee_exponent)
                edge = p_true - cps
                conf = normal_prob_above(p_true, est.std, cps)
                if edge >= p.min_edge and conf >= p.min_confidence and (best is None or edge > best[3]):
                    best = (side, p_true, px, edge, cps)
            if not best:
                continue
            side, p_true, px, edge, cps = best
            equity = cash + sum(o.stake for o in open_pos.values())  # cost-basis equity (no MTM)
            f = uncertainty_adjusted_kelly(p_true, est.std, cps, p.kelly_fraction)
            stake = min(f * equity, p.max_position_pct * equity, cash - p.min_cash_reserve_pct * equity)
            if stake < 1.0:
                continue
            cash -= stake
            open_pos[m.condition_id] = BTTrade(m.condition_id, m.question, t, m.resolve_ts, side, px, p_true,
                                               cps, stake / cps, stake)
            heapq.heappush(events, (m.resolve_ts, next(seq), "settle", m))
        return BTResult(asdict(p), self.start_equity, cash, done, curve, len(markets), n_dec)

    def walk_forward(self, grid: dict[str, list], base: BTParams, train_frac: float = 0.6,
                     min_train_trades: int = 5) -> dict:
        n = len(self.markets)
        cut = int(n * train_frac)
        train, test = self.markets[:cut], self.markets[cut:]
        results = []
        keys = list(grid)
        for combo in itertools.product(*(grid[k] for k in keys)):
            p = BTParams(**{**asdict(base), **dict(zip(keys, combo))})
            r = self.run(p, train)
            s = r.summary()
            results.append((p, s))
        eligible = [(p, s) for p, s in results if s["trades"] >= min_train_trades]
        if not eligible:
            return {"train_markets": len(train), "test_markets": len(test), "chosen": None,
                    "note": "no parameter set produced enough trades on the training slice",
                    "grid": [(asdict(p), s) for p, s in results]}
        chosen, train_s = max(eligible, key=lambda ps: ps[1]["total_return_pct"])
        test_r = self.run(chosen, test)
        return {"train_markets": len(train), "test_markets": len(test), "chosen": asdict(chosen),
                "train": train_s, "test": test_r.summary(),
                "test_trades": [asdict(t) for t in test_r.trades],
                "grid": [({k: getattr(p, k) for k in keys}, s) for p, s in results]}


# --- dataset construction --------------------------------------------------

def _ts(iso: str | None) -> int | None:
    if not iso:
        return None
    s = iso.strip().replace("Z", "+00:00").replace(" ", "T")
    if re.search(r"[+-]\d\d$", s):
        s += ":00"
    try:
        d = dt.datetime.fromisoformat(s)
    except ValueError:
        return None
    if d.tzinfo is None:
        d = d.replace(tzinfo=dt.timezone.utc)
    return int(d.timestamp())


def build_dataset(api, n_markets: int, end_date_min: str, end_date_max: str, cache: Path,
                  max_trades: int = 3000, max_days: int = 60, min_trade_usd: float = 200.0,
                  workers: int = 6) -> list[MarketRecord]:
    """Fetch resolved binary markets, their YES price history and large public trades.
    Cached to `cache` (JSON) so repeated backtests are reproducible offline."""
    from concurrent.futures import ThreadPoolExecutor

    if cache.exists():
        return [MarketRecord.from_json(d) for d in json.loads(cache.read_text())]
    cands = []
    for m in api.closed_markets(limit=n_markets * 3, end_date_min=end_date_min, end_date_max=end_date_max):
        if len(m.token_ids) != 2 or set(m.outcome_prices) != {0.0, 1.0}:
            continue
        resolve = _ts(m.raw.get("closedTime")) or _ts(m.end_date)
        start = _ts(m.raw.get("startDate")) or (resolve - 30 * 86400 if resolve else None)
        if resolve and start:
            cands.append((m, max(start, resolve - max_days * 86400), resolve))
    cands = cands[: int(n_markets * 1.3)]

    def fetch(item):
        m, start, resolve = item
        try:
            hist = api.price_history_range(m.yes_token, start, resolve, fidelity=60)
            if len(hist) < 48:
                return None
            trades = [Trade.from_api(t) for t in api.market_trades(m.condition_id, max_items=max_trades,
                                                                    min_usd=min_trade_usd)]
        except Exception as e:  # noqa: BLE001
            log.warning("dataset fetch %s failed: %s", m.condition_id, e)
            return None
        return MarketRecord(m.condition_id, m.question, m.event_id, m.yes_token, m.no_token, start, resolve,
                            m.outcome_prices[0], m.fee_rate, m.fee_exponent, hist, trades)

    out: list[MarketRecord] = []
    with ThreadPoolExecutor(workers) as ex:
        for rec in ex.map(fetch, cands):
            if rec is not None and len(out) < n_markets:
                out.append(rec)
                if len(out) % 25 == 0:
                    log.info("dataset %d/%d", len(out), n_markets)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps([r.to_json() for r in out]))
    return out


def market_calibration(markets: list[MarketRecord], hours_before: int = 24) -> dict:
    """How well-calibrated was the market price `hours_before` resolution? A low Brier
    score here means there is little room for an independent estimate to add value."""
    ps, os_ = [], []
    for m in markets:
        p = m.price_at(m.resolve_ts - hours_before * 3600)
        if p is not None:
            ps.append(p)
            os_.append(m.outcome_yes)
    if not ps:
        return {}
    return {"n": len(ps), "brier": brier_score(ps, os_), "hours_before": hours_before,
            "naive_brier_0.5": brier_score([0.5] * len(ps), os_),
            "log_loss": float(-np.mean([o * math.log(max(p, 1e-4)) + (1 - o) * math.log(max(1 - p, 1e-4))
                                        for p, o in zip(ps, os_)]))}


def flow_signal_study(bt: "Backtester", p: BTParams, smart_only: bool = True, step_hours: int = 12,
                      lookback_hours: int = 24) -> dict:
    """Does the direction of (smart) wallet net flow predict outcome minus price?

    For every decision point with flow, record sign(flow) * (outcome_yes - price). Points are
    averaged per *event* (markets in one event are correlated) and the t-stat is computed
    across events. Wallet rankings are point-in-time (see Backtester.weights_at).
    """
    per_event: dict[str, list[float]] = {}
    for m in bt.markets:
        if not m.history:
            continue
        t = m.history[0][0] + 24 * 3600
        while t < m.resolve_ts - p.min_hours_to_close * 3600:
            w = bt.weights_at(t, p) if smart_only else None
            if smart_only and not w:
                t += step_hours * 3600
                continue
            flow = 0.0
            for tr in m.trades_between(t - lookback_hours * 3600, t):
                if w is None or tr.wallet in w:
                    s = 1.0 if tr.side == "BUY" else -1.0
                    flow += (s if tr.token_id == m.yes_token else -s) * tr.size * tr.price
            price = m.price_at(t)
            if abs(flow) > 1 and price is not None and p.price_floor <= price <= p.price_ceiling:
                per_event.setdefault(m.event_id, []).append(math.copysign(1, flow) * (m.outcome_yes - price))
            t += step_hours * 3600
    v = np.array([np.mean(x) for x in per_event.values()])
    if len(v) < 3:
        return {"events": int(len(v)), "note": "too few events"}
    se = float(v.std(ddof=1) / math.sqrt(len(v)))
    return {"events": int(len(v)), "points": int(sum(len(x) for x in per_event.values())),
            "mean_signed_residual": float(v.mean()), "se": se, "t_stat": float(v.mean() / se) if se else None}
