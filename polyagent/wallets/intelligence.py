"""Wallet intelligence: reconstruct P&L from fills, compute risk-adjusted metrics,
flag suspicious / unreliable records, and rank.

The core function `compute_metrics` is pure and point-in-time: it only uses trades
with ts <= as_of and resolutions known at as_of, so the backtester can call the
exact same code without look-ahead.
"""
from __future__ import annotations

import logging
import math
import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field

import numpy as np

from ..analytics.stats import herfindahl, max_drawdown_abs, wilson_lower_bound

log = logging.getLogger(__name__)


@dataclass
class Trade:
    wallet: str
    token_id: str
    condition_id: str
    side: str  # BUY / SELL
    size: float
    price: float
    ts: int

    @classmethod
    def from_api(cls, d: dict) -> "Trade":
        return cls(
            wallet=(d.get("proxy_wallet") or d.get("wallet") or "").lower(),
            token_id=str(d.get("token_id") or d.get("asset") or ""),
            condition_id=d.get("condition_id") or d.get("conditionId") or "",
            side=(d.get("side") or "").upper(),
            size=float(d.get("size") or 0),
            price=float(d.get("price") or 0),
            ts=int(d.get("timestamp") or d.get("ts") or 0),
        )


@dataclass
class WalletMetrics:
    wallet: str
    n_trades: int = 0
    n_markets: int = 0
    n_resolved: int = 0  # markets closed out (resolved or fully exited)
    days_active: float = 0.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    total_cost: float = 0.0
    roi: float = 0.0
    win_rate: float = 0.0
    win_rate_lcb: float = 0.0
    max_drawdown: float = 0.0  # USDC, on realized P&L curve
    trades_per_day: float = 0.0
    avg_hold_hours: float = 0.0
    avg_trade_usd: float = 0.0
    size_cv: float = 0.0  # coefficient of variation of trade notional
    avg_entry_price: float = 0.0
    avg_entry_edge: float = 0.0  # mean(payout - entry) on resolved buys, notional-weighted
    entry_hours_before_close: float = 0.0
    top_market_share: float = 0.0  # share of gross positive P&L from best market
    hhi: float = 0.0  # concentration of notional across markets
    sharpe: float = 0.0  # mean/std of per-market returns * sqrt(n)
    mean_market_return: float = 0.0
    return_lcb: float = 0.0  # shrunk lower bound of per-market return
    score: float = 0.0
    reliable: bool = False
    flags: list[str] = field(default_factory=list)

    def as_row(self) -> dict:
        d = asdict(self)
        d["flags"] = ",".join(self.flags)
        return d


@dataclass
class _Lot:
    size: float
    price: float
    ts: int


def compute_metrics(wallet: str, trades: list[Trade], resolutions: dict[str, float],
                    marks: dict[str, float] | None = None, as_of: int | None = None,
                    resolution_ts: dict[str, int] | None = None, *, min_trades: int = 50,
                    min_resolved: int = 15, min_days: int = 30,
                    max_market_share: float = 0.5, shrink_n0: float = 10.0) -> WalletMetrics:
    """FIFO P&L reconstruction plus performance statistics.

    resolutions: token_id -> payout per share (1.0 or 0.0) for markets resolved by as_of.
    marks: token_id -> current price for unrealized P&L.
    resolution_ts: token_id -> unix ts of resolution (for holding period / drawdown timing).
    """
    as_of = as_of or int(time.time())
    marks = marks or {}
    resolution_ts = resolution_ts or {}
    m = WalletMetrics(wallet=wallet)
    trades = sorted((t for t in trades if t.ts <= as_of and t.size > 0), key=lambda t: t.ts)
    m.n_trades = len(trades)
    if not trades:
        m.flags.append("no_trades")
        return m

    lots: dict[str, deque[_Lot]] = defaultdict(deque)
    token_market: dict[str, str] = {}
    market_pnl: dict[str, float] = defaultdict(float)
    market_cost: dict[str, float] = defaultdict(float)
    market_notional: dict[str, float] = defaultdict(float)
    pnl_events: list[tuple[int, float]] = []
    holds: list[tuple[float, float]] = []  # (hours, size)
    entry_edges: list[tuple[float, float]] = []  # (edge, notional)
    entry_lead: list[float] = []
    notionals: list[float] = []
    buy_prices: list[tuple[float, float]] = []
    exited_markets: set[str] = set()

    for t in trades:
        token_market[t.token_id] = t.condition_id
        notional = t.size * t.price
        notionals.append(notional)
        market_notional[t.condition_id] += notional
        if t.side == "BUY":
            lots[t.token_id].append(_Lot(t.size, t.price, t.ts))
            market_cost[t.condition_id] += notional
            buy_prices.append((t.price, notional))
            if t.token_id in resolutions:
                entry_edges.append((resolutions[t.token_id] - t.price, notional))
                if t.token_id in resolution_ts:
                    entry_lead.append((resolution_ts[t.token_id] - t.ts) / 3600)
        elif t.side == "SELL":
            remaining = t.size
            pnl = 0.0
            q = lots[t.token_id]
            while remaining > 1e-9 and q:
                lot = q[0]
                take = min(lot.size, remaining)
                pnl += take * (t.price - lot.price)
                holds.append(((t.ts - lot.ts) / 3600, take))
                lot.size -= take
                remaining -= take
                if lot.size <= 1e-9:
                    q.popleft()
            # Shares sold beyond known lots (history truncated / acquired via split):
            # ignored for P&L but flagged.
            if remaining > 1e-6 * max(t.size, 1):
                if "truncated_history" not in m.flags:
                    m.flags.append("truncated_history")
            if pnl:
                market_pnl[t.condition_id] += pnl
                pnl_events.append((t.ts, pnl))
            if not q:
                exited_markets.add(t.condition_id)

    # Resolution settlement and mark-to-market of remaining lots.
    resolved_markets: set[str] = set()
    for token, q in lots.items():
        cid = token_market.get(token, "")
        if token in resolutions:
            payout = resolutions[token]
            rts = resolution_ts.get(token, as_of)
            pnl = 0.0
            for lot in q:
                pnl += lot.size * (payout - lot.price)
                holds.append(((rts - lot.ts) / 3600, lot.size))
            if pnl:
                market_pnl[cid] += pnl
                pnl_events.append((rts, pnl))
            resolved_markets.add(cid)
        elif q and token in marks:
            m.unrealized_pnl += sum(lot.size * (marks[token] - lot.price) for lot in q)

    # A market counts as "closed out" if resolved, or fully exited with no open lots
    open_markets = {token_market[tk] for tk, q in lots.items() if q and tk not in resolutions}
    closed = (resolved_markets | exited_markets) - open_markets
    closed = {c for c in closed if market_cost.get(c, 0) > 0}

    m.n_markets = len(market_notional)
    m.n_resolved = len(closed)
    m.realized_pnl = float(sum(market_pnl[c] for c in market_pnl))
    m.total_cost = float(sum(market_cost.values()))
    m.roi = (m.realized_pnl + m.unrealized_pnl) / m.total_cost if m.total_cost > 0 else 0.0
    span_days = max((trades[-1].ts - trades[0].ts) / 86400, 1 / 24)
    m.days_active = span_days
    m.trades_per_day = m.n_trades / span_days

    wins = sum(1 for c in closed if market_pnl[c] > 0)
    m.win_rate = wins / len(closed) if closed else 0.0
    m.win_rate_lcb = wilson_lower_bound(wins, len(closed))

    pnl_events.sort()
    m.max_drawdown = max_drawdown_abs(np.cumsum([p for _, p in pnl_events])) if pnl_events else 0.0

    if holds:
        w = sum(s for _, s in holds)
        m.avg_hold_hours = sum(h * s for h, s in holds) / w if w else 0.0
    m.avg_trade_usd = float(np.mean(notionals))
    m.size_cv = float(np.std(notionals) / m.avg_trade_usd) if m.avg_trade_usd > 0 else 0.0
    if buy_prices:
        w = sum(n for _, n in buy_prices)
        m.avg_entry_price = sum(p * n for p, n in buy_prices) / w if w else 0.0
    if entry_edges:
        w = sum(n for _, n in entry_edges)
        m.avg_entry_edge = sum(e * n for e, n in entry_edges) / w if w else 0.0
    if entry_lead:
        m.entry_hours_before_close = float(np.median(entry_lead))

    m.hhi = herfindahl(list(market_notional.values()))
    entries = {c: _avg_entry(trades, c) for c in closed}
    _return_stats(m, {c: market_pnl[c] for c in closed}, {c: market_cost[c] for c in closed}, entries, shrink_n0)
    _apply_gates(m, min_trades, min_resolved, min_days, max_market_share)
    return m


def _return_stats(m: WalletMetrics, pnl: dict[str, float], cost: dict[str, float],
                  entry: dict[str, float], shrink_n0: float) -> None:
    """Per-closed-market returns -> win rate, concentration, shrunk risk-adjusted score."""
    closed = [c for c in pnl if cost.get(c, 0) > 0]
    m.n_resolved = len(closed)
    wins = sum(1 for c in closed if pnl[c] > 0)
    m.win_rate = wins / len(closed) if closed else 0.0
    m.win_rate_lcb = wilson_lower_bound(wins, len(closed))
    positive = [pnl[c] for c in closed if pnl[c] > 0]
    m.top_market_share = max(positive) / sum(positive) if positive else 0.0
    rets = np.array([pnl[c] / cost[c] for c in closed])
    m.sharpe = m.mean_market_return = m.return_lcb = 0.0
    if len(rets) >= 2:
        mu, sd = float(rets.mean()), float(rets.std(ddof=1))
        m.mean_market_return = mu
        m.sharpe = mu / sd * math.sqrt(len(rets)) if sd > 1e-9 else 0.0
        shrunk = mu * len(rets) / (len(rets) + shrink_n0)  # shrink toward zero edge
        se = sd / math.sqrt(len(rets))
        m.return_lcb = shrunk - 1.645 * se  # one-sided 95%
    m.score = m.return_lcb * math.log1p(m.n_resolved)
    total = sum(pnl[c] for c in closed)
    lottery = [pnl[c] for c in closed if pnl[c] > 0 and entry.get(c, 1.0) < 0.1]
    m._lottery = bool(total > 0 and lottery and max(lottery) > 0.5 * total)  # type: ignore[attr-defined]


def _apply_gates(m: WalletMetrics, min_trades: int, min_resolved: int, min_days: float,
                 max_market_share: float) -> None:
    keep = {"no_trades", "truncated_history", "history_capped", "pnl_from_positions"}
    m.flags = [f for f in m.flags if f in keep]
    if m.n_trades < min_trades:
        m.flags.append("few_trades")
    if m.n_resolved < min_resolved:
        m.flags.append("few_resolved_markets")
    if m.days_active < min_days:
        m.flags.append("short_history")
    if m.top_market_share > max_market_share and m.n_resolved >= 3:
        m.flags.append("concentrated_returns")
    if m.win_rate > 0.9 and m.n_resolved >= 10:
        m.flags.append("implausible_win_rate")
    if m.avg_entry_price > 0.9:
        m.flags.append("buys_near_certainties")  # tiny edge, tail risk
    if m.trades_per_day > 200:
        m.flags.append("bot_or_market_maker")  # copying an MM's fills is meaningless
    if getattr(m, "_lottery", False):
        m.flags.append("lottery_driven")
    hard = {"few_trades", "few_resolved_markets", "short_history", "concentrated_returns",
            "implausible_win_rate", "bot_or_market_maker", "lottery_driven", "no_trades"}
    m.reliable = not (hard & set(m.flags)) and m.return_lcb > 0


def apply_closed_positions(m: WalletMetrics, closed_positions: list[dict], *, min_trades: int = 50,
                           min_resolved: int = 15, min_days: int = 30, max_market_share: float = 0.5,
                           shrink_n0: float = 10.0) -> WalletMetrics:
    """Replace trade-reconstructed P&L statistics with Polymarket's own per-position
    realized P&L (computed from the wallet's full history, so immune to our trade-history
    cap). Timing/sizing/frequency metrics still come from the trade tape."""
    pnl: dict[str, float] = defaultdict(float)
    cost: dict[str, float] = defaultdict(float)
    entry: dict[str, float] = {}
    ends = []
    for p in closed_positions:
        cid = p.get("condition_id") or str(p.get("token_id"))
        pnl[cid] += float(p.get("realized_pnl") or 0)
        # total_cost_usdc is the *residual* basis of a closed position (~0); the historical
        # basis is total bought x average price (matches the API's percent_realized_pnl).
        cost[cid] += float(p.get("total_size") or 0) * float(p.get("avg_price") or 0)
        entry[cid] = float(p.get("avg_price") or 1.0)
        if p.get("end_date"):
            try:
                ends.append(time.mktime(time.strptime(p["end_date"][:10], "%Y-%m-%d")))
            except ValueError:
                pass
    if not pnl:
        return m
    m.realized_pnl = float(sum(pnl.values()))
    if len(ends) >= 2:
        m.days_active = max(m.days_active, (max(ends) - min(ends)) / 86400)
    _return_stats(m, dict(pnl), dict(cost), entry, shrink_n0)
    m.roi = (m.realized_pnl + m.unrealized_pnl) / max(sum(cost.values()), 1e-9)
    if "pnl_from_positions" not in m.flags:
        m.flags.append("pnl_from_positions")
    _apply_gates(m, min_trades, min_resolved, min_days, max_market_share)
    return m


def _avg_entry(trades: list[Trade], cid: str) -> float:
    buys = [(t.price, t.size) for t in trades if t.condition_id == cid and t.side == "BUY"]
    w = sum(s for _, s in buys)
    return sum(p * s for p, s in buys) / w if w else 1.0


def resolutions_from_positions(positions: list[dict]) -> tuple[dict, dict]:
    """Derive (resolutions, marks) from Data API v2 /positions rows."""
    resolutions: dict[str, float] = {}
    marks: dict[str, float] = {}
    for p in positions:
        tok = str(p.get("token_id", ""))
        price = p.get("current_price")
        if price is None:
            continue
        price = float(price)
        if price in (0.0, 1.0) and (p.get("redeemable") or (p.get("status") or "").upper() == "CLOSED"):
            resolutions[tok] = price
            opp = p.get("opposite_token_id")
            if opp:
                resolutions[str(opp)] = 1.0 - price
        else:
            marks[tok] = price
    return resolutions, marks


class WalletRanker:
    """Fetches wallets, scores them, persists to wallet_scores."""

    def __init__(self, api, db, cfg):
        self.api = api
        self.db = db
        self.cfg = cfg

    def candidate_wallets(self) -> list[str]:
        cands: dict[str, None] = {}
        # Leaderboard seeds are survivorship-biased (only winners appear); we also
        # sample active traders from the public tape and rely on shrinkage + gates.
        for period in ("month", "all"):
            try:
                for r in self.api.leaderboard(period=period, n=self.cfg.wallet_seed_count // 2):
                    cands[r["user_id"].lower()] = None
            except Exception as e:  # noqa: BLE001
                log.warning("leaderboard fetch failed: %s", e)
        try:
            for t in self.api.recent_trades(max_items=1000):
                w = (t.get("proxy_wallet") or "").lower()
                if w and float(t.get("size", 0)) * float(t.get("price", 0)) >= 100:
                    cands[w] = None
        except Exception as e:  # noqa: BLE001
            log.warning("tape sample failed: %s", e)
        return list(cands)[: self.cfg.wallet_seed_count]

    def score_wallet(self, wallet: str) -> WalletMetrics:
        raw = self.api.wallet_trades(wallet, max_items=2000)
        trades = [Trade.from_api(t) for t in raw]
        open_pos = self.api.wallet_positions(wallet, status="OPEN", max_items=500)
        closed_pos = self.api.wallet_positions(wallet, status="CLOSED", max_items=1000)
        resolutions, marks = resolutions_from_positions(open_pos + closed_pos)
        rows = [(t.get("transaction_hash", ""), wallet, str(t.get("token_id")), t.get("condition_id"),
                 t.get("side"), float(t.get("size", 0)), float(t.get("price", 0)),
                 int(t.get("timestamp", 0)), t.get("outcome_index")) for t in raw]
        if rows:
            self.db.executemany("INSERT OR IGNORE INTO wallet_trades VALUES (?,?,?,?,?,?,?,?,?)", rows)
        m = compute_metrics(
            wallet, trades, resolutions, marks,
            min_trades=self.cfg.wallet_min_trades, min_resolved=self.cfg.wallet_min_resolved_markets,
            min_days=self.cfg.wallet_min_days_active, max_market_share=self.cfg.wallet_max_market_share)
        if len(raw) >= 2000:
            m.flags.append("history_capped")
        c = self.cfg
        return apply_closed_positions(m, closed_pos, min_trades=c.wallet_min_trades,
                                      min_resolved=c.wallet_min_resolved_markets, min_days=c.wallet_min_days_active,
                                      max_market_share=c.wallet_max_market_share)

    def refresh(self, wallets: list[str] | None = None) -> list[WalletMetrics]:
        wallets = wallets or self.candidate_wallets()
        out = []
        for i, w in enumerate(wallets):
            try:
                m = self.score_wallet(w)
            except Exception as e:  # noqa: BLE001
                log.warning("wallet %s failed: %s", w, e)
                continue
            out.append(m)
            self.persist(m)
            if (i + 1) % 25 == 0:
                log.info("scored %d/%d wallets", i + 1, len(wallets))
        self.db.audit("wallet_refresh", {"scored": len(out), "reliable": sum(m.reliable for m in out)})
        return out

    def persist(self, m: WalletMetrics) -> None:
        self.db.execute(
            """INSERT INTO wallet_scores(wallet,computed_at,n_trades,n_markets,n_resolved,days_active,
               realized_pnl,unrealized_pnl,roi,win_rate,win_rate_lcb,max_drawdown,trades_per_day,
               avg_hold_hours,top_market_share,hhi,sharpe,score,reliable,flags)
               VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(wallet) DO UPDATE SET computed_at=excluded.computed_at,
               n_trades=excluded.n_trades,n_markets=excluded.n_markets,n_resolved=excluded.n_resolved,
               days_active=excluded.days_active,realized_pnl=excluded.realized_pnl,
               unrealized_pnl=excluded.unrealized_pnl,roi=excluded.roi,win_rate=excluded.win_rate,
               win_rate_lcb=excluded.win_rate_lcb,max_drawdown=excluded.max_drawdown,
               trades_per_day=excluded.trades_per_day,avg_hold_hours=excluded.avg_hold_hours,
               top_market_share=excluded.top_market_share,hhi=excluded.hhi,sharpe=excluded.sharpe,
               score=excluded.score,reliable=excluded.reliable,flags=excluded.flags""",
            (m.wallet, time.time(), m.n_trades, m.n_markets, m.n_resolved, m.days_active,
             m.realized_pnl, m.unrealized_pnl, m.roi, m.win_rate, m.win_rate_lcb, m.max_drawdown,
             m.trades_per_day, m.avg_hold_hours, m.top_market_share, m.hhi, m.sharpe, m.score,
             int(m.reliable), ",".join(m.flags)))

    def top_wallets(self, n: int | None = None) -> dict[str, float]:
        """wallet -> weight in (0, 1], reliable wallets only."""
        rows = self.db.query(
            "SELECT wallet, score, win_rate_lcb FROM wallet_scores WHERE reliable=1 "
            "ORDER BY score DESC LIMIT ?", (n or self.cfg.wallet_top_n,))
        if not rows:
            return {}
        top = max(r["score"] for r in rows) or 1.0
        return {r["wallet"]: max(min(r["score"] / top, 1.0), 0.05) for r in rows}
