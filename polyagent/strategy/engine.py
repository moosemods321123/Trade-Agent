"""Turns estimates + live order books into tradeable, cost-aware opportunities."""
from __future__ import annotations

from dataclasses import dataclass, field

from ..analytics.costs import taker_fee, walk_book
from ..api.polymarket import Market, OrderBook
from .estimators import Estimate, normal_prob_above


@dataclass
class Opportunity:
    kind: str  # "directional" | "arbitrage"
    strategy: str
    condition_id: str
    event_id: str
    token_id: str
    side: str  # always BUY for entries
    outcome: str
    p_market: float  # mid of the token being bought
    p_est: float  # estimated prob that the bought token pays $1
    p_std: float
    cost_per_share: float  # all-in incl. fees & slippage at probe size
    limit_price: float  # worst acceptable price per share (excl. fees)
    edge: float  # p_est - cost_per_share
    confidence: float  # P(true prob > cost_per_share)
    legs: list[dict] = field(default_factory=list)  # arbitrage legs
    detail: dict = field(default_factory=dict)
    reject_reason: str = ""

    @property
    def accepted(self) -> bool:
        return not self.reject_reason


def evaluate_directional(est: Estimate, market: Market, yes_book: OrderBook, no_book: OrderBook,
                         probe_usd: float, min_edge: float, min_confidence: float,
                         max_spread: float) -> Opportunity | None:
    """Decide whether to buy YES, buy NO, or do nothing, using executable asks."""
    candidates = []
    for side_name, book, p_true, outcome_idx in (
        ("YES", yes_book, est.p, 0), ("NO", no_book, 1 - est.p, 1)):
        if not book.asks or book.mid is None:
            continue
        fill = walk_book(book.asks, "BUY", usd=probe_usd, fee_rate=market.fee_rate,
                         fee_exponent=market.fee_exponent)
        if fill.shares <= 0:
            continue
        cps = (fill.cost + fill.fees) / fill.shares
        edge = p_true - cps
        conf = normal_prob_above(p_true, est.std, cps)
        # allowable limit: the highest price whose all-in cost still clears min_edge
        lim = _max_price_for_edge(p_true, min_edge, market.fee_rate, market.fee_exponent)
        outcome = market.outcomes[outcome_idx] if len(market.outcomes) > outcome_idx else side_name
        opp = Opportunity(
            kind="directional", strategy=est.strategy, condition_id=market.condition_id,
            event_id=market.event_id, token_id=book.token_id, side="BUY", outcome=outcome,
            p_market=book.mid, p_est=p_true, p_std=est.std, cost_per_share=cps,
            limit_price=min(lim, 0.99), edge=edge, confidence=conf, detail=dict(est.detail))
        if book.spread is not None and book.spread > max_spread:
            opp.reject_reason = f"spread {book.spread:.3f} > {max_spread}"
        elif edge < min_edge:
            opp.reject_reason = f"edge {edge:.4f} < {min_edge}"
        elif conf < min_confidence:
            opp.reject_reason = f"confidence {conf:.2f} < {min_confidence}"
        elif not fill.fully_filled:
            opp.reject_reason = "insufficient depth at probe size"
        candidates.append(opp)
    if not candidates:
        return None
    candidates.sort(key=lambda o: (o.accepted, o.edge), reverse=True)
    return candidates[0]


def _max_price_for_edge(p_true: float, min_edge: float, rate: float, exponent: float) -> float:
    """Largest price x with x + fee_per_share(x) <= p_true - min_edge (bisection)."""
    target = p_true - min_edge
    lo, hi = 0.0, 1.0
    for _ in range(40):
        mid = (lo + hi) / 2
        cost = mid + (rate * (mid * (1 - mid)) ** exponent if rate else 0.0)
        if cost <= target:
            lo = mid
        else:
            hi = mid
    return round(lo, 4)


def evaluate_binary_arbitrage(market: Market, yes_book: OrderBook, no_book: OrderBook,
                              sets_usd: float, min_profit: float) -> Opportunity | None:
    """Complete-set arbitrage: buy YES + NO for < $1 all-in; one of them pays $1.

    Risks that remain (documented, not ignored): leg risk (orders are not atomic),
    resolution/oracle disputes, capital locked until resolution or merge.
    """
    if not yes_book.asks or not no_book.asks:
        return None
    # size: limited by the thinner side, sweep level by level
    target_sets = sets_usd  # ~$1 per set
    y = walk_book(yes_book.asks, "BUY", shares=target_sets, fee_rate=market.fee_rate,
                  fee_exponent=market.fee_exponent)
    n = walk_book(no_book.asks, "BUY", shares=target_sets, fee_rate=market.fee_rate,
                  fee_exponent=market.fee_exponent)
    sets = min(y.shares, n.shares)
    if sets < max(market.min_size, 1):
        return None
    y = walk_book(yes_book.asks, "BUY", shares=sets, fee_rate=market.fee_rate, fee_exponent=market.fee_exponent)
    n = walk_book(no_book.asks, "BUY", shares=sets, fee_rate=market.fee_rate, fee_exponent=market.fee_exponent)
    cost = (y.cost + y.fees + n.cost + n.fees) / sets
    profit = 1.0 - cost
    opp = Opportunity(
        kind="arbitrage", strategy="complete_set_arb", condition_id=market.condition_id,
        event_id=market.event_id, token_id=yes_book.token_id, side="BUY", outcome="YES+NO",
        p_market=(yes_book.best_ask or 0) + (no_book.best_ask or 0), p_est=1.0, p_std=0.0,
        cost_per_share=cost, limit_price=0.0, edge=profit, confidence=1.0 if profit > 0 else 0.0,
        legs=[{"token_id": yes_book.token_id, "shares": sets, "limit": y.worst_price},
              {"token_id": no_book.token_id, "shares": sets, "limit": n.worst_price}],
        detail={"sets": round(sets, 2)})
    if profit < min_profit:
        opp.reject_reason = f"arb profit {profit:.4f} < {min_profit}"
    return opp


def multi_outcome_arbitrage_gap(asks: list[float], fee_rate: float) -> float:
    """For a negative-risk event with mutually exclusive, exhaustive outcomes:
    1 - sum(best YES asks incl. fees). Positive = theoretical arb. Detection only:
    exhaustiveness of Gamma event listings is NOT guaranteed (placeholder/'other'
    outcomes), so the agent never auto-executes this."""
    total = sum(a + taker_fee(1, a, fee_rate) for a in asks)
    return 1.0 - total
