"""Polymarket liquidity-reward scoring (docs: programs/liquidity-rewards).

    S(v, s) = ((v - s) / v)^2 * size          for orders within v of the (size-adjusted) midpoint
    Qone    = score of liquidity bidding for YES  (YES bids, incl. mirrored NO asks)
    Qtwo    = score of liquidity offering YES     (YES asks, incl. mirrored NO bids)
    mid in [0.10, 0.90]: Qmin = max(min(Qone, Qtwo), max(Qone / c, Qtwo / c)),  c = 3
    otherwise:           Qmin = min(Qone, Qtwo)          (must be two-sided)

Each minute, a maker's Qmin is normalised against all makers; the day's reward is the
market's daily pool times the average normalised share.

Approximation (documented): we only see the aggregated book, not per-maker orders, so
competitors are treated as one aggregate maker whose Qmin is computed from the book.
The minimum-size filter is applied per price level, which can only overstate competition.
"""
from __future__ import annotations

from dataclasses import dataclass

C_SCALE = 3.0


def order_score(max_spread: float, spread: float, size: float) -> float:
    if max_spread <= 0 or spread >= max_spread or size <= 0:
        return 0.0
    return ((max_spread - spread) / max_spread) ** 2 * size


def side_score(levels: list[tuple[float, float]], mid: float, max_spread: float, min_size: float) -> float:
    return sum(order_score(max_spread, abs(p - mid), s) for p, s in levels if s >= min_size)


def q_min(q_one: float, q_two: float, mid: float, c: float = C_SCALE) -> float:
    if 0.10 <= mid <= 0.90:
        return max(min(q_one, q_two), max(q_one / c, q_two / c))
    return min(q_one, q_two)


@dataclass
class RewardEstimate:
    ours: float
    others: float
    share: float


def estimate_share(*, mid: float, max_spread: float, min_size: float,
                   book_bids: list[tuple[float, float]], book_asks: list[tuple[float, float]],
                   our_yes_bid: tuple[float, float] | None, our_no_bid: tuple[float, float] | None) -> RewardEstimate:
    """Our normalised share of the market's reward pool at this instant.

    our_yes_bid = (price, size) of our YES bid; our_no_bid = (price, size) of our NO bid,
    which sits on the YES book as an ask at 1 - price.
    """
    q1_others = side_score(book_bids, mid, max_spread, min_size)
    q2_others = side_score(book_asks, mid, max_spread, min_size)
    q1 = q2 = 0.0
    if our_yes_bid and our_yes_bid[1] >= min_size:
        q1 = order_score(max_spread, abs(mid - our_yes_bid[0]), our_yes_bid[1])
    if our_no_bid and our_no_bid[1] >= min_size:
        q2 = order_score(max_spread, abs((1 - our_no_bid[0]) - mid), our_no_bid[1])
    ours = q_min(q1, q2, mid)
    others = q_min(q1_others, q2_others, mid)
    total = ours + others
    return RewardEstimate(ours, others, ours / total if total > 0 else 0.0)
