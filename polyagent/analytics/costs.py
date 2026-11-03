"""Fees, slippage and executable price calculations against a real order book.

Taker fee (Polymarket docs, trading/fees): fee_usdc = shares * rate * p * (1 - p).
Markets publish feeSchedule {rate, exponent}; we generalise to
shares * rate * (p * (1 - p)) ** exponent, which equals the documented formula for
exponent == 1 (the only value observed). Makers pay no fee.
"""
from __future__ import annotations

from dataclasses import dataclass


def taker_fee(shares: float, price: float, rate: float, exponent: float = 1.0) -> float:
    if rate <= 0 or shares <= 0:
        return 0.0
    return round(shares * rate * (price * (1 - price)) ** exponent, 5)


@dataclass
class FillEstimate:
    shares: float
    cost: float  # USDC paid (buy) or received (sell), excluding fees
    fees: float
    avg_price: float
    worst_price: float
    fully_filled: bool

    @property
    def all_in_per_share(self) -> float:
        """Buy: USDC paid per share incl. fees. Sell: USDC received per share net of fees."""
        if self.shares <= 0:
            return float("nan")
        return self.avg_price + self.fees / self.shares if self.cost >= 0 else float("nan")


def walk_book(levels: list[tuple[float, float]], side: str, *, usd: float | None = None,
              shares: float | None = None, limit_price: float | None = None,
              fee_rate: float = 0.0, fee_exponent: float = 1.0) -> FillEstimate:
    """Simulate a taker order sweeping `levels` (asks for BUY, bids for SELL).

    Specify either a USDC budget (`usd`, buys only) or a share quantity.
    Fees are computed per level, as each level is a separate match.
    """
    assert side in ("BUY", "SELL")
    assert (usd is None) != (shares is None), "give exactly one of usd/shares"
    got_shares = cost = fees = 0.0
    worst = None
    remaining_usd = usd
    remaining_sh = shares
    for price, size in levels:
        if limit_price is not None:
            if side == "BUY" and price > limit_price + 1e-12:
                break
            if side == "SELL" and price < limit_price - 1e-12:
                break
        if remaining_usd is not None:
            per_share = price + (fee_rate * (price * (1 - price)) ** fee_exponent if fee_rate else 0.0)
            take = min(size, remaining_usd / per_share)
        else:
            take = min(size, remaining_sh)
        if take <= 1e-9:
            break
        f = taker_fee(take, price, fee_rate, fee_exponent)
        got_shares += take
        cost += take * price
        fees += f
        worst = price
        if remaining_usd is not None:
            remaining_usd -= take * price + f
            if remaining_usd <= 1e-6:
                break
        else:
            remaining_sh -= take
            if remaining_sh <= 1e-9:
                break
    if usd is not None:
        full = remaining_usd is not None and remaining_usd <= max(1e-6, 0.001 * usd)
    else:
        full = remaining_sh is not None and remaining_sh <= 1e-9
    avg = cost / got_shares if got_shares > 0 else float("nan")
    return FillEstimate(got_shares, cost, fees, avg, worst if worst is not None else float("nan"), full)


def buy_cost_per_share(levels, usd: float, fee_rate: float, fee_exponent: float = 1.0) -> float:
    """All-in USDC per share for spending `usd` on the asks."""
    est = walk_book(levels, "BUY", usd=usd, fee_rate=fee_rate, fee_exponent=fee_exponent)
    if est.shares <= 0:
        return float("inf")
    return (est.cost + est.fees) / est.shares
