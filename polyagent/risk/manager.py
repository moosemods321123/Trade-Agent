"""Position sizing (uncertainty-adjusted fractional Kelly) and portfolio risk limits."""
from __future__ import annotations

import datetime as dt
import time
from dataclasses import dataclass

from ..config import Config


def kelly_fraction_binary(p: float, cost: float) -> float:
    """Full-Kelly fraction of bankroll for buying a $1-payout contract at all-in `cost`.
    Net odds b = (1 - cost)/cost; f* = (b p - (1-p))/b = (p - cost)/(1 - cost)."""
    if cost <= 0 or cost >= 1 or p <= cost:
        return 0.0
    return (p - cost) / (1 - cost)


def uncertainty_adjusted_kelly(p: float, std: float, cost: float, fraction: float) -> float:
    """Fractional Kelly shrunk by estimate uncertainty: f = frac * f* * edge^2/(edge^2+std^2).
    When std is comparable to the edge the stake shrinks sharply."""
    edge = p - cost
    if edge <= 0:
        return 0.0
    shrink = edge * edge / (edge * edge + std * std) if std > 0 else 1.0
    return fraction * kelly_fraction_binary(p, cost) * shrink


@dataclass
class PortfolioState:
    cash: float
    equity: float
    high_water: float
    day_start_equity: float
    exposure_by_market: dict[str, float]  # condition_id -> cost basis
    exposure_by_event: dict[str, float]  # event_id -> cost basis

    @property
    def total_exposure(self) -> float:
        return sum(self.exposure_by_market.values())

    @property
    def drawdown(self) -> float:
        return (self.high_water - self.equity) / self.high_water if self.high_water > 0 else 0.0

    @property
    def daily_loss(self) -> float:
        if self.day_start_equity <= 0:
            return 0.0
        return max(0.0, (self.day_start_equity - self.equity) / self.day_start_equity)


class RiskManager:
    def __init__(self, cfg: Config, db):
        self.cfg = cfg
        self.db = db

    # --- halts --------------------------------------------------------------
    def check_limits(self, st: PortfolioState) -> str | None:
        """Return a reason string if trading must halt (and orders be cancelled)."""
        if st.drawdown >= self.cfg.max_drawdown_pct:
            return f"max drawdown breached: {st.drawdown:.2%} >= {self.cfg.max_drawdown_pct:.2%}"
        if st.daily_loss >= self.cfg.daily_loss_limit_pct:
            return f"daily loss limit breached: {st.daily_loss:.2%}"
        if st.equity <= 0:
            return "non-positive equity"
        return None

    def is_halted(self) -> tuple[bool, str]:
        halt = self.db.get_state("halt")
        if halt:
            # daily-loss halts auto-expire at next UTC day; drawdown/manual halts do not
            if halt.get("until") and time.time() >= halt["until"]:
                self.db.set_state("halt", None)
                return False, ""
            return True, halt.get("reason", "halted")
        return False, ""

    def halt(self, reason: str, until_next_day: bool = False) -> None:
        until = None
        if until_next_day:
            tomorrow = dt.datetime.now(dt.timezone.utc).date() + dt.timedelta(days=1)
            until = dt.datetime.combine(tomorrow, dt.time(), tzinfo=dt.timezone.utc).timestamp()
        self.db.set_state("halt", {"reason": reason, "ts": time.time(), "until": until})
        self.db.audit("halt", {"reason": reason, "until": until}, level="CRITICAL")

    # --- sizing -------------------------------------------------------------
    def size_order(self, st: PortfolioState, condition_id: str, event_id: str, p: float,
                   std: float, cost: float) -> tuple[float, str]:
        """Return (usd_to_spend, reason_if_zero)."""
        c = self.cfg
        f = uncertainty_adjusted_kelly(p, std, cost, c.kelly_fraction)
        if f <= 0:
            return 0.0, "no positive Kelly stake"
        stake = f * st.equity
        caps = {
            "kelly": stake,
            "market_cap": c.max_position_pct * st.equity - st.exposure_by_market.get(condition_id, 0.0),
            "event_cap": c.max_event_pct * st.equity - st.exposure_by_event.get(event_id, 0.0),
            "total_cap": c.max_total_exposure_pct * st.equity - st.total_exposure,
            "cash_reserve": st.cash - c.min_cash_reserve_pct * st.equity,
            "max_order": c.max_order_usd,
        }
        binding = min(caps, key=caps.get)
        usd = max(0.0, caps[binding])
        if usd < c.min_order_usd:
            return 0.0, f"stake {usd:.2f} below minimum (binding: {binding})"
        return round(usd, 2), binding
