"""Rug filter + momentum entry and multi-stage exits. Pure functions so the live paper
trader and the backtester share exactly the same rules."""
from __future__ import annotations

from dataclasses import dataclass

# RugCheck risk names that are disqualifying regardless of score
HARD_RISKS = ("Mint Authority", "Freeze Authority", "Single holder ownership", "Copycat token",
              "High ownership", "Top 10 holders high ownership", "Large Amount of LP Unlocked",
              "Creator history of rugged tokens", "High holder concentration")


@dataclass
class MemeParams:
    min_age_min: float = 20.0  # let the first-minutes snipe/dump play out
    max_age_min: float = 240.0
    min_liquidity: float = 20_000.0
    min_fdv: float = 50_000.0
    max_fdv: float = 5_000_000.0
    max_rug_score: float = 60.0  # RugCheck normalised score (higher = riskier)
    allow_hard_risks: bool = False
    min_vol_h1: float = 20_000.0
    min_buy_ratio_m5: float = 0.55  # buys / (buys + sells) over last 5 min
    min_chg_h1: float = 10.0  # % price change over 1h
    max_chg_h1: float = 300.0  # don't chase vertical candles
    min_chg_m5: float = 0.0
    # exits (multiples of entry value, measured on executable sell quotes)
    take_profit_mult: float = 2.0
    take_profit_frac: float = 0.5  # sell this fraction at the first target
    trail_drawdown: float = 0.35  # after TP: exit rest if value falls this far from peak
    stop_loss_mult: float = 0.6
    max_hold_hours: float = 12.0
    # sizing
    stake_pct: float = 0.01
    max_positions: int = 8
    max_entry_impact: float = 0.03


@dataclass
class TokenState:
    age_min: float
    liquidity: float
    fdv: float
    vol_h1: float
    buys_m5: int
    sells_m5: int
    chg_m5: float
    chg_h1: float
    rug_score: float | None
    risks: list[str]


def rug_ok(st: TokenState, p: MemeParams) -> tuple[bool, str]:
    if st.rug_score is None:
        return False, "no rug report"
    if st.rug_score > p.max_rug_score:
        return False, f"rug score {st.rug_score:.0f}"
    if not p.allow_hard_risks:
        hits = [r for r in st.risks if any(h.lower() in r.lower() for h in HARD_RISKS)]
        if hits:
            return False, "risk: " + hits[0]
    return True, ""


def entry_signal(st: TokenState, p: MemeParams) -> tuple[bool, str]:
    if not (p.min_age_min <= st.age_min <= p.max_age_min):
        return False, "age"
    ok, why = rug_ok(st, p)
    if not ok:
        return False, why
    if st.liquidity < p.min_liquidity:
        return False, "liquidity"
    if not (p.min_fdv <= st.fdv <= p.max_fdv):
        return False, "fdv"
    if st.vol_h1 < p.min_vol_h1:
        return False, "volume"
    n = st.buys_m5 + st.sells_m5
    if n == 0 or st.buys_m5 / n < p.min_buy_ratio_m5:
        return False, "buy pressure"
    if not (p.min_chg_h1 <= st.chg_h1 <= p.max_chg_h1) or st.chg_m5 < p.min_chg_m5:
        return False, "momentum"
    return True, "entry"


def exit_signal(mult: float, peak_mult: float, hours_held: float, took_profit: bool,
                p: MemeParams) -> tuple[str, float]:
    """Returns (reason, fraction_to_sell). mult = current executable value / entry cost."""
    if not took_profit and mult >= p.take_profit_mult:
        return "take_profit", p.take_profit_frac
    if mult <= p.stop_loss_mult:
        return "stop_loss", 1.0
    if took_profit and peak_mult > 0 and mult <= peak_mult * (1 - p.trail_drawdown):
        return "trailing_stop", 1.0
    if hours_held >= p.max_hold_hours:
        return "time_exit", 1.0
    return "", 0.0
