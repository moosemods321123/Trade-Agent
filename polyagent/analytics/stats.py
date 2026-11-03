"""Statistical primitives: regression, Bayesian updating, intervals, drawdown."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

EPS = 1e-9


def logit(p: float) -> float:
    p = min(max(p, 1e-4), 1 - 1e-4)
    return math.log(p / (1 - p))


def sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))


@dataclass
class Regression:
    slope: float
    intercept: float
    r_squared: float
    slope_se: float
    n: int

    @property
    def t_stat(self) -> float:
        return self.slope / self.slope_se if self.slope_se > 0 else 0.0


def linear_regression(x, y) -> Regression:
    """OLS of y on x with slope standard error and R^2."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    if n < 3 or np.ptp(x) == 0:
        return Regression(0.0, float(y.mean()) if n else 0.0, 0.0, float("inf"), n)
    xm, ym = x.mean(), y.mean()
    sxx = ((x - xm) ** 2).sum()
    slope = ((x - xm) * (y - ym)).sum() / sxx
    intercept = ym - slope * xm
    resid = y - (intercept + slope * x)
    ss_res = (resid ** 2).sum()
    ss_tot = ((y - ym) ** 2).sum()
    r2 = 1 - ss_res / ss_tot if ss_tot > EPS else 0.0
    se = math.sqrt(ss_res / (n - 2) / sxx) if n > 2 else float("inf")
    return Regression(float(slope), float(intercept), float(max(r2, 0.0)), float(se), n)


# --- Beta-Binomial Bayesian updating --------------------------------------

@dataclass
class Beta:
    a: float
    b: float

    @classmethod
    def from_mean(cls, mean: float, concentration: float) -> "Beta":
        mean = min(max(mean, 1e-3), 1 - 1e-3)
        return cls(mean * concentration, (1 - mean) * concentration)

    @property
    def mean(self) -> float:
        return self.a / (self.a + self.b)

    @property
    def var(self) -> float:
        s = self.a + self.b
        return self.a * self.b / (s * s * (s + 1))

    @property
    def std(self) -> float:
        return math.sqrt(self.var)

    def update(self, successes: float, failures: float) -> "Beta":
        return Beta(self.a + max(successes, 0), self.b + max(failures, 0))

    def prob_greater(self, threshold: float) -> float:
        """P(theta > threshold). Normal approximation; adequate for a+b >= ~10."""
        if self.std < EPS:
            return 1.0 if self.mean > threshold else 0.0
        return 1 - normal_cdf((threshold - self.mean) / self.std)


def normal_cdf(z: float) -> float:
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


def wilson_lower_bound(successes: float, n: float, z: float = 1.96) -> float:
    if n <= 0:
        return 0.0
    p = successes / n
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, (centre - margin) / denom)


def max_drawdown(equity_curve) -> float:
    """Max peak-to-trough drop as a fraction of the peak (0..1)."""
    eq = np.asarray(equity_curve, dtype=float)
    if len(eq) == 0:
        return 0.0
    peaks = np.maximum.accumulate(eq)
    with np.errstate(divide="ignore", invalid="ignore"):
        dd = np.where(peaks > 0, (peaks - eq) / peaks, 0.0)
    return float(np.nanmax(dd)) if len(dd) else 0.0


def max_drawdown_abs(pnl_curve) -> float:
    """Max peak-to-trough drop of a cumulative P&L curve in currency units."""
    c = np.asarray(pnl_curve, dtype=float)
    if len(c) == 0:
        return 0.0
    c = np.concatenate([[0.0], c])
    return float(np.max(np.maximum.accumulate(c) - c))


def sharpe(returns, periods_per_year: float = 1.0) -> float:
    r = np.asarray(returns, dtype=float)
    if len(r) < 2 or r.std(ddof=1) < EPS:
        return 0.0
    return float(r.mean() / r.std(ddof=1) * math.sqrt(periods_per_year))


def herfindahl(weights) -> float:
    w = np.abs(np.asarray(weights, dtype=float))
    s = w.sum()
    if s <= 0:
        return 0.0
    w = w / s
    return float((w ** 2).sum())


def brier_score(probs, outcomes) -> float:
    p = np.asarray(probs, dtype=float)
    o = np.asarray(outcomes, dtype=float)
    return float(((p - o) ** 2).mean()) if len(p) else float("nan")
