"""Independent probability estimators.

Each estimator returns an Estimate (mean, std) for the YES outcome of a binary
market, using only data passed in (point-in-time safe), or None for "no view".
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from ..analytics.stats import Beta, linear_regression, logit, sigmoid


@dataclass
class Estimate:
    p: float  # estimated P(YES)
    std: float  # uncertainty of the estimate itself
    strategy: str
    detail: dict = field(default_factory=dict)


@dataclass
class EstimatorParams:
    # market prior strength: Beta concentration around market mid (higher = trust market more)
    prior_concentration: float = 200.0
    # max pseudo-observations a single smart wallet can contribute
    per_wallet_pseudo: float = 6.0
    # USDC notional for a wallet's flow to count fully
    flow_full_usd: float = 500.0
    min_smart_wallets: int = 2
    # trend estimator
    trend_lookback_hours: int = 72
    trend_horizon_hours: float = 24.0
    trend_damping: float = 0.3
    min_r_squared: float = 0.5
    min_t_stat: float = 3.0
    min_points: int = 24


class SmartMoneyEstimator:
    """Bayesian update of the market-implied probability using net directional flow
    of wallets that passed the reliability gates.

    Prior: Beta centred at market mid with concentration K (market is informative).
    Evidence: each reliable wallet contributes pseudo-counts toward YES or NO in
    proportion to (wallet weight) x (net flow / flow_full_usd), capped per wallet.
    The posterior mean is the estimate; its std is the estimate's uncertainty.
    """

    name = "smart_money"

    def __init__(self, params: EstimatorParams | None = None):
        self.p = params or EstimatorParams()

    def estimate(self, mid: float, yes_token: str, no_token: str, trades: list,
                 wallet_weights: dict[str, float]) -> Estimate | None:
        """trades: iterable of objects with wallet, token_id, side, size, price."""
        if not wallet_weights or mid is None:
            return None
        flow: dict[str, float] = {}
        for t in trades:
            w = t.wallet
            if w not in wallet_weights:
                continue
            notional = t.size * t.price
            sign = 1.0 if t.side == "BUY" else -1.0
            if t.token_id == yes_token:
                flow[w] = flow.get(w, 0.0) + sign * notional
            elif t.token_id == no_token:
                flow[w] = flow.get(w, 0.0) - sign * notional
        active = {w: f for w, f in flow.items() if abs(f) > 1.0}
        if len(active) < self.p.min_smart_wallets:
            return None
        yes = no = 0.0
        for w, f in active.items():
            pseudo = wallet_weights[w] * min(abs(f) / self.p.flow_full_usd, 1.0) * self.p.per_wallet_pseudo
            if f > 0:
                yes += pseudo
            else:
                no += pseudo
        prior = Beta.from_mean(mid, self.p.prior_concentration)
        post = prior.update(yes, no)
        return Estimate(post.mean, post.std, self.name,
                        {"wallets": len(active), "yes_pseudo": round(yes, 2), "no_pseudo": round(no, 2),
                         "prior_mean": round(mid, 4)})


class TrendEstimator:
    """Regression of logit(price) on time. Only emits a view when the trend is
    statistically strong (R^2 and slope t-stat gates); the projection is damped.
    Prediction-market momentum is weak/unstable — expect this to fire rarely."""

    name = "trend"

    def __init__(self, params: EstimatorParams | None = None):
        self.p = params or EstimatorParams()

    def estimate(self, history: list[tuple[int, float]], now_ts: int) -> Estimate | None:
        start = now_ts - self.p.trend_lookback_hours * 3600
        pts = [(t, p) for t, p in history if start <= t <= now_ts and 0 < p < 1]
        if len(pts) < self.p.min_points:
            return None
        xs = [(t - now_ts) / 3600 for t, _ in pts]
        ys = [logit(p) for _, p in pts]
        reg = linear_regression(xs, ys)
        if reg.r_squared < self.p.min_r_squared or abs(reg.t_stat) < self.p.min_t_stat:
            return None
        p_now = pts[-1][1]
        drift = reg.slope * self.p.trend_horizon_hours * self.p.trend_damping
        p_est = sigmoid(logit(p_now) + drift)
        # uncertainty: slope standard error propagated + model-risk floor
        dlogit_sd = reg.slope_se * self.p.trend_horizon_hours * self.p.trend_damping
        p_sd = max(p_est * (1 - p_est) * dlogit_sd, 0.02 + abs(p_est - p_now) * 0.5)
        return Estimate(p_est, p_sd, self.name,
                        {"slope_per_h": round(reg.slope, 5), "r2": round(reg.r_squared, 3),
                         "t": round(reg.t_stat, 2), "n": reg.n})


def apply_news_uncertainty(est: Estimate, fresh_news: bool, boost: float) -> Estimate:
    """Fresh unprocessed news means the price may be stale OR our view may be — we
    widen uncertainty rather than guess direction from headlines."""
    if fresh_news and est is not None:
        est.std *= boost
        est.detail["news_boost"] = boost
    return est


def normal_prob_above(mean: float, std: float, threshold: float) -> float:
    if std <= 1e-9:
        return 1.0 if mean > threshold else 0.0
    return 0.5 * (1 + math.erf((mean - threshold) / (std * math.sqrt(2))))
