import pytest

from polyagent.risk.manager import (PortfolioState, RiskManager, kelly_fraction_binary,
                                    uncertainty_adjusted_kelly)
from polyagent.strategy.engine import (evaluate_binary_arbitrage, evaluate_directional,
                                       multi_outcome_arbitrage_gap)
from polyagent.strategy.estimators import Estimate, SmartMoneyEstimator, TrendEstimator
from polyagent.wallets.intelligence import Trade

from .conftest import make_book, make_market


def test_kelly_basics():
    assert kelly_fraction_binary(0.6, 0.5) == pytest.approx(0.2)
    assert kelly_fraction_binary(0.4, 0.5) == 0.0
    # uncertainty shrinks the stake
    sure = uncertainty_adjusted_kelly(0.6, 0.0, 0.5, 0.25)
    unsure = uncertainty_adjusted_kelly(0.6, 0.1, 0.5, 0.25)
    assert sure == pytest.approx(0.05) and unsure == pytest.approx(0.025)


def test_sizing_respects_every_cap(cfg, db):
    rm = RiskManager(cfg, db)
    st = PortfolioState(cash=1000, equity=1000, high_water=1000, day_start_equity=1000,
                        exposure_by_market={"m": 45.0}, exposure_by_event={})
    usd, binding = rm.size_order(st, "m", "e", p=0.9, std=0.0, cost=0.5)
    assert usd == pytest.approx(5.0) and binding == "market_cap"  # 5% cap minus existing 45
    st2 = PortfolioState(cash=210, equity=1000, high_water=1000, day_start_equity=1000,
                         exposure_by_market={}, exposure_by_event={})
    usd2, _ = rm.size_order(st2, "x", "e", p=0.9, std=0.0, cost=0.5)
    assert usd2 == pytest.approx(10.0)  # cash 210 - 20% reserve of 1000
    usd3, why = rm.size_order(st2, "x", "e", p=0.5, std=0.0, cost=0.5)
    assert usd3 == 0.0


def test_drawdown_and_daily_halts(cfg, db):
    rm = RiskManager(cfg, db)
    ok = PortfolioState(1000, 950, 1000, 1000, {}, {})
    assert rm.check_limits(ok) is None
    assert "drawdown" in rm.check_limits(PortfolioState(500, 790, 1000, 790, {}, {}))
    assert "daily" in rm.check_limits(PortfolioState(500, 910, 920, 1000, {}, {}))
    rm.halt("test", until_next_day=True)
    halted, why = rm.is_halted()
    assert halted and why == "test"


def test_directional_accepts_only_real_edge():
    m = make_market()
    yes = make_book("YES", [(0.49, 1000)], [(0.51, 1000)])
    no = make_book("NO", [(0.49, 1000)], [(0.51, 1000)])
    strong = evaluate_directional(Estimate(0.62, 0.02, "t"), m, yes, no, 25, 0.03, 0.8, 0.06)
    assert strong.accepted and strong.token_id == "YES" and strong.edge > 0.09
    weak = evaluate_directional(Estimate(0.53, 0.02, "t"), m, yes, no, 25, 0.03, 0.8, 0.06)
    assert not weak.accepted  # 0.53 - (0.51 + fee) < 0.03
    uncertain = evaluate_directional(Estimate(0.62, 0.15, "t"), m, yes, no, 25, 0.03, 0.8, 0.06)
    assert not uncertain.accepted and "confidence" in uncertain.reject_reason
    bear = evaluate_directional(Estimate(0.35, 0.02, "t"), m, yes, no, 25, 0.03, 0.8, 0.06)
    assert bear.accepted and bear.token_id == "NO"


def test_limit_price_keeps_edge_after_fees():
    m = make_market()
    yes = make_book("YES", [(0.49, 1000)], [(0.51, 1000)])
    no = make_book("NO", [(0.49, 1000)], [(0.51, 1000)])
    o = evaluate_directional(Estimate(0.70, 0.01, "t"), m, yes, no, 25, 0.03, 0.8, 0.06)
    lim = o.limit_price
    assert lim + 0.04 * lim * (1 - lim) <= 0.70 - 0.03 + 1e-6


def test_binary_arbitrage_detects_and_rejects():
    m = make_market(fee_rate=0.0)
    yes = make_book("YES", [], [(0.45, 100)])
    no = make_book("NO", [], [(0.50, 100)])
    arb = evaluate_binary_arbitrage(m, yes, no, sets_usd=50, min_profit=0.005)
    assert arb.accepted and arb.edge == pytest.approx(0.05)
    no2 = make_book("NO", [], [(0.56, 100)])
    assert not evaluate_binary_arbitrage(m, yes, no2, 50, 0.005).accepted
    assert multi_outcome_arbitrage_gap([0.3, 0.3, 0.3], 0.0) == pytest.approx(0.1)


def test_smart_money_estimator_moves_toward_flow():
    est = SmartMoneyEstimator()
    trades = [Trade("w1", "YES", "c", "BUY", 2000, 0.5, 0), Trade("w2", "NO", "c", "SELL", 2000, 0.5, 0)]
    e = est.estimate(0.5, "YES", "NO", trades, {"w1": 1.0, "w2": 1.0})
    assert e.p > 0.5
    assert est.estimate(0.5, "YES", "NO", trades[:1], {"w1": 1.0}) is None  # needs >=2 wallets
    ignored = [Trade("rando", "YES", "c", "BUY", 1e6, 0.5, 0)] * 2
    assert est.estimate(0.5, "YES", "NO", ignored, {"w1": 1.0}) is None


def test_trend_estimator_requires_strong_fit():
    now = 100 * 3600
    trending = [(now - h * 3600, 0.3 + 0.004 * (72 - h)) for h in range(72, -1, -1)]
    e = TrendEstimator().estimate(trending, now)
    assert e is not None and e.p > trending[-1][1]
    flat_noise = [(now - h * 3600, 0.5 + (0.02 if h % 2 else -0.02)) for h in range(72, -1, -1)]
    assert TrendEstimator().estimate(flat_noise, now) is None
