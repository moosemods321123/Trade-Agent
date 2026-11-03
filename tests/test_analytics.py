import math

import pytest

from polyagent.analytics.costs import taker_fee, walk_book
from polyagent.analytics.stats import (Beta, linear_regression, max_drawdown, max_drawdown_abs,
                                       wilson_lower_bound)


def test_taker_fee_matches_documented_examples():
    # docs: 100 shares at 50c -> politics (0.04) $1.00, crypto (0.07) $1.75
    assert taker_fee(100, 0.5, 0.04) == pytest.approx(1.00)
    assert taker_fee(100, 0.5, 0.07) == pytest.approx(1.75)
    assert taker_fee(100, 0.5, 0.0) == 0.0
    # symmetric in p and 1-p, smaller near extremes
    assert taker_fee(100, 0.1, 0.04) == pytest.approx(taker_fee(100, 0.9, 0.04))
    assert taker_fee(100, 0.1, 0.04) < taker_fee(100, 0.5, 0.04)


def test_walk_book_by_shares_and_limit():
    asks = [(0.50, 10), (0.52, 10), (0.60, 100)]
    f = walk_book(asks, "BUY", shares=15)
    assert f.shares == pytest.approx(15)
    assert f.cost == pytest.approx(10 * 0.5 + 5 * 0.52)
    assert f.worst_price == 0.52 and f.fully_filled
    lim = walk_book(asks, "BUY", shares=50, limit_price=0.52)
    assert lim.shares == pytest.approx(20) and not lim.fully_filled


def test_walk_book_by_usd_includes_fees():
    asks = [(0.5, 1000)]
    f = walk_book(asks, "BUY", usd=51.0, fee_rate=0.04)
    # per share all-in = 0.5 + 0.04*0.25 = 0.51 -> 100 shares
    assert f.shares == pytest.approx(100, rel=1e-6)
    assert f.cost + f.fees == pytest.approx(51.0, abs=1e-3)


def test_regression_recovers_slope_and_r2():
    xs = list(range(50))
    ys = [2 + 0.3 * x for x in xs]
    r = linear_regression(xs, ys)
    assert r.slope == pytest.approx(0.3) and r.r_squared == pytest.approx(1.0)
    noisy = linear_regression(xs, [math.sin(x) for x in xs])
    assert noisy.r_squared < 0.1


def test_beta_update_moves_mean_and_shrinks_std():
    prior = Beta.from_mean(0.5, 20)
    post = prior.update(10, 0)
    assert post.mean > prior.mean and post.std < prior.std
    assert 0 < post.prob_greater(0.5) <= 1


def test_wilson_lcb_penalises_small_samples():
    assert wilson_lower_bound(3, 3) < wilson_lower_bound(90, 100)
    assert wilson_lower_bound(0, 0) == 0.0


def test_drawdowns():
    assert max_drawdown([100, 120, 90, 130]) == pytest.approx(0.25)
    assert max_drawdown_abs([10, 30, 5, 40]) == pytest.approx(25)
