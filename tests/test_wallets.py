import pytest

from polyagent.wallets.intelligence import Trade, compute_metrics, resolutions_from_positions

DAY = 86400


def t(token, side, size, price, ts, cid=None, wallet="0xw"):
    return Trade(wallet, token, cid or f"c-{token}", side, size, price, ts)


def test_fifo_realized_and_resolution_pnl():
    trades = [t("A", "BUY", 10, 0.4, 0), t("A", "BUY", 10, 0.6, DAY), t("A", "SELL", 15, 0.7, 2 * DAY),
              t("B", "BUY", 100, 0.2, 0)]
    m = compute_metrics("0xw", trades, resolutions={"B": 1.0}, as_of=10 * DAY, min_trades=1,
                        min_resolved=1, min_days=0)
    # FIFO: 10@0.4 + 5@0.6 sold at 0.7 = 3.0 + 0.5 = 3.5; B resolves YES: 100*(1-0.2)=80
    assert m.realized_pnl == pytest.approx(83.5)
    assert m.n_resolved == 1  # A still has 5 open shares, so only B is closed out


def test_point_in_time_ignores_future_trades_and_unknown_resolutions():
    trades = [t("A", "BUY", 10, 0.5, 0), t("A", "SELL", 10, 0.9, 5 * DAY)]
    m = compute_metrics("0xw", trades, resolutions={}, as_of=2 * DAY, min_trades=1, min_resolved=0, min_days=0)
    assert m.n_trades == 1 and m.realized_pnl == 0.0


def test_reliability_gates_and_concentration_flag():
    # one huge win + many tiny losses -> concentrated, not reliable
    trades = [t("BIG", "BUY", 10000, 0.05, 0)]
    res = {"BIG": 1.0}
    for i in range(20):
        trades.append(t(f"L{i}", "BUY", 10, 0.5, i * 3 * DAY))
        res[f"L{i}"] = 0.0
    m = compute_metrics("0xw", trades, res, as_of=100 * DAY, min_trades=10, min_resolved=10, min_days=30)
    assert "concentrated_returns" in m.flags
    assert "lottery_driven" in m.flags
    assert not m.reliable


def test_consistent_winner_is_reliable():
    trades, res = [], {}
    for i in range(60):
        tok = f"T{i}"
        trades.append(t(tok, "BUY", 100, 0.5, i * DAY))
        res[tok] = 1.0 if i % 10 < 7 else 0.0  # 70% win rate, even payoff
    m = compute_metrics("0xw", trades, res, as_of=200 * DAY, min_trades=50, min_resolved=15, min_days=30)
    assert m.win_rate == pytest.approx(0.7)
    assert m.return_lcb > 0 and m.reliable, m.flags


def test_few_observations_not_reliable_even_if_perfect():
    trades = [t(f"T{i}", "BUY", 100, 0.5, i * DAY) for i in range(5)]
    res = {f"T{i}": 1.0 for i in range(5)}
    m = compute_metrics("0xw", trades, res, as_of=100 * DAY)
    assert not m.reliable and "few_trades" in m.flags


def test_resolutions_from_positions():
    rows = [{"token_id": "1", "current_price": 1.0, "redeemable": True, "opposite_token_id": "2"},
            {"token_id": "3", "current_price": 0.42, "redeemable": False}]
    res, marks = resolutions_from_positions(rows)
    assert res == {"1": 1.0, "2": 0.0} and marks == {"3": 0.42}


def test_closed_positions_override_truncated_trade_history():
    from polyagent.wallets.intelligence import WalletMetrics, apply_closed_positions

    m = WalletMetrics("0xw", n_trades=500, days_active=5, trades_per_day=10)
    rows = []
    for i in range(40):
        win = i % 5 != 0  # 80% winners, +20% return; losers -20%
        rows.append({"condition_id": f"c{i}", "realized_pnl": 20.0 if win else -20.0, "total_size": 200.0,
                     "avg_price": 0.5, "end_date": f"2026-0{1 + i % 9}-1{i % 9}"})
    apply_closed_positions(m, rows)
    assert m.n_resolved == 40 and m.win_rate == pytest.approx(0.8)
    assert m.realized_pnl == pytest.approx(32 * 20 - 8 * 20)
    assert m.days_active > 30  # span taken from position end dates
    assert "pnl_from_positions" in m.flags
    assert m.reliable, m.flags


def test_thin_edge_with_big_losses_is_not_reliable():
    from polyagent.wallets.intelligence import WalletMetrics, apply_closed_positions

    m = WalletMetrics("0xw", n_trades=500, days_active=60, trades_per_day=10)
    rows = [{"condition_id": f"c{i}", "realized_pnl": 20.0 if i % 5 else -50.0, "total_size": 200.0,
             "avg_price": 0.5} for i in range(40)]
    apply_closed_positions(m, rows)
    assert m.win_rate == pytest.approx(0.8) and not m.reliable  # 80% wins but LCB of return < 0
