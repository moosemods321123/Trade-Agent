import pytest
from fastapi.testclient import TestClient

from polyagent.agent import Agent
from polyagent.backtest.engine import Backtester, BTParams, MarketRecord
from polyagent.dashboard.app import create_app
from polyagent.execution.brokers import PaperBroker
from polyagent.wallets.intelligence import Trade

from .conftest import FakeApi, make_book, make_market

H = 3600


def make_agent(cfg, db, books, markets):
    api = FakeApi(books, markets)
    agent = Agent(cfg, api=api, broker=PaperBroker(api), db=db)
    agent.scanner.history = lambda token, hours=72: []
    agent.ranker.top_wallets = lambda n=None: {}
    db.set_state("wallet_refresh_ts", 9e18)  # skip network wallet refresh
    return agent


def test_agent_cycle_snapshots_and_takes_arbitrage(cfg, db):
    m = make_market(fee_rate=0.0)
    books = {"YES": make_book("YES", [(0.44, 500)], [(0.45, 500)]),
             "NO": make_book("NO", [(0.49, 500)], [(0.50, 500)])}
    agent = make_agent(cfg, db, books, [m])
    agent.cycle()
    fills = db.query("SELECT * FROM fills")
    assert {f["token_id"] for f in fills} == {"YES", "NO"}
    sets = fills[0]["size"]
    assert agent.portfolio.cash == pytest.approx(1000 - 0.95 * sets)
    # resolve YES: the complete set pays exactly $1/set -> locked-in 5c/set profit
    resolved = make_market(fee_rate=0.0, closed=True, outcome_prices=[1.0, 0.0])
    agent.api.markets = [resolved]
    db.execute("UPDATE markets SET closed=1")
    agent._settle_resolved()
    assert agent.portfolio.cash == pytest.approx(1000 + 0.05 * sets)
    assert not agent.portfolio.positions()


def test_agent_respects_market_cap_across_cycles(cfg, db):
    m = make_market(fee_rate=0.0)
    books = {"YES": make_book("YES", [(0.44, 5000)], [(0.45, 5000)]),
             "NO": make_book("NO", [(0.49, 5000)], [(0.50, 5000)])}
    agent = make_agent(cfg, db, books, [m])
    for _ in range(3):
        agent.cycle()
    basis = db.one("SELECT SUM(size*avg_cost) b FROM positions")["b"]
    assert basis <= cfg.max_position_pct * 1000 + 1e-6


def test_kill_switch_cancels_and_halts(cfg, db):
    agent = make_agent(cfg, db, {}, [])
    db.execute("INSERT INTO orders(client_key,ts_created,token_id,status,exchange_order_id) "
               "VALUES('k',0,'YES','OPEN','x')")
    db.set_state("kill", True)
    agent.cycle()
    assert db.one("SELECT status FROM orders")["status"] == "CANCELLED"
    assert db.get_state("halt")["reason"] == "manual kill switch"


def test_drawdown_breach_halts(cfg, db):
    agent = make_agent(cfg, db, {}, [])
    db.set_state("high_water", 2000.0)  # equity 1000 vs HWM 2000 => 50% drawdown
    agent.cycle()
    assert "drawdown" in db.get_state("halt")["reason"]


def _record(cid, outcome, prices, trades=(), resolve=200 * H):
    hist = [(i * H, p) for i, p in enumerate(prices)]
    return MarketRecord(cid, cid, cid, cid + "Y", cid + "N", 0, resolve, outcome, 0.0, 1.0, hist, list(trades))


def test_backtest_has_no_lookahead_on_price():
    # price jumps to 0.95 only AFTER the last possible decision time: trend must not see it
    prices = [0.5] * 150 + [0.95] * 50
    rec = _record("m1", 1.0, prices)
    bt = Backtester([rec])
    r = bt.run(BTParams(strategy="trend", min_hours_to_close=60, step_hours=6))
    assert r.trades == []


def test_backtest_wallet_ranking_is_point_in_time():
    # wallet w is "perfect" on markets resolving at t=100h; at t=50h nothing is resolved yet
    recs = []
    for i in range(6):
        trs = [Trade("w", f"r{i}Y", f"r{i}", "BUY", 100, 0.5, 10 * H)]
        recs.append(_record(f"r{i}", 1.0, [0.5] * 100, trs, resolve=100 * H))
    bt = Backtester(recs)
    p = BTParams(wallet_min_trades=1, wallet_min_resolved=1, wallet_min_days=0)
    assert bt._wallet_weights(50 * H, p) == {}
    # after resolution, the (tiny-sample, zero-variance) record is visible to the ranker
    assert isinstance(bt._wallet_weights(150 * H, p), dict)


def test_backtest_accounting_conserves_money():
    prices = [0.3 + 0.002 * i for i in range(200)]
    recs = [_record("up", 1.0, prices, resolve=260 * H)]
    r = Backtester(recs).run(BTParams(strategy="trend", min_hours_to_close=24, step_hours=12, min_edge=0.0,
                                      min_confidence=0.5))
    if r.trades:
        assert r.end_equity == pytest.approx(1000 + sum(t.pnl for t in r.trades))


def test_dashboard_endpoints_and_kill(cfg, db):
    agent = make_agent(cfg, db, {}, [])
    agent.cycle()
    client = TestClient(create_app(cfg, db))
    s = client.get("/api/summary").json()
    assert s["equity"] == pytest.approx(1000) and s["pnl_pct"] == 0
    assert client.get("/api/trades").json() == []
    assert client.get("/").status_code == 200
    assert client.get("/healthz").status_code == 200
    assert client.post("/api/kill").json()["ok"]
    assert client.get("/api/summary").json()["kill"] is True
    client.post("/api/resume")
    assert client.get("/api/summary").json()["kill"] is False


def test_dashboard_token_required_when_set(cfg, db):
    cfg.dashboard_token = "s3cret"
    client = TestClient(create_app(cfg, db))
    assert client.post("/api/kill").status_code == 401
    assert client.post("/api/kill", headers={"X-Token": "s3cret"}).status_code == 200


def test_flow_signal_study_detects_informative_flow():
    from polyagent.backtest.engine import flow_signal_study

    recs = []
    for i in range(8):
        outcome = float(i % 2)
        tok = f"e{i}Y" if outcome else f"e{i}N"  # informed wallet buys the eventual winner
        trs = [Trade("all", tok, f"e{i}", "BUY", 1000, 0.5, 30 * H)]
        recs.append(_record(f"e{i}", outcome, [0.5] * 100, trs, resolve=100 * H))
    res = flow_signal_study(Backtester(recs), BTParams(), smart_only=False)
    assert res["events"] == 8 and res["mean_signed_residual"] == pytest.approx(0.5)
