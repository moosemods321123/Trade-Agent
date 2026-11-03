import pytest

from polyagent.execution.brokers import LiveBroker, OrderRequest, PaperBroker
from polyagent.execution.orders import OrderManager, client_key
from polyagent.execution.portfolio import Portfolio

from .conftest import FakeApi, make_book


def setup(cfg, db, asks=((0.5, 100),), bids=((0.48, 100),)):
    api = FakeApi({"YES": make_book("YES", list(bids), list(asks))})
    pf = Portfolio(db, 1000.0)
    om = OrderManager(db, PaperBroker(api), pf, cfg)
    return api, pf, om


def test_paper_fill_updates_cash_position_and_fees(cfg, db):
    _, pf, om = setup(cfg, db)
    res = om.submit(OrderRequest("YES", "BUY", 0.5, 20, "FAK", 0.04), condition_id="c", event_id="e", strategy="s")
    assert res.status == "FILLED" and res.filled_size == pytest.approx(20)
    fee = 20 * 0.04 * 0.25
    assert pf.cash == pytest.approx(1000 - 10 - fee)
    pos = pf.positions()[0]
    assert pos["size"] == pytest.approx(20) and pos["avg_cost"] == pytest.approx((10 + fee) / 20)
    order = db.one("SELECT * FROM orders")
    assert order["status"] == "FILLED" and order["fees_paid"] == pytest.approx(fee)


def test_limit_respected_and_partial_fill_recorded(cfg, db):
    _, pf, om = setup(cfg, db, asks=((0.5, 10), (0.6, 100)))
    res = om.submit(OrderRequest("YES", "BUY", 0.55, 50, "FAK"), condition_id="c", event_id="e", strategy="s")
    assert res.status == "PARTIAL" and res.filled_size == pytest.approx(10)
    assert db.one("SELECT filled_size FROM orders")["filled_size"] == pytest.approx(10)


def test_fok_not_filled_leaves_state_untouched(cfg, db):
    _, pf, om = setup(cfg, db, asks=((0.5, 10),))
    res = om.submit(OrderRequest("YES", "BUY", 0.5, 50, "FOK"), condition_id="c", event_id="e", strategy="s")
    assert res.status == "CANCELLED" and pf.cash == 1000 and not pf.positions()


def test_duplicate_orders_blocked(cfg, db):
    _, pf, om = setup(cfg, db)
    key = client_key("s", "YES", "BUY", now=1000)
    a = om.submit(OrderRequest("YES", "BUY", 0.5, 5), condition_id="c", event_id="e", strategy="s", key=key)
    b = om.submit(OrderRequest("YES", "BUY", 0.5, 5), condition_id="c", event_id="e", strategy="s", key=key)
    assert a is not None and b is None
    assert db.one("SELECT COUNT(*) n FROM orders")["n"] == 1
    assert db.one("SELECT COUNT(*) n FROM audit_log WHERE event='order_duplicate_blocked'")["n"] == 1


def test_sell_realizes_pnl_and_settlement(cfg, db):
    api, pf, om = setup(cfg, db, asks=((0.5, 100),), bids=((0.7, 100),))
    om.submit(OrderRequest("YES", "BUY", 0.5, 20), condition_id="c", event_id="e", strategy="s", key="k1")
    om.submit(OrderRequest("YES", "SELL", 0.7, 10), condition_id="c", event_id="e", strategy="s", key="k2")
    pos = pf.positions()[0]
    assert pos["realized_pnl"] == pytest.approx(10 * 0.2)
    pnl = pf.settle("YES", 1.0)
    assert pnl == pytest.approx(10 * 0.5)
    assert pf.cash == pytest.approx(1000 - 10 + 7 + 10)
    assert not pf.positions()


def test_cancel_all_marks_open_orders(cfg, db):
    _, pf, om = setup(cfg, db)
    db.execute("INSERT INTO orders(client_key,ts_created,token_id,status) VALUES('x',0,'YES','OPEN')")
    om.cancel_all("risk")
    assert db.one("SELECT status FROM orders WHERE client_key='x'")["status"] == "CANCELLED"


class FakeClob:
    def __init__(self):
        self.cancelled = []

    def create_order(self, args):
        return {"signed": args}

    def post_order(self, order, order_type):
        return {"success": True, "orderID": "oid1", "status": "matched"}

    def get_order(self, oid):
        return {"id": oid, "status": "MATCHED", "original_size": "10", "size_matched": "10",
                "associate_trades": ["t1"]}

    def get_trades(self, params):
        return [{"id": "t1", "price": "0.42", "size": "10", "status": "CONFIRMED", "match_time": "1700000000"}]

    def cancel(self, oid):
        self.cancelled.append(oid)

    def cancel_all(self):
        self.cancelled.append("*")

    def get_balance_allowance(self, params):
        return {"balance": "123450000"}


def test_live_broker_verifies_fills_via_trades(cfg, monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    b = LiveBroker(cfg, client=FakeClob())
    res = b.place(OrderRequest("YES", "BUY", 0.45, 10, "FAK", 0.04))
    assert res.status == "FILLED" and res.fills[0].price == 0.42 and res.fills[0].fee > 0
    assert b.collateral_balance() == pytest.approx(123.45)
    assert b.cancel_all()


def test_live_broker_reports_rejection(cfg):
    class Rejecting(FakeClob):
        def post_order(self, order, order_type):
            return {"success": False, "errorMsg": "not enough balance"}

    res = LiveBroker(cfg, client=Rejecting()).place(OrderRequest("YES", "BUY", 0.45, 10))
    assert res.status == "REJECTED" and "balance" in res.error
