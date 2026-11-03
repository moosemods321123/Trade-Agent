import time

import pytest

from polyagent.mm.maker import MarketMaker, MMMarket, target_quotes
from polyagent.mm.rewards import estimate_share, order_score, q_min
from polyagent.mm.sim import FillSimulator, Quote, TapeTrade

from .conftest import FakeApi, make_book, make_market


# --- reward scoring ----------------------------------------------------------

def test_order_score_formula():
    assert order_score(0.04, 0.0, 100) == pytest.approx(100)
    assert order_score(0.04, 0.02, 100) == pytest.approx(25)  # ((0.04-0.02)/0.04)^2 * 100
    assert order_score(0.04, 0.04, 100) == 0.0  # at/outside the band scores nothing


def test_qmin_two_sided_rule():
    assert q_min(90, 0, 0.5) == pytest.approx(30)  # single-sided scores at 1/c inside [0.1, 0.9]
    assert q_min(90, 0, 0.05) == 0.0  # must be two-sided at the extremes
    assert q_min(40, 60, 0.5) == pytest.approx(40)


def test_share_grows_with_tighter_quotes_and_less_competition():
    bids, asks = [(0.48, 1000)], [(0.52, 1000)]
    loose = estimate_share(mid=0.5, max_spread=0.04, min_size=10, book_bids=bids, book_asks=asks,
                           our_yes_bid=(0.48, 100), our_no_bid=(0.48, 100))
    tight = estimate_share(mid=0.5, max_spread=0.04, min_size=10, book_bids=bids, book_asks=asks,
                           our_yes_bid=(0.49, 100), our_no_bid=(0.49, 100))
    assert 0 < loose.share < tight.share < 1
    alone = estimate_share(mid=0.5, max_spread=0.04, min_size=10, book_bids=[], book_asks=[],
                           our_yes_bid=(0.49, 100), our_no_bid=(0.49, 100))
    assert alone.share == pytest.approx(1.0)
    small = estimate_share(mid=0.5, max_spread=0.04, min_size=200, book_bids=[], book_asks=[],
                           our_yes_bid=(0.49, 100), our_no_bid=(0.49, 100))
    assert small.share == 0.0  # below rewardsMinSize


# --- quote placement -----------------------------------------------------------

def test_target_quotes_inside_band_and_not_crossing():
    yb, nb = target_quotes(0.50, 0.045, 0.01, 0.5, 0.49, 0.51)
    assert 0.5 - yb < 0.045 and 0.5 - nb < 0.045
    assert yb < 0.51 and nb < 1 - 0.49
    assert yb + nb < 1  # a filled pair is always a profitable complete set
    assert target_quotes(0.5, 0.005, 0.01, 0.5, 0.49, 0.51) is None  # band narrower than a tick


# --- fill simulation -------------------------------------------------------------

def _sim(queue=50.0, placed=100.0):
    s = FillSimulator(tick=0.01)
    s.place(Quote("Y", "N", 0.48, 100, placed, queue))
    return s


def test_trade_through_fills_immediately():
    s = _sim()
    f = s.on_trade(TapeTrade("t1", "Y", "SELL", 0.45, 30, 110))
    assert len(f) == 1 and f[0].size == 30 and f[0].price == 0.48


def test_equal_price_respects_queue():
    s = _sim(queue=50)
    assert s.on_trade(TapeTrade("t1", "Y", "SELL", 0.48, 40, 110)) == []
    f = s.on_trade(TapeTrade("t2", "Y", "SELL", 0.48, 30, 111))
    assert f[0].size == pytest.approx(20)


def test_complement_buy_fills_our_bid():
    s = _sim(queue=0)
    f = s.on_trade(TapeTrade("t1", "N", "BUY", 0.55, 10, 110))  # 1 - 0.55 = 0.45 < 0.48 -> through
    assert f and f[0].size == 10


def test_no_fill_before_placement_after_cancel_or_duplicate():
    s = _sim(placed=100)
    assert s.on_trade(TapeTrade("early", "Y", "SELL", 0.40, 10, 99)) == []
    assert s.on_trade(TapeTrade("x", "Y", "SELL", 0.40, 10, 105))
    assert s.on_trade(TapeTrade("x", "Y", "SELL", 0.40, 10, 105)) == []  # dedupe
    s.cancel("Y", 106)
    assert s.on_trade(TapeTrade("late", "Y", "SELL", 0.40, 10, 107)) == []
    assert s.on_trade(TapeTrade("buy", "Y", "BUY", 0.40, 10, 105.5)) == []  # same-token buy never hits a bid


# --- maker integration -------------------------------------------------------------

class MMApi(FakeApi):
    def __init__(self, books, markets, tape=None):
        super().__init__(books, markets)
        self.tape = tape or []

    def market_trades(self, cid, max_items=200, min_usd=None):
        return self.tape


def _maker(cfg, db, book, tape=None):
    m = make_market(token_ids=["Y", "N"])
    api = MMApi({"Y": book}, [m], tape)
    mk = MarketMaker(cfg, api=api, db=db)
    mk.last_select = 9e18  # skip network selection
    mk.markets[m.condition_id] = MMMarket(m, rate=100.0, max_spread=0.045, min_size=10, last_accrual=time.time())
    return mk, m


def test_maker_quotes_accrues_and_merges(cfg, db):
    book = make_book("Y", [(0.49, 500)], [(0.51, 500)])
    mk, m = _maker(cfg, db, book)
    t0 = time.time()
    mk.cycle(t0)
    mm = mk.markets[m.condition_id]
    assert mm.yes_q and mm.no_q and mm.yes_q.price + mm.no_q.price < 1
    # both sides get traded through -> a complete set forms and is merged at a profit
    mk.api.tape = [
        {"transaction_hash": "a", "token_id": "Y", "side": "SELL", "price": 0.40, "size": 20, "timestamp": t0 + 5},
        {"transaction_hash": "b", "token_id": "N", "side": "SELL", "price": 0.40, "size": 20, "timestamp": t0 + 6},
    ]
    mk.cycle(t0 + 15)
    assert not mk.portfolio.positions()  # merged away
    merge = db.one("SELECT * FROM fills WHERE side='MERGE'")
    assert merge["size"] == pytest.approx(20) and merge["realized_pnl"] > 0
    acc = db.one("SELECT SUM(accrued) a FROM mm_rewards")["a"]
    assert acc and acc > 0


def test_inventory_limit_stops_the_heavy_side(cfg, db):
    cfg.mm_max_inventory_usd = 5.0
    book = make_book("Y", [(0.49, 500)], [(0.51, 500)])
    mk, m = _maker(cfg, db, book)
    t0 = time.time()
    mk.cycle(t0)
    mk.api.tape = [{"transaction_hash": "a", "token_id": "Y", "side": "SELL", "price": 0.40, "size": 50,
                    "timestamp": t0 + 5}]
    mk.cycle(t0 + 15)
    mm = mk.markets[m.condition_id]
    assert mm.yes_q is None and mm.no_q is not None  # only the set-completing side stays


def test_volatility_guard_pulls_quotes(cfg, db):
    book = make_book("Y", [(0.49, 500)], [(0.51, 500)])
    mk, m = _maker(cfg, db, book)
    t0 = time.time()
    mk.cycle(t0)
    mk.api.books["Y"] = make_book("Y", [(0.59, 500)], [(0.61, 500)])
    mk.cycle(t0 + 15)
    mm = mk.markets[m.condition_id]
    assert mm.yes_q is None and mm.no_q is None and "moving fast" in mm.paused


def test_rewards_credited_at_day_roll_with_minimum(cfg, db):
    book = make_book("Y", [(0.49, 500)], [(0.51, 500)])
    mk, _ = _maker(cfg, db, book)
    db.execute("INSERT INTO mm_rewards(day,condition_id,accrued) VALUES('2000-01-01','c',3.5)")
    db.execute("INSERT INTO mm_rewards(day,condition_id,accrued) VALUES('2000-01-02','c',0.4)")
    cash0 = mk.portfolio.cash
    mk.credit_rewards(time.time())
    assert mk.portfolio.cash == pytest.approx(cash0 + 3.5)  # second day below $1 minimum: unpaid
    assert db.one("SELECT COUNT(*) n FROM mm_rewards WHERE paid=0")["n"] == 0


def test_kill_switch_pulls_quotes(cfg, db):
    book = make_book("Y", [(0.49, 500)], [(0.51, 500)])
    mk, m = _maker(cfg, db, book)
    mk.cycle()
    db.set_state("kill", True)
    mk.cycle()
    mm = mk.markets[m.condition_id]
    assert mm.yes_q is None and mm.no_q is None


def test_live_mode_refused(cfg):
    cfg.mode = "live"
    with pytest.raises(RuntimeError):
        MarketMaker(cfg, api=MMApi({}, []))
