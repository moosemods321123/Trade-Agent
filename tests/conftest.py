import datetime as dt

import pytest

from polyagent.api.polymarket import Market, OrderBook
from polyagent.config import Config
from polyagent.db import Database


@pytest.fixture
def cfg(tmp_path):
    c = Config(db_path=str(tmp_path / "t.db"), kill_file=str(tmp_path / "KILL"), starting_capital=1000.0)
    c.validate()
    return c


@pytest.fixture
def db(cfg):
    return Database(cfg.db_path)


def make_market(**kw) -> Market:
    base = dict(condition_id="0xc1", question="Will X happen?", slug="x", event_id="ev1", event_slug="ev",
                neg_risk=False, end_date=(dt.datetime.now(dt.timezone.utc) + dt.timedelta(days=10)).isoformat(), active=True, closed=False,
                outcomes=["Yes", "No"], outcome_prices=[0.5, 0.5], token_ids=["YES", "NO"], fee_rate=0.04,
                fee_exponent=1.0, tick_size=0.01, min_size=5, liquidity=50000, volume24h=10000,
                best_bid=0.49, best_ask=0.51)
    base.update(kw)
    return Market(**base)


def make_book(token, bids, asks) -> OrderBook:
    return OrderBook(token, sorted(bids, key=lambda x: -x[0]), sorted(asks, key=lambda x: x[0]))


class FakeApi:
    """Stand-in for PolymarketData; tests set books directly."""

    def __init__(self, books=None, markets=None):
        self.books = books or {}
        self.markets = markets or []

    def order_book(self, token_id):
        return self.books[token_id]

    def order_books(self, token_ids):
        return {t: self.books[t] for t in token_ids if t in self.books}

    def active_markets(self, limit=200, order="volume24hr"):
        return self.markets

    def market_by_condition(self, cid):
        return next((m for m in self.markets if m.condition_id == cid), None)

    def wallet_trades(self, w, max_items=100):
        return []

    def price_history(self, *a, **k):
        return []

    def leaderboard(self, **k):
        return []

    def recent_trades(self, **k):
        return []
