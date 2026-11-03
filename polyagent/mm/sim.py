"""Paper fills for resting maker bids, driven by the real public trade tape.

A resting bid on token T at price b is filled by:
  * a taker SELL of T at price < b      (it traded through us: we were better priced) -> fill
  * a taker SELL of T at price == b     -> fill only after the queue ahead of us is consumed
  * a taker BUY of the complement at price > 1 - b   (matched against our bid via minting) -> fill
  * a taker BUY of the complement at price == 1 - b  -> queue rule as above

Queue-ahead is the displayed size at our price when we placed (others' cancellations are
ignored, which makes equal-price fills rarer, i.e. conservative on spread capture, while
through-trades - the adverse ones - always fill us).
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Quote:
    token_id: str
    complement_id: str
    price: float
    size: float  # remaining shares
    placed_ts: float
    queue_ahead: float
    cancelled_ts: float | None = None
    filled: float = 0.0

    def live_at(self, ts: float) -> bool:
        return self.placed_ts <= ts and (self.cancelled_ts is None or ts < self.cancelled_ts) and self.size > 1e-9


@dataclass
class TapeTrade:
    key: str
    token_id: str
    side: str
    price: float
    size: float
    ts: float


@dataclass
class SimFill:
    token_id: str
    price: float
    size: float
    ts: float
    trade_key: str


@dataclass
class FillSimulator:
    quotes: list[Quote] = field(default_factory=list)
    seen: set[str] = field(default_factory=set)
    tick: float = 0.01

    def place(self, q: Quote) -> None:
        self.quotes.append(q)

    def cancel(self, token_id: str, ts: float) -> None:
        for q in self.quotes:
            if q.token_id == token_id and q.cancelled_ts is None:
                q.cancelled_ts = ts

    def prune(self, before_ts: float) -> None:
        self.quotes = [q for q in self.quotes if q.cancelled_ts is None or q.cancelled_ts >= before_ts]

    def on_trade(self, t: TapeTrade) -> list[SimFill]:
        if t.key in self.seen:
            return []
        self.seen.add(t.key)
        eps = self.tick / 10
        fills: list[SimFill] = []
        for q in self.quotes:
            if not q.live_at(t.ts):
                continue
            if t.token_id == q.token_id and t.side == "SELL":
                eff = t.price
            elif t.token_id == q.complement_id and t.side == "BUY":
                eff = 1 - t.price  # their buy of the complement at p crosses our bid at >= 1-p
            else:
                continue
            avail = t.size
            if eff < q.price - eps:
                pass  # traded through us
            elif abs(eff - q.price) <= eps:
                eat = min(q.queue_ahead, avail)
                q.queue_ahead -= eat
                avail -= eat
            else:
                continue
            take = min(avail, q.size)
            if take > 1e-9:
                q.size -= take
                q.filled += take
                fills.append(SimFill(q.token_id, q.price, take, t.ts, t.key))
        return fills
