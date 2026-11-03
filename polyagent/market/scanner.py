"""Market monitoring: universe selection, order-book snapshots, price history, news."""
from __future__ import annotations

import datetime as dt
import json
import logging
import re
import time
import xml.etree.ElementTree as ET

from ..api.polymarket import Market, OrderBook

log = logging.getLogger(__name__)


def hours_to_close(m: Market, now: float | None = None) -> float | None:
    if not m.end_date:
        return None
    try:
        end = dt.datetime.fromisoformat(m.end_date.replace("Z", "+00:00"))
    except ValueError:
        return None
    return (end.timestamp() - (now or time.time())) / 3600


class MarketScanner:
    def __init__(self, api, db, cfg):
        self.api = api
        self.db = db
        self.cfg = cfg

    def eligible(self, m: Market) -> tuple[bool, str]:
        c = self.cfg
        if m.closed or not m.active or not m.accepting_orders:
            return False, "not accepting orders"
        if len(m.token_ids) != 2:
            return False, "not binary"
        if m.liquidity < c.min_liquidity_usd:
            return False, "low liquidity"
        if m.volume24h < c.min_volume24h_usd:
            return False, "low volume"
        h = hours_to_close(m)
        if h is None or h < c.min_hours_to_close or h > c.max_days_to_close * 24:
            return False, "time to close out of range"
        mid = None
        if m.best_bid is not None and m.best_ask is not None:
            mid = (m.best_bid + m.best_ask) / 2
        elif m.outcome_prices:
            mid = m.outcome_prices[0]
        if mid is None or not (c.price_floor <= mid <= c.price_ceiling):
            return False, "price out of band"
        return True, ""

    def scan(self) -> tuple[list[Market], dict[str, OrderBook]]:
        markets = self.api.active_markets(limit=self.cfg.markets_per_scan * 2)
        self.upsert(markets)
        chosen = [m for m in markets if self.eligible(m)[0]][: self.cfg.markets_per_scan]
        tokens = [t for m in chosen for t in m.token_ids]
        books = self.api.order_books(tokens) if tokens else {}
        now = time.time()
        rows = [(tok, now, b.best_bid, b.best_ask, sum(s for _, s in b.bids[:5]), sum(s for _, s in b.asks[:5]))
                for tok, b in books.items()]
        if rows:
            self.db.executemany("INSERT INTO book_snapshots(token_id,ts,best_bid,best_ask,bid_depth,ask_depth) "
                                "VALUES(?,?,?,?,?,?)", rows)
        return chosen, books

    def upsert(self, markets: list[Market]) -> None:
        now = time.time()
        rows = [(m.condition_id, m.question, m.slug, m.event_id, m.event_slug, int(m.neg_risk), m.end_date,
                 int(m.active), int(m.closed), m.yes_token, m.no_token, json.dumps(m.outcomes),
                 json.dumps(m.outcome_prices), m.fee_rate, m.fee_exponent, m.tick_size, m.min_size,
                 m.liquidity, m.volume24h, now) for m in markets if m.condition_id]
        self.db.executemany(
            "INSERT INTO markets VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(condition_id) DO UPDATE SET "
            "question=excluded.question,active=excluded.active,closed=excluded.closed,"
            "outcome_prices=excluded.outcome_prices,fee_rate=excluded.fee_rate,fee_exponent=excluded.fee_exponent,"
            "tick_size=excluded.tick_size,min_size=excluded.min_size,liquidity=excluded.liquidity,"
            "volume24h=excluded.volume24h,updated_at=excluded.updated_at,end_date=excluded.end_date", rows)

    def history(self, token_id: str, hours: int = 72) -> list[tuple[int, float]]:
        now = int(time.time())
        try:
            h = self.api.price_history(token_id, fidelity=60, start_ts=now - hours * 3600, end_ts=now)
        except Exception as e:  # noqa: BLE001
            log.warning("history %s failed: %s", token_id, e)
            return []
        if h:
            self.db.executemany("INSERT OR IGNORE INTO price_history VALUES(?,?,?)", [(token_id, t, p) for t, p in h])
        return h


_WORD = re.compile(r"[a-z0-9]{4,}")
_STOP = {"will", "before", "after", "with", "than", "more", "less", "2025", "2026", "2027", "election",
         "market", "what", "which", "from", "this", "that", "have", "price", "above", "below", "between"}


def keywords(text: str) -> set[str]:
    return {w for w in _WORD.findall(text.lower()) if w not in _STOP}


class NewsMonitor:
    """Polls RSS/Atom feeds and matches headlines to markets by keyword overlap.

    Headlines are NOT converted into directional probability changes (that would be
    fabricating an edge). A fresh match raises estimate uncertainty instead."""

    def __init__(self, db, cfg, http_get=None):
        self.db = db
        self.cfg = cfg
        import httpx

        self._get = http_get or (lambda url: httpx.get(url, timeout=10, follow_redirects=True).text)

    def poll(self, markets: list[Market]) -> int:
        if not self.cfg.news_feeds:
            return 0
        index = {m.condition_id: keywords(m.question) for m in markets}
        added = 0
        for feed in self.cfg.news_feeds:
            try:
                root = ET.fromstring(self._get(feed))
            except Exception as e:  # noqa: BLE001
                log.warning("news feed %s failed: %s", feed, e)
                continue
            items = root.findall(".//item") or root.findall(".//{http://www.w3.org/2005/Atom}entry")
            for it in items[:50]:
                title = (it.findtext("title") or it.findtext("{http://www.w3.org/2005/Atom}title") or "").strip()
                link = it.findtext("link") or ""
                if not link:
                    el = it.find("{http://www.w3.org/2005/Atom}link")
                    link = el.get("href", "") if el is not None else title
                kw = keywords(title)
                matched = [cid for cid, mk in index.items() if len(kw & mk) >= 2]
                cur = self.db.execute("INSERT OR IGNORE INTO news_items(ts,source,title,link,matched) VALUES(?,?,?,?,?)",
                                      (time.time(), feed, title, link, ",".join(matched)))
                added += cur.rowcount
        return added

    def fresh_news(self, condition_id: str, hours: float = 6) -> bool:
        return bool(self.db.one("SELECT 1 FROM news_items WHERE ts>? AND matched LIKE ?",
                                (time.time() - hours * 3600, f"%{condition_id}%")))
