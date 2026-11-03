"""Read-only Polymarket APIs: Gamma (markets), Data API v2 (wallets), CLOB public (books)."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Iterator

from ..config import Config
from .http import ApiError, HttpClient

log = logging.getLogger(__name__)


def _json_list(v) -> list:
    if isinstance(v, list):
        return v
    if not v:
        return []
    try:
        return json.loads(v)
    except (TypeError, ValueError):
        return []


@dataclass
class Market:
    condition_id: str
    question: str
    slug: str
    event_id: str
    event_slug: str
    neg_risk: bool
    end_date: str
    active: bool
    closed: bool
    outcomes: list[str]
    outcome_prices: list[float]
    token_ids: list[str]
    fee_rate: float
    fee_exponent: float
    tick_size: float
    min_size: float
    liquidity: float
    volume24h: float
    best_bid: float | None = None
    best_ask: float | None = None
    accepting_orders: bool = True
    raw: dict = field(default_factory=dict, repr=False)

    @property
    def yes_token(self) -> str | None:
        return self.token_ids[0] if self.token_ids else None

    @property
    def no_token(self) -> str | None:
        return self.token_ids[1] if len(self.token_ids) > 1 else None

    @classmethod
    def from_gamma(cls, m: dict) -> "Market":
        events = m.get("events") or [{}]
        ev = events[0] if events else {}
        fs = m.get("feeSchedule") or {}
        if isinstance(fs, str):
            try:
                fs = json.loads(fs)
            except ValueError:
                fs = {}
        fees_enabled = bool(m.get("feesEnabled"))
        return cls(
            condition_id=m.get("conditionId", ""),
            question=m.get("question", ""),
            slug=m.get("slug", ""),
            event_id=str(ev.get("id", m.get("conditionId", ""))),
            event_slug=ev.get("slug", ""),
            neg_risk=bool(m.get("negRisk")),
            end_date=m.get("endDate") or "",
            active=bool(m.get("active")),
            closed=bool(m.get("closed")),
            outcomes=_json_list(m.get("outcomes")),
            outcome_prices=[float(x) for x in _json_list(m.get("outcomePrices"))],
            token_ids=[str(x) for x in _json_list(m.get("clobTokenIds"))],
            fee_rate=float(fs.get("rate", 0.0)) if fees_enabled else 0.0,
            fee_exponent=float(fs.get("exponent", 1.0) or 1.0),
            tick_size=float(m.get("orderPriceMinTickSize") or 0.01),
            min_size=float(m.get("orderMinSize") or 5),
            liquidity=float(m.get("liquidityNum") or m.get("liquidity") or 0),
            volume24h=float(m.get("volume24hr") or 0),
            best_bid=float(m["bestBid"]) if m.get("bestBid") not in (None, "") else None,
            best_ask=float(m["bestAsk"]) if m.get("bestAsk") not in (None, "") else None,
            accepting_orders=bool(m.get("acceptingOrders", True)),
            raw=m,
        )


@dataclass
class OrderBook:
    token_id: str
    bids: list[tuple[float, float]]  # (price, size) best first
    asks: list[tuple[float, float]]
    ts: float = 0.0

    @property
    def best_bid(self) -> float | None:
        return self.bids[0][0] if self.bids else None

    @property
    def best_ask(self) -> float | None:
        return self.asks[0][0] if self.asks else None

    @property
    def mid(self) -> float | None:
        if self.best_bid is None or self.best_ask is None:
            return None
        return (self.best_bid + self.best_ask) / 2

    @property
    def spread(self) -> float | None:
        if self.best_bid is None or self.best_ask is None:
            return None
        return self.best_ask - self.best_bid

    @classmethod
    def from_clob(cls, d: dict) -> "OrderBook":
        bids = sorted(((float(x["price"]), float(x["size"])) for x in d.get("bids", [])), key=lambda t: -t[0])
        asks = sorted(((float(x["price"]), float(x["size"])) for x in d.get("asks", [])), key=lambda t: t[0])
        ts = float(d.get("timestamp", 0)) / 1000.0
        return cls(token_id=d.get("asset_id", ""), bids=bids, asks=asks, ts=ts)


class PolymarketData:
    def __init__(self, cfg: Config):
        self.gamma = HttpClient(cfg.gamma_url, cfg.http_timeout_sec, cfg.http_max_retries)
        self.data = HttpClient(cfg.data_url, cfg.http_timeout_sec, cfg.http_max_retries)
        self.clob = HttpClient(cfg.clob_url, cfg.http_timeout_sec, cfg.http_max_retries)

    # --- Gamma -------------------------------------------------------------
    def active_markets(self, limit: int = 200, order: str = "volume24hr") -> list[Market]:
        out: list[Market] = []
        offset = 0
        while len(out) < limit:
            page = self.gamma.get("/markets", {
                "active": "true", "closed": "false", "limit": min(100, limit - len(out)),
                "offset": offset, "order": order, "ascending": "false",
            })
            if not page:
                break
            out.extend(Market.from_gamma(m) for m in page if m.get("enableOrderBook"))
            offset += len(page)
            if len(page) < 100:
                break
        return out

    def closed_markets(self, limit: int = 200, end_date_min: str | None = None,
                       end_date_max: str | None = None) -> list[Market]:
        out: list[Market] = []
        offset = 0
        while len(out) < limit:
            page = self.gamma.get("/markets", {
                "closed": "true", "limit": 100, "offset": offset, "order": "volumeNum",
                "ascending": "false", "end_date_min": end_date_min, "end_date_max": end_date_max,
            })
            if not page:
                break
            out.extend(Market.from_gamma(m) for m in page if m.get("enableOrderBook"))
            offset += len(page)
            if len(page) < 100:
                break
        return out[:limit]

    def market_by_condition(self, condition_id: str) -> Market | None:
        page = self.gamma.get("/markets", {"condition_ids": condition_id})
        return Market.from_gamma(page[0]) if page else None

    # --- CLOB public -------------------------------------------------------
    def order_book(self, token_id: str) -> OrderBook:
        return OrderBook.from_clob(self.clob.get("/book", {"token_id": token_id}))

    def order_books(self, token_ids: list[str]) -> dict[str, OrderBook]:
        out: dict[str, OrderBook] = {}
        for i in range(0, len(token_ids), 100):
            chunk = token_ids[i:i + 100]
            for d in self.clob.post("/books", [{"token_id": t} for t in chunk]) or []:
                b = OrderBook.from_clob(d)
                out[b.token_id] = b
        return out

    def price_history(self, token_id: str, interval: str = "max", fidelity: int = 60,
                      start_ts: int | None = None, end_ts: int | None = None) -> list[tuple[int, float]]:
        params = {"market": token_id, "fidelity": fidelity}
        if start_ts or end_ts:
            params.update(startTs=start_ts, endTs=end_ts)
        else:
            params["interval"] = interval
        d = self.clob.get("/prices-history", params)
        return [(int(h["t"]), float(h["p"])) for h in d.get("history", [])]

    def price_history_range(self, token_id: str, start_ts: int, end_ts: int, fidelity: int = 60,
                            chunk_sec: int = 10 * 86400) -> list[tuple[int, float]]:
        """Long ranges must be chunked: the CLOB rejects windows longer than ~2 weeks."""
        out: dict[int, float] = {}
        t = start_ts
        while t < end_ts:
            hi = min(t + chunk_sec, end_ts)
            try:
                for ts, p in self.price_history(token_id, fidelity=fidelity, start_ts=t, end_ts=hi):
                    out[ts] = p
            except ApiError as e:
                log.warning("history chunk %s %d-%d failed: %s", token_id[:12], t, hi, e)
            t = hi
        return sorted(out.items())

    # --- Data API v2 -------------------------------------------------------
    def _paginate(self, path: str, params: dict, max_items: int) -> Iterator[dict]:
        cursor = None
        n = 0
        while n < max_items:
            p = dict(params, limit=min(500, max_items - n))
            if cursor:
                p["cursor"] = cursor
            try:
                d = self.data.get(path, p)
            except ApiError as e:
                log.warning("data-api %s failed: %s", path, e)
                return
            rows = d.get("data", []) if isinstance(d, dict) else d
            for r in rows:
                yield r
                n += 1
            pag = d.get("pagination", {}) if isinstance(d, dict) else {}
            cursor = pag.get("next_cursor")
            if not rows or not pag.get("has_more") or not cursor:
                return

    def leaderboard(self, period: str = "all", order_by: str = "PNL", n: int = 200) -> list[dict]:
        return list(self._paginate("/v2/leaderboard", {"timePeriod": period, "orderBy": order_by}, n))

    def wallet_trades(self, wallet: str, max_items: int = 2000) -> list[dict]:
        return list(self._paginate("/v2/trades", {"user": wallet}, max_items))

    def market_trades(self, condition_id: str, max_items: int = 5000, min_usd: float | None = None) -> list[dict]:
        """Newest first. The API has no time window for market feeds, so for long-lived markets a
        cap only reaches back a short time; `min_usd` (CASH filter) keeps coverage of large flow."""
        params = {"condition": condition_id}
        if min_usd:
            params.update(filter_type="CASH", filter_amount=min_usd)
        return list(self._paginate("/v2/trades", params, max_items))

    def recent_trades(self, max_items: int = 500) -> list[dict]:
        return list(self._paginate("/v2/trades", {}, max_items))

    def wallet_positions(self, wallet: str, status: str | None = None, max_items: int = 1000,
                         sort_by: str | None = "TIMESTAMP") -> list[dict]:
        """NOTE: the API's default sort is REALIZED_PNL DESC. Truncating that list keeps only a
        wallet's best trades (survivorship bias), so we sort by recency instead."""
        return list(self._paginate("/v2/positions", {"user": wallet, "status": status, "sort_by": sort_by,
                                                     "sort_direction": "DESC" if sort_by else None}, max_items))

    def wallet_value(self, wallet: str) -> float:
        d = self.data.get("/v2/value", {"user": wallet})
        return float((d.get("data") or {}).get("value", 0.0))
