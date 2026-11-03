"""Broker adapters. PaperBroker simulates taker fills against the *live* order book;
LiveBroker wraps the official py-clob-client. Both report actual fills — callers
never assume a submitted order executed."""
from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field

from ..analytics.costs import taker_fee, walk_book

log = logging.getLogger(__name__)


@dataclass
class OrderRequest:
    token_id: str
    side: str  # BUY / SELL
    price: float  # limit price
    size: float  # shares
    order_type: str = "FAK"  # FAK | FOK | GTC
    fee_rate: float = 0.0
    fee_exponent: float = 1.0


@dataclass
class BrokerFill:
    trade_id: str
    price: float
    size: float
    fee: float
    ts: float


@dataclass
class BrokerOrder:
    order_id: str
    status: str  # FILLED | PARTIAL | OPEN | CANCELLED | REJECTED
    filled_size: float = 0.0
    fills: list[BrokerFill] = field(default_factory=list)
    error: str = ""

    @property
    def avg_price(self) -> float:
        s = sum(f.size for f in self.fills)
        return sum(f.price * f.size for f in self.fills) / s if s else 0.0


class PaperBroker:
    mode = "paper"

    def __init__(self, api):
        self.api = api

    def place(self, req: OrderRequest) -> BrokerOrder:
        oid = "paper-" + uuid.uuid4().hex[:16]
        if req.order_type == "GTC":
            return BrokerOrder(oid, "REJECTED", error="paper broker only simulates taker orders")
        try:
            book = self.api.order_book(req.token_id)
        except Exception as e:  # noqa: BLE001
            return BrokerOrder(oid, "REJECTED", error=f"book fetch failed: {e}")
        levels = book.asks if req.side == "BUY" else book.bids
        est = walk_book(levels, req.side, shares=req.size, limit_price=req.price,
                        fee_rate=req.fee_rate, fee_exponent=req.fee_exponent)
        if est.shares <= 1e-9:
            return BrokerOrder(oid, "CANCELLED", error="no liquidity within limit")
        if req.order_type == "FOK" and not est.fully_filled:
            return BrokerOrder(oid, "CANCELLED", error="FOK not fully fillable")
        # one fill per price level, mirroring how matches are reported on-exchange
        fills = []
        remaining = est.shares
        for price, size in levels:
            if remaining <= 1e-9:
                break
            if (req.side == "BUY" and price > req.price + 1e-12) or (req.side == "SELL" and price < req.price - 1e-12):
                break
            take = min(size, remaining)
            fills.append(BrokerFill(f"{oid}-{len(fills)}", price, take,
                                    taker_fee(take, price, req.fee_rate, req.fee_exponent), time.time()))
            remaining -= take
        filled = sum(f.size for f in fills)
        status = "FILLED" if filled >= req.size - 1e-6 else "PARTIAL"
        return BrokerOrder(oid, status, filled, fills)

    def get_order(self, order_id: str, req: OrderRequest | None = None) -> BrokerOrder | None:
        return None  # paper orders are terminal on placement

    def cancel(self, order_id: str) -> bool:
        return True

    def cancel_all(self) -> bool:
        return True

    def collateral_balance(self) -> float | None:
        return None  # ledger is the source of truth in paper mode


class LiveBroker:
    """Thin wrapper around py-clob-client (https://github.com/Polymarket/py-clob-client).

    NOTE: exercised in tests only against a fake client. Validate with tiny sizes
    before trusting it with capital (see README "Going live")."""

    mode = "live"

    def __init__(self, cfg, client=None):
        if client is None:
            from py_clob_client.client import ClobClient

            client = ClobClient(cfg.clob_url, key=cfg.private_key, chain_id=cfg.chain_id,
                                signature_type=cfg.signature_type, funder=cfg.funder_address)
            client.set_api_creds(client.create_or_derive_api_creds())
        self.client = client
        self.cfg = cfg

    def place(self, req: OrderRequest) -> BrokerOrder:
        from py_clob_client.clob_types import OrderArgs, OrderType

        try:
            signed = self.client.create_order(OrderArgs(token_id=req.token_id, price=req.price,
                                                        size=round(req.size, 2), side=req.side))
            resp = self.client.post_order(signed, getattr(OrderType, req.order_type))
        except Exception as e:  # noqa: BLE001
            return BrokerOrder("", "REJECTED", error=str(e)[:500])
        if not resp or not resp.get("success", False):
            return BrokerOrder(resp.get("orderID", "") if resp else "", "REJECTED",
                               error=(resp or {}).get("errorMsg", "unknown error"))
        oid = resp.get("orderID", "")
        # Verify, don't assume: poll the order and its trades.
        time.sleep(1.0)
        return self.get_order(oid, req) or BrokerOrder(oid, "OPEN")

    def get_order(self, order_id: str, req: OrderRequest | None = None) -> BrokerOrder | None:
        from py_clob_client.clob_types import TradeParams

        try:
            o = self.client.get_order(order_id)
        except Exception as e:  # noqa: BLE001
            log.warning("get_order %s failed: %s", order_id, e)
            return None
        if not o:
            return None
        matched = float(o.get("size_matched") or 0)
        original = float(o.get("original_size") or 0)
        raw_status = (o.get("status") or "").upper()
        fills: list[BrokerFill] = []
        trade_ids = o.get("associate_trades") or []
        if trade_ids:
            for tid in trade_ids:
                try:
                    trades = self.client.get_trades(TradeParams(id=tid))
                except Exception as e:  # noqa: BLE001
                    log.warning("get_trades %s failed: %s", tid, e)
                    continue
                for t in trades or []:
                    if (t.get("status") or "").upper() == "FAILED":
                        continue
                    price = float(t.get("price", 0))
                    size = float(t.get("size", 0))
                    fee = taker_fee(size, price, req.fee_rate, req.fee_exponent) if req else 0.0
                    fills.append(BrokerFill(str(t.get("id", tid)), price, size, fee,
                                            float(t.get("match_time") or time.time())))
        filled = sum(f.size for f in fills) if fills else matched
        if raw_status in ("CANCELED", "CANCELLED", "UNMATCHED"):
            status = "PARTIAL" if filled > 0 else "CANCELLED"
        elif original and filled >= original - 1e-6:
            status = "FILLED"
        elif raw_status in ("LIVE", "DELAYED"):
            status = "OPEN"
        else:
            status = "PARTIAL" if filled > 0 else "OPEN"
        return BrokerOrder(order_id, status, filled, fills)

    def cancel(self, order_id: str) -> bool:
        try:
            self.client.cancel(order_id)
            return True
        except Exception as e:  # noqa: BLE001
            log.error("cancel %s failed: %s", order_id, e)
            return False

    def cancel_all(self) -> bool:
        try:
            self.client.cancel_all()
            return True
        except Exception as e:  # noqa: BLE001
            log.error("cancel_all failed: %s", e)
            return False

    def collateral_balance(self) -> float | None:
        from py_clob_client.clob_types import AssetType, BalanceAllowanceParams

        try:
            r = self.client.get_balance_allowance(BalanceAllowanceParams(
                asset_type=AssetType.COLLATERAL, signature_type=self.cfg.signature_type))
            return float(r.get("balance", 0)) / 1e6
        except Exception as e:  # noqa: BLE001
            log.warning("balance fetch failed: %s", e)
            return None
