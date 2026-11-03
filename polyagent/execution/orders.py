"""Order lifecycle: idempotent submission, fill verification, stale-order cancellation,
and reconciliation against the exchange. Every state change is audit-logged."""
from __future__ import annotations

import hashlib
import logging
import sqlite3
import time

from .brokers import BrokerOrder, OrderRequest
from .portfolio import Portfolio

log = logging.getLogger(__name__)


def client_key(strategy: str, token_id: str, side: str, bucket_sec: int = 600, now: float | None = None) -> str:
    """Deterministic key: the same intent within a time bucket maps to one order."""
    bucket = int((now or time.time()) // bucket_sec)
    return hashlib.sha256(f"{strategy}|{token_id}|{side}|{bucket}".encode()).hexdigest()[:32]


class OrderManager:
    def __init__(self, db, broker, portfolio: Portfolio, cfg):
        self.db = db
        self.broker = broker
        self.portfolio = portfolio
        self.cfg = cfg

    def has_open_order(self, token_id: str) -> bool:
        return bool(self.db.one("SELECT 1 FROM orders WHERE token_id=? AND status IN ('PENDING','OPEN')", (token_id,)))

    def submit(self, req: OrderRequest, *, condition_id: str, event_id: str, strategy: str,
               key: str | None = None, question: str = "", outcome: str = "") -> BrokerOrder | None:
        key = key or client_key(strategy, req.token_id, req.side)
        if self.has_open_order(req.token_id):
            self.db.audit("order_skipped_open_exists", {"token_id": req.token_id})
            return None
        now = time.time()
        try:
            # Reserve the key BEFORE touching the exchange: a crash after this point
            # leaves a PENDING row that reconciliation resolves, never a duplicate.
            cur = self.db.execute(
                "INSERT INTO orders(client_key,ts_created,ts_updated,token_id,condition_id,side,order_type,"
                "price,size,status,mode,strategy) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                (key, now, now, req.token_id, condition_id, req.side, req.order_type, req.price, req.size,
                 "PENDING", self.broker.mode, strategy))
        except sqlite3.IntegrityError:
            self.db.audit("order_duplicate_blocked", {"client_key": key, "token_id": req.token_id})
            return None
        row_id = cur.lastrowid
        self.db.audit("order_submit", {"id": row_id, "key": key, **req.__dict__, "strategy": strategy})
        result = self.broker.place(req)
        self._record(row_id, result, condition_id, event_id, question, outcome)
        return result

    def _record(self, row_id: int, res: BrokerOrder, condition_id: str, event_id: str,
                question: str = "", outcome: str = "") -> None:
        new_fills = 0
        for f in res.fills:
            try:
                self.db.execute(
                    "INSERT INTO fills(order_id,exchange_trade_id,ts,token_id,condition_id,side,price,size,fee,"
                    "question,outcome) SELECT ?,?,?,token_id,condition_id,side,?,?,?,?,? FROM orders WHERE id=?",
                    (row_id, f.trade_id, f.ts, f.price, f.size, f.fee, question, outcome, row_id))
            except sqlite3.IntegrityError:
                continue  # already recorded this trade id
            o = self.db.one("SELECT token_id, side FROM orders WHERE id=?", (row_id,))
            pnl = self.portfolio.apply_fill(token_id=o["token_id"], condition_id=condition_id, event_id=event_id,
                                            side=o["side"], price=f.price, size=f.size, fee=f.fee)
            self.db.execute("UPDATE fills SET realized_pnl=? WHERE exchange_trade_id=?", (pnl, f.trade_id))
            new_fills += 1
        agg = self.db.one("SELECT COALESCE(SUM(size),0) s, COALESCE(SUM(price*size),0) n, COALESCE(SUM(fee),0) f "
                          "FROM fills WHERE order_id=?", (row_id,))
        avg = agg["n"] / agg["s"] if agg["s"] else None
        self.db.execute(
            "UPDATE orders SET exchange_order_id=COALESCE(NULLIF(?, ''), exchange_order_id), status=?, filled_size=?,"
            " avg_fill_price=?, fees_paid=?, ts_updated=?, error=? WHERE id=?",
            (res.order_id, res.status, agg["s"], avg, agg["f"], time.time(), res.error or None, row_id))
        self.db.audit("order_update", {"id": row_id, "status": res.status, "filled": agg["s"],
                                       "new_fills": new_fills, "error": res.error})

    def monitor(self) -> None:
        """Poll open orders for fills; cancel ones older than order_stale_sec."""
        for o in self.db.query("SELECT * FROM orders WHERE status IN ('OPEN','PENDING')"):
            ev = self.db.one("SELECT event_id FROM markets WHERE condition_id=?", (o["condition_id"],))
            event_id = ev["event_id"] if ev else o["condition_id"]
            if not o["exchange_order_id"]:
                # PENDING with no exchange id: crashed between reserve and place, or rejected pre-id.
                if time.time() - o["ts_created"] > 60:
                    self.db.execute("UPDATE orders SET status='CANCELLED', error='orphaned pending' WHERE id=?",
                                    (o["id"],))
                    self.db.audit("order_orphan_cancelled", {"id": o["id"]}, level="WARNING")
                continue
            req = OrderRequest(o["token_id"], o["side"], o["price"], o["size"], o["order_type"],
                               *self._fee(o["condition_id"]))
            res = self.broker.get_order(o["exchange_order_id"], req)
            if res:
                self._record(o["id"], res, o["condition_id"], event_id)
                if res.status not in ("OPEN", "PENDING"):
                    continue
            if time.time() - o["ts_created"] > self.cfg.order_stale_sec:
                if self.broker.cancel(o["exchange_order_id"]):
                    self.db.execute("UPDATE orders SET status=?, ts_updated=? WHERE id=?",
                                    ("PARTIAL" if (o["filled_size"] or 0) > 0 else "CANCELLED", time.time(), o["id"]))
                    self.db.audit("order_stale_cancelled", {"id": o["id"]})

    def _fee(self, cid: str) -> tuple[float, float]:
        m = self.db.one("SELECT fee_rate, fee_exponent FROM markets WHERE condition_id=?", (cid,))
        return (m["fee_rate"] or 0.0, m["fee_exponent"] or 1.0) if m else (0.0, 1.0)

    def cancel_all(self, reason: str) -> None:
        ok = self.broker.cancel_all()
        self.db.execute("UPDATE orders SET status='CANCELLED', ts_updated=?, error=? "
                        "WHERE status IN ('OPEN','PENDING')", (time.time(), reason))
        self.db.audit("cancel_all", {"reason": reason, "broker_ok": ok}, level="CRITICAL")

    def reconcile_live(self, api, funder: str) -> list[dict]:
        """Compare local positions with the exchange's view; adopt exchange truth on mismatch."""
        diffs = []
        remote = {str(p["token_id"]): p for p in api.wallet_positions(funder, status="OPEN")}
        local = {p["token_id"]: p for p in self.portfolio.positions()}
        for tok in set(remote) | set(local):
            r_size = float(remote[tok]["current_size"]) if tok in remote else 0.0
            l_size = float(local[tok]["size"]) if tok in local else 0.0
            if abs(r_size - l_size) > 0.01:
                rp = remote.get(tok, {})
                diffs.append({"token_id": tok, "local": l_size, "remote": r_size})
                self.db.execute(
                    "INSERT INTO positions(token_id,condition_id,event_id,size,avg_cost,realized_pnl,updated_at) "
                    "VALUES(?,?,?,?,?,0,?) ON CONFLICT(token_id) DO UPDATE SET size=excluded.size,"
                    "avg_cost=excluded.avg_cost,updated_at=excluded.updated_at",
                    (tok, rp.get("condition_id") or (local.get(tok) or {}).get("condition_id"),
                     rp.get("event_id") or (local.get(tok) or {}).get("event_id"), r_size,
                     float(rp.get("avg_price") or (local.get(tok) or {}).get("avg_cost") or 0), time.time()))
        bal = self.broker.collateral_balance()
        if bal is not None and abs(bal - self.portfolio.cash) > 0.01:
            diffs.append({"cash_local": self.portfolio.cash, "cash_remote": bal})
            self.portfolio.set_cash(bal)
        if diffs:
            self.db.audit("reconcile_mismatch", diffs, level="WARNING")
        return diffs
