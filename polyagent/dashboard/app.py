"""Monitoring dashboard: live P&L, trade history, positions, health, kill switch."""
from __future__ import annotations

import time
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import FileResponse

from ..config import Config
from ..db import Database

STATIC = Path(__file__).parent / "static"


def create_app(cfg: Config, db: Database | None = None) -> FastAPI:
    db = db or Database(cfg.db_path)
    app = FastAPI(title="PolyAgent", docs_url=None, redoc_url=None)

    def _auth(token: str | None) -> None:
        if cfg.dashboard_token and token != cfg.dashboard_token:
            raise HTTPException(401, "bad token")

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/api/summary")
    def summary():
        snap = db.one("SELECT * FROM equity_snapshots ORDER BY ts DESC LIMIT 1")
        start = db.get_state("starting_capital", cfg.starting_capital)
        hb = db.get_state("heartbeat") or {}
        realized = db.one("SELECT COALESCE(SUM(realized_pnl),0) r, COALESCE(SUM(fee),0) f, COUNT(*) n FROM fills")
        open_orders = db.one("SELECT COUNT(*) n FROM orders WHERE status IN ('OPEN','PENDING')")["n"]
        return {
            "mode": hb.get("mode", cfg.mode),
            "starting_capital": start,
            "equity": snap["equity"] if snap else start,
            "cash": snap["cash"] if snap else start,
            "positions_value": snap["positions_value"] if snap else 0.0,
            "pnl_usd": snap["pnl_usd"] if snap else 0.0,
            "pnl_pct": snap["pnl_pct"] if snap else 0.0,
            "drawdown": snap["drawdown"] if snap else 0.0,
            "realized_pnl": realized["r"], "fees_paid": realized["f"], "n_fills": realized["n"],
            "open_orders": open_orders,
            "heartbeat_age_sec": round(time.time() - hb["ts"], 1) if hb.get("ts") else None,
            "cycle": hb.get("cycle"),
            "halt": db.get_state("halt"),
            "kill": bool(db.get_state("kill")) or Path(cfg.kill_file).exists(),
            "last_error": db.get_state("last_error"),
            "updated": snap["ts"] if snap else None,
        }

    @app.get("/api/equity")
    def equity(limit: int = 500):
        rows = db.query("SELECT ts, equity, pnl_usd, pnl_pct FROM equity_snapshots ORDER BY ts DESC LIMIT ?",
                        (min(limit, 5000),))
        return list(reversed(rows))

    @app.get("/api/trades")
    def trades(limit: int = 100):
        return db.query(
            "SELECT f.ts, f.side, f.price, f.size, f.fee, f.realized_pnl, f.question, f.outcome, "
            "f.token_id, o.strategy, o.mode FROM fills f LEFT JOIN orders o ON o.id=f.order_id "
            "ORDER BY f.ts DESC LIMIT ?", (min(limit, 1000),))

    @app.get("/api/positions")
    def positions():
        rows = db.query(
            "SELECT p.*, m.question, (SELECT (best_bid+best_ask)/2 FROM book_snapshots b WHERE b.token_id=p.token_id "
            "ORDER BY ts DESC LIMIT 1) AS mark, CASE WHEN p.token_id=m.yes_token THEN 'YES' ELSE 'NO' END AS outcome "
            "FROM positions p LEFT JOIN markets m ON m.condition_id=p.condition_id WHERE p.size > 1e-6")
        for r in rows:
            mark = r["mark"] if r["mark"] is not None else r["avg_cost"]
            r["unrealized"] = r["size"] * (mark - r["avg_cost"])
        return rows

    @app.get("/api/signals")
    def signals(limit: int = 50):
        return db.query("SELECT * FROM signals ORDER BY ts DESC LIMIT ?", (min(limit, 500),))

    @app.get("/api/wallets")
    def wallets(limit: int = 25):
        return db.query("SELECT wallet, n_trades, n_resolved, realized_pnl, roi, win_rate, win_rate_lcb, "
                        "max_drawdown, sharpe, score, reliable, flags FROM wallet_scores "
                        "ORDER BY reliable DESC, score DESC LIMIT ?", (min(limit, 500),))

    @app.get("/api/audit")
    def audit(limit: int = 50):
        return db.query("SELECT * FROM audit_log ORDER BY id DESC LIMIT ?", (min(limit, 500),))

    @app.post("/api/kill")
    def kill(x_token: str | None = Header(default=None)):
        _auth(x_token)
        db.set_state("kill", True)
        db.set_state("kill_handled", False)
        Path(cfg.kill_file).parent.mkdir(parents=True, exist_ok=True)
        Path(cfg.kill_file).touch()
        db.audit("kill_requested", {"via": "dashboard"}, level="CRITICAL")
        return {"ok": True}

    @app.post("/api/resume")
    def resume(x_token: str | None = Header(default=None)):
        _auth(x_token)
        db.set_state("kill", False)
        db.set_state("kill_handled", False)
        db.set_state("halt", None)
        Path(cfg.kill_file).unlink(missing_ok=True)
        db.audit("resume_requested", {"via": "dashboard"}, level="WARNING")
        return {"ok": True}

    @app.get("/healthz")
    def healthz():
        hb = db.get_state("heartbeat") or {}
        age = time.time() - hb.get("ts", 0)
        stale = age > 5 * cfg.scan_interval_sec
        if stale:
            raise HTTPException(503, f"agent heartbeat stale ({age:.0f}s)")
        return {"ok": True, "heartbeat_age_sec": round(age, 1)}

    return app
