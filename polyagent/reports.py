"""Performance reports generated from the database (no estimates, only recorded data)."""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

from .analytics.stats import max_drawdown


def performance_report(db) -> dict:
    start = db.get_state("starting_capital", 0.0)
    snaps = db.query("SELECT ts, equity FROM equity_snapshots ORDER BY ts")
    last = db.one("SELECT * FROM equity_snapshots ORDER BY ts DESC LIMIT 1") or {}
    fills = db.query("SELECT f.*, o.strategy FROM fills f LEFT JOIN orders o ON o.id=f.order_id")
    orders = db.query("SELECT status, COUNT(*) n FROM orders GROUP BY status")
    closing = [f for f in fills if f["side"] in ("SELL", "SETTLE")]
    by_strat: dict[str, dict] = {}
    for f in fills:
        s = by_strat.setdefault(f["strategy"] or "settlement", {"fills": 0, "realized_pnl": 0.0, "fees": 0.0})
        s["fills"] += 1
        s["realized_pnl"] += f["realized_pnl"] or 0.0
        s["fees"] += f["fee"] or 0.0
    return {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "mode": (db.get_state("heartbeat") or {}).get("mode"),
        "starting_capital": start,
        "equity": last.get("equity", start),
        "pnl_usd": last.get("pnl_usd", 0.0),
        "pnl_pct": last.get("pnl_pct", 0.0),
        "max_drawdown_pct": max_drawdown([s["equity"] for s in snaps]) if snaps else 0.0,
        "realized_pnl": sum(f["realized_pnl"] or 0 for f in fills),
        "fees_paid": sum(f["fee"] or 0 for f in fills),
        "n_fills": len(fills),
        "closed_trades": len(closing),
        "closed_win_rate": (sum(1 for f in closing if (f["realized_pnl"] or 0) > 0) / len(closing)) if closing else None,
        "orders_by_status": {o["status"]: o["n"] for o in orders},
        "by_strategy": by_strat,
        "first_snapshot": snaps[0]["ts"] if snaps else None,
        "last_snapshot": snaps[-1]["ts"] if snaps else None,
    }


def write_report(db, out_dir: str = "data/reports") -> Path:
    r = performance_report(db)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d-%H%M%S")
    path = Path(out_dir) / f"report-{stamp}.md"
    lines = [f"# PolyAgent performance report ({r['generated_at']})", "",
             f"- Mode: {r['mode']}",
             f"- Starting capital: ${r['starting_capital']:,.2f}",
             f"- Equity: ${r['equity']:,.2f}",
             f"- P&L: ${r['pnl_usd']:,.2f} ({100 * r['pnl_pct']:.2f}%)",
             f"- Max drawdown: {100 * r['max_drawdown_pct']:.2f}%",
             f"- Realized P&L: ${r['realized_pnl']:,.2f}; fees paid ${r['fees_paid']:,.2f}",
             f"- Fills: {r['n_fills']}; closed trades: {r['closed_trades']}; "
             f"win rate: {'n/a' if r['closed_win_rate'] is None else f'{100 * r['closed_win_rate']:.1f}%'}",
             "", "## By strategy", "", "```json", json.dumps(r["by_strategy"], indent=2), "```",
             "", "## Orders by status", "", "```json", json.dumps(r["orders_by_status"], indent=2), "```"]
    path.write_text("\n".join(lines) + "\n")
    return path
