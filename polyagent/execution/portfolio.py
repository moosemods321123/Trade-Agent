"""Local ledger: cash, positions (avg cost), realized P&L, equity snapshots."""
from __future__ import annotations

import datetime as dt
import time

from ..risk.manager import PortfolioState


class Portfolio:
    def __init__(self, db, starting_capital: float):
        self.db = db
        if self.db.get_state("starting_capital") is None:
            self.db.set_state("starting_capital", starting_capital)
            self.db.set_state("cash", starting_capital)
            self.db.audit("ledger_init", {"starting_capital": starting_capital})

    @property
    def starting_capital(self) -> float:
        return float(self.db.get_state("starting_capital", 0.0))

    @property
    def cash(self) -> float:
        return float(self.db.get_state("cash", 0.0))

    def set_cash(self, value: float) -> None:
        self.db.set_state("cash", round(value, 6))

    def positions(self) -> list[dict]:
        return self.db.query("SELECT * FROM positions WHERE size > 1e-6")

    def apply_fill(self, *, token_id: str, condition_id: str, event_id: str, side: str,
                   price: float, size: float, fee: float, adjust_cash: bool = True) -> float:
        """Update position and cash for one fill. Returns realized P&L of this fill."""
        pos = self.db.one("SELECT * FROM positions WHERE token_id=?", (token_id,))
        cur_size = pos["size"] if pos else 0.0
        avg = pos["avg_cost"] if pos else 0.0
        realized_total = pos["realized_pnl"] if pos else 0.0
        realized = 0.0
        if side == "BUY":
            new_size = cur_size + size
            # fees are capitalised into cost basis on buys
            avg = (cur_size * avg + size * price + fee) / new_size if new_size > 0 else 0.0
            cash_delta = -(size * price + fee)
        else:
            sell = min(size, cur_size)
            realized = sell * (price - avg) - fee
            new_size = cur_size - sell
            cash_delta = sell * price - fee
        realized_total += realized
        self.db.execute(
            "INSERT INTO positions(token_id,condition_id,event_id,size,avg_cost,realized_pnl,updated_at) "
            "VALUES(?,?,?,?,?,?,?) ON CONFLICT(token_id) DO UPDATE SET size=excluded.size,"
            "avg_cost=excluded.avg_cost,realized_pnl=excluded.realized_pnl,updated_at=excluded.updated_at",
            (token_id, condition_id, event_id, new_size, avg, realized_total, time.time()))
        if adjust_cash:
            self.set_cash(self.cash + cash_delta)
        return realized

    def settle(self, token_id: str, payout: float) -> float:
        """Market resolved: each share pays `payout` (0 or 1). Returns realized P&L."""
        pos = self.db.one("SELECT * FROM positions WHERE token_id=?", (token_id,))
        if not pos or pos["size"] <= 1e-9:
            return 0.0
        pnl = pos["size"] * (payout - pos["avg_cost"])
        self.set_cash(self.cash + pos["size"] * payout)
        self.db.execute("UPDATE positions SET size=0, realized_pnl=realized_pnl+?, updated_at=? WHERE token_id=?",
                        (pnl, time.time(), token_id))
        self.db.execute(
            "INSERT INTO fills(order_id,exchange_trade_id,ts,token_id,condition_id,side,price,size,fee,"
            "realized_pnl,question,outcome) VALUES(NULL,?,?,?,?,?,?,?,0,?,?,?)",
            (f"settle:{token_id}", time.time(), token_id, pos["condition_id"], "SETTLE", payout,
             pos["size"], pnl, self._question(pos["condition_id"]), "resolved"))
        self.db.audit("settlement", {"token_id": token_id, "payout": payout, "size": pos["size"], "pnl": pnl})
        return pnl

    def _question(self, cid: str) -> str:
        r = self.db.one("SELECT question FROM markets WHERE condition_id=?", (cid,))
        return r["question"] if r else ""

    def state(self, marks: dict[str, float]) -> PortfolioState:
        positions = self.positions()
        value = 0.0
        by_market: dict[str, float] = {}
        by_event: dict[str, float] = {}
        for p in positions:
            mark = marks.get(p["token_id"], p["avg_cost"])
            value += p["size"] * mark
            basis = p["size"] * p["avg_cost"]
            by_market[p["condition_id"]] = by_market.get(p["condition_id"], 0.0) + basis
            ev = p["event_id"] or p["condition_id"]
            by_event[ev] = by_event.get(ev, 0.0) + basis
        # pending (unfilled) open orders also consume exposure budget
        for o in self.db.query("SELECT condition_id, price, size, filled_size FROM orders "
                               "WHERE status IN ('OPEN','PENDING') AND side='BUY'"):
            rem = (o["size"] - (o["filled_size"] or 0)) * o["price"]
            by_market[o["condition_id"]] = by_market.get(o["condition_id"], 0.0) + rem
        equity = self.cash + value
        hw = max(float(self.db.get_state("high_water", equity)), equity)
        self.db.set_state("high_water", hw)
        today = dt.datetime.now(dt.timezone.utc).date().isoformat()
        day = self.db.get_state("day_start", {})
        if day.get("date") != today:
            day = {"date": today, "equity": equity}
            self.db.set_state("day_start", day)
        return PortfolioState(cash=self.cash, equity=equity, high_water=hw,
                              day_start_equity=day["equity"], exposure_by_market=by_market,
                              exposure_by_event=by_event)

    def snapshot(self, st: PortfolioState) -> dict:
        start = self.starting_capital
        pnl = st.equity - start
        row = {"ts": time.time(), "cash": st.cash, "positions_value": st.equity - st.cash,
               "equity": st.equity, "pnl_usd": pnl, "pnl_pct": pnl / start if start else 0.0,
               "high_water": st.high_water, "drawdown": st.drawdown}
        self.db.execute("INSERT OR REPLACE INTO equity_snapshots VALUES(:ts,:cash,:positions_value,:equity,"
                        ":pnl_usd,:pnl_pct,:high_water,:drawdown)", row)
        return row
