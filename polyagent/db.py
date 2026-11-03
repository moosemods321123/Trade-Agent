"""SQLite persistence (WAL mode so the dashboard can read while the agent writes)."""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from contextlib import contextmanager
from typing import Any, Iterator

SCHEMA = """
CREATE TABLE IF NOT EXISTS markets (
    condition_id   TEXT PRIMARY KEY,
    question       TEXT,
    slug           TEXT,
    event_id       TEXT,
    event_slug     TEXT,
    neg_risk       INTEGER,
    end_date       TEXT,
    active         INTEGER,
    closed         INTEGER,
    yes_token      TEXT,
    no_token       TEXT,
    outcomes       TEXT,
    outcome_prices TEXT,
    fee_rate       REAL,
    fee_exponent   REAL,
    tick_size      REAL,
    min_size       REAL,
    liquidity      REAL,
    volume24h      REAL,
    updated_at     REAL
);
CREATE INDEX IF NOT EXISTS ix_markets_event ON markets(event_id);

CREATE TABLE IF NOT EXISTS book_snapshots (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    token_id   TEXT NOT NULL,
    ts         REAL NOT NULL,
    best_bid   REAL,
    best_ask   REAL,
    bid_depth  REAL,
    ask_depth  REAL
);
CREATE INDEX IF NOT EXISTS ix_book_token_ts ON book_snapshots(token_id, ts);

CREATE TABLE IF NOT EXISTS price_history (
    token_id TEXT NOT NULL,
    ts       INTEGER NOT NULL,
    price    REAL NOT NULL,
    PRIMARY KEY (token_id, ts)
);

CREATE TABLE IF NOT EXISTS wallet_trades (
    tx_hash      TEXT NOT NULL,
    wallet       TEXT NOT NULL,
    token_id     TEXT NOT NULL,
    condition_id TEXT,
    side         TEXT NOT NULL,
    size         REAL NOT NULL,
    price        REAL NOT NULL,
    ts           INTEGER NOT NULL,
    outcome_index INTEGER,
    PRIMARY KEY (tx_hash, wallet, token_id, side, size, price)
);
CREATE INDEX IF NOT EXISTS ix_wt_wallet_ts ON wallet_trades(wallet, ts);
CREATE INDEX IF NOT EXISTS ix_wt_token_ts ON wallet_trades(token_id, ts);

CREATE TABLE IF NOT EXISTS wallet_scores (
    wallet           TEXT PRIMARY KEY,
    computed_at      REAL,
    n_trades         INTEGER,
    n_markets        INTEGER,
    n_resolved       INTEGER,
    days_active      REAL,
    realized_pnl     REAL,
    unrealized_pnl   REAL,
    roi              REAL,
    win_rate         REAL,
    win_rate_lcb     REAL,
    max_drawdown     REAL,
    trades_per_day   REAL,
    avg_hold_hours   REAL,
    top_market_share REAL,
    hhi              REAL,
    sharpe           REAL,
    score            REAL,
    reliable         INTEGER,
    flags            TEXT
);

CREATE TABLE IF NOT EXISTS signals (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    ts           REAL,
    strategy     TEXT,
    condition_id TEXT,
    token_id     TEXT,
    side         TEXT,
    p_market     REAL,
    p_est        REAL,
    p_std        REAL,
    edge         REAL,
    ev_per_share REAL,
    confidence   REAL,
    action       TEXT,
    reason       TEXT
);
CREATE INDEX IF NOT EXISTS ix_signals_ts ON signals(ts);

CREATE TABLE IF NOT EXISTS orders (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    client_key        TEXT NOT NULL UNIQUE,
    exchange_order_id TEXT,
    ts_created        REAL,
    ts_updated        REAL,
    token_id          TEXT,
    condition_id      TEXT,
    side              TEXT,
    order_type        TEXT,
    price             REAL,
    size              REAL,
    filled_size       REAL DEFAULT 0,
    avg_fill_price    REAL,
    fees_paid         REAL DEFAULT 0,
    status            TEXT,
    mode              TEXT,
    strategy          TEXT,
    error             TEXT
);
CREATE INDEX IF NOT EXISTS ix_orders_status ON orders(status);

CREATE TABLE IF NOT EXISTS fills (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    order_id          INTEGER REFERENCES orders(id),
    exchange_trade_id TEXT UNIQUE,
    ts                REAL,
    token_id          TEXT,
    condition_id      TEXT,
    side              TEXT,
    price             REAL,
    size              REAL,
    fee               REAL,
    realized_pnl      REAL DEFAULT 0,
    question          TEXT,
    outcome           TEXT
);
CREATE INDEX IF NOT EXISTS ix_fills_ts ON fills(ts);

CREATE TABLE IF NOT EXISTS positions (
    token_id     TEXT PRIMARY KEY,
    condition_id TEXT,
    event_id     TEXT,
    size         REAL,
    avg_cost     REAL,
    realized_pnl REAL DEFAULT 0,
    updated_at   REAL
);

CREATE TABLE IF NOT EXISTS equity_snapshots (
    ts              REAL PRIMARY KEY,
    cash            REAL,
    positions_value REAL,
    equity          REAL,
    pnl_usd         REAL,
    pnl_pct         REAL,
    high_water      REAL,
    drawdown        REAL
);

CREATE TABLE IF NOT EXISTS audit_log (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    ts      REAL,
    level   TEXT,
    event   TEXT,
    payload TEXT
);
CREATE INDEX IF NOT EXISTS ix_audit_ts ON audit_log(ts);

CREATE TABLE IF NOT EXISTS news_items (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    ts         REAL,
    source     TEXT,
    title      TEXT,
    link       TEXT UNIQUE,
    matched    TEXT
);

CREATE TABLE IF NOT EXISTS mm_quotes (
    condition_id TEXT,
    question     TEXT,
    side         TEXT,
    price        REAL,
    size         REAL,
    queue_ahead  REAL,
    pool_per_day REAL,
    paused       TEXT,
    ts           REAL
);

CREATE TABLE IF NOT EXISTS mm_rewards (
    day          TEXT,
    condition_id TEXT,
    accrued      REAL DEFAULT 0,
    seconds      REAL DEFAULT 0,
    share_sum    REAL DEFAULT 0,
    paid         INTEGER DEFAULT 0,
    PRIMARY KEY (day, condition_id)
);

CREATE TABLE IF NOT EXISTS system_state (
    key   TEXT PRIMARY KEY,
    value TEXT
);
"""


class Database:
    def __init__(self, path: str):
        self.path = path
        self._lock = threading.RLock()
        self.conn = sqlite3.connect(path, check_same_thread=False, timeout=30, isolation_level=None)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.executescript(SCHEMA)
        self._migrate()

    def _migrate(self) -> None:
        cols = {r[1] for r in self.conn.execute("PRAGMA table_info(fills)")}
        if "strategy" not in cols:
            self.conn.execute("ALTER TABLE fills ADD COLUMN strategy TEXT")

    @contextmanager
    def tx(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            self.conn.execute("BEGIN IMMEDIATE")
            try:
                yield self.conn
                self.conn.execute("COMMIT")
            except BaseException:
                self.conn.execute("ROLLBACK")
                raise

    def execute(self, sql: str, params: tuple | dict = ()) -> sqlite3.Cursor:
        with self._lock:
            return self.conn.execute(sql, params)

    def executemany(self, sql: str, rows: list) -> None:
        with self.tx() as c:
            c.executemany(sql, rows)

    def query(self, sql: str, params: tuple | dict = ()) -> list[dict]:
        with self._lock:
            return [dict(r) for r in self.conn.execute(sql, params).fetchall()]

    def one(self, sql: str, params: tuple | dict = ()) -> dict | None:
        rows = self.query(sql, params)
        return rows[0] if rows else None

    # --- key/value state ---------------------------------------------------
    def get_state(self, key: str, default: Any = None) -> Any:
        row = self.one("SELECT value FROM system_state WHERE key=?", (key,))
        return json.loads(row["value"]) if row else default

    def set_state(self, key: str, value: Any) -> None:
        self.execute(
            "INSERT INTO system_state(key,value) VALUES(?,?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, json.dumps(value)),
        )

    # --- audit -------------------------------------------------------------
    def audit(self, event: str, payload: Any = None, level: str = "INFO") -> None:
        self.execute(
            "INSERT INTO audit_log(ts,level,event,payload) VALUES(?,?,?,?)",
            (time.time(), level, event, json.dumps(payload, default=str)),
        )
