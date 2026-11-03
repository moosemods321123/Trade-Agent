"""The autonomous loop: scan -> estimate -> evaluate -> size -> execute -> reconcile."""
from __future__ import annotations

import json
import logging
import os
import signal
import threading
import time
from pathlib import Path

from .analytics.costs import taker_fee, walk_book
from .api.polymarket import Market, OrderBook, PolymarketData
from .config import Config
from .db import Database
from .execution.brokers import LiveBroker, OrderRequest, PaperBroker
from .execution.orders import OrderManager, client_key
from .execution.portfolio import Portfolio
from .market.scanner import MarketScanner, NewsMonitor
from .risk.manager import RiskManager
from .strategy.engine import Opportunity, evaluate_binary_arbitrage, evaluate_directional
from .strategy.estimators import (EstimatorParams, SmartMoneyEstimator, TrendEstimator,
                                  apply_news_uncertainty)
from .wallets.intelligence import Trade, WalletRanker

log = logging.getLogger("polyagent")


class Agent:
    def __init__(self, cfg: Config, api: PolymarketData | None = None, broker=None, db: Database | None = None):
        self.cfg = cfg
        self.db = db or Database(cfg.db_path)
        self.api = api or PolymarketData(cfg)
        if broker is None:
            broker = LiveBroker(cfg) if cfg.mode == "live" else PaperBroker(self.api)
        self.broker = broker
        start = cfg.starting_capital
        if cfg.mode == "live" and self.db.get_state("starting_capital") is None:
            start = broker.collateral_balance() or 0.0
        self.portfolio = Portfolio(self.db, start)
        self.orders = OrderManager(self.db, broker, self.portfolio, cfg)
        self.risk = RiskManager(cfg, self.db)
        self.scanner = MarketScanner(self.api, self.db, cfg)
        self.news = NewsMonitor(self.db, cfg)
        self.ranker = WalletRanker(self.api, self.db, cfg)
        params = EstimatorParams(min_r_squared=cfg.min_r_squared,
                                 prior_concentration=cfg.smart_prior_concentration,
                                 per_wallet_pseudo=cfg.smart_per_wallet_pseudo,
                                 min_smart_wallets=cfg.smart_min_wallets)
        self.smart = SmartMoneyEstimator(params)
        self.trend = TrendEstimator(params)
        self._stop = threading.Event()
        self._wallet_thread: threading.Thread | None = None
        self._hist_cache: dict[str, tuple[float, list]] = {}
        self.cycle_n = 0

    # --- lifecycle ----------------------------------------------------------
    def run_forever(self) -> None:
        signal.signal(signal.SIGTERM, lambda *_: self._stop.set())
        signal.signal(signal.SIGINT, lambda *_: self._stop.set())
        self.db.audit("agent_start", {"mode": self.cfg.mode, "pid": os.getpid()})
        log.info("agent started in %s mode", self.cfg.mode.upper())
        while not self._stop.is_set():
            t0 = time.time()
            try:
                self.cycle()
                self.db.set_state("last_error", None)
            except Exception as e:  # noqa: BLE001 — never let one bad cycle kill the process
                log.exception("cycle failed")
                self.db.set_state("last_error", {"ts": time.time(), "error": repr(e)[:500]})
                self.db.audit("cycle_error", {"error": repr(e)[:1000]}, level="ERROR")
            self._stop.wait(max(1.0, self.cfg.scan_interval_sec - (time.time() - t0)))
        self.db.audit("agent_stop", {})
        log.info("agent stopped")

    def kill_requested(self) -> bool:
        return Path(self.cfg.kill_file).exists() or bool(self.db.get_state("kill"))

    def cycle(self) -> None:
        self.cycle_n += 1
        self.db.set_state("heartbeat", {"ts": time.time(), "cycle": self.cycle_n, "mode": self.cfg.mode})

        if self.kill_requested():
            if not self.db.get_state("kill_handled"):
                self.orders.cancel_all("manual kill switch")
                self.risk.halt("manual kill switch")
                self.db.set_state("kill_handled", True)
            self._mark_and_snapshot()
            return

        markets, books = self.scanner.scan()
        self.orders.monitor()
        self._settle_resolved()
        if self.cfg.mode == "live" and self.cycle_n % 5 == 1:
            self.orders.reconcile_live(self.api, self.cfg.funder_address)

        st, marks, held_books = self._mark_and_snapshot(books)
        breach = self.risk.check_limits(st)
        if breach:
            self.orders.cancel_all(breach)
            self.risk.halt(breach, until_next_day="daily" in breach)
            return
        halted, why = self.risk.is_halted()
        if halted:
            log.info("halted: %s", why)
            return

        self._maybe_refresh_wallets()
        if self.cycle_n % 5 == 1:
            self.news.poll(markets)
        weights = self.ranker.top_wallets()
        smart_trades = self._smart_flow(weights) if weights else {}

        self._manage_exits(held_books, weights, smart_trades)
        self._find_and_trade(markets, books, weights, smart_trades)
        self._mark_and_snapshot(books)  # post-trade snapshot so the dashboard reflects new fills

    # --- marking ------------------------------------------------------------
    def _mark_and_snapshot(self, books: dict[str, OrderBook] | None = None):
        books = dict(books or {})
        held = [p["token_id"] for p in self.portfolio.positions()]
        missing = [t for t in held if t not in books]
        if missing:
            try:
                books.update(self.api.order_books(missing))
            except Exception as e:  # noqa: BLE001
                log.warning("could not fetch books for held tokens: %s", e)
        marks = {t: b.mid for t, b in books.items() if b.mid is not None}
        st = self.portfolio.state(marks)
        self.portfolio.snapshot(st)
        return st, marks, {t: books[t] for t in held if t in books}

    def _settle_resolved(self) -> None:
        for p in self.portfolio.positions():
            m = self.db.one("SELECT * FROM markets WHERE condition_id=?", (p["condition_id"],))
            if m and not m["closed"] and m["updated_at"] and time.time() - m["updated_at"] < 600:
                continue
            try:
                fresh = self.api.market_by_condition(p["condition_id"])
            except Exception:  # noqa: BLE001
                continue
            if not fresh:
                continue
            self.scanner.upsert([fresh])
            if fresh.closed and fresh.outcome_prices and set(fresh.outcome_prices) <= {0.0, 1.0}:
                idx = fresh.token_ids.index(p["token_id"]) if p["token_id"] in fresh.token_ids else None
                if idx is not None:
                    if self.cfg.mode == "paper":
                        self.portfolio.settle(p["token_id"], fresh.outcome_prices[idx])
                    else:  # live: redemption happens on-chain; record and let reconcile adopt truth
                        self.db.audit("live_resolution_pending_redeem",
                                      {"token_id": p["token_id"], "payout": fresh.outcome_prices[idx]})

    # --- intelligence -------------------------------------------------------
    def _maybe_refresh_wallets(self) -> None:
        last = self.db.get_state("wallet_refresh_ts", 0)
        if time.time() - last < self.cfg.wallet_refresh_minutes * 60:
            return
        if self._wallet_thread and self._wallet_thread.is_alive():
            return
        self.db.set_state("wallet_refresh_ts", time.time())

        def job():
            try:
                self.ranker.refresh()
            except Exception as e:  # noqa: BLE001
                log.exception("wallet refresh failed")
                self.db.audit("wallet_refresh_error", {"error": repr(e)}, level="ERROR")

        self._wallet_thread = threading.Thread(target=job, daemon=True, name="wallet-refresh")
        self._wallet_thread.start()

    def _smart_flow(self, weights: dict[str, float]) -> dict[str, list[Trade]]:
        """condition_id -> recent trades by reliable wallets."""
        cutoff = time.time() - self.cfg.smart_flow_lookback_hours * 3600
        by_market: dict[str, list[Trade]] = {}
        for w in weights:
            try:
                raw = self.api.wallet_trades(w, max_items=100)
            except Exception:  # noqa: BLE001
                continue
            for d in raw:
                t = Trade.from_api(d)
                if t.ts >= cutoff:
                    by_market.setdefault(t.condition_id, []).append(t)
        return by_market

    def _history(self, token_id: str) -> list:
        hit = self._hist_cache.get(token_id)
        if hit and time.time() - hit[0] < 1800:
            return hit[1]
        h = self.scanner.history(token_id, hours=self.trend.p.trend_lookback_hours)
        self._hist_cache[token_id] = (time.time(), h)
        return h

    def _estimates(self, m: Market, yes_book: OrderBook, weights, smart_trades, with_trend: bool):
        ests = []
        if weights and m.condition_id in smart_trades and yes_book.mid is not None:
            e = self.smart.estimate(yes_book.mid, m.yes_token, m.no_token, smart_trades[m.condition_id], weights)
            if e:
                ests.append(e)
        if with_trend:
            e = self.trend.estimate(self._history(m.yes_token), int(time.time()))
            if e:
                ests.append(e)
        fresh = self.news.fresh_news(m.condition_id)
        return [apply_news_uncertainty(e, fresh, self.cfg.news_uncertainty_boost) for e in ests]

    # --- trading ------------------------------------------------------------
    def _log_signal(self, o: Opportunity, action: str) -> None:
        self.db.execute(
            "INSERT INTO signals(ts,strategy,condition_id,token_id,side,p_market,p_est,p_std,edge,ev_per_share,"
            "confidence,action,reason) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (time.time(), o.strategy, o.condition_id, o.token_id, o.side, o.p_market, o.p_est, o.p_std,
             o.edge, o.edge, o.confidence, action, o.reject_reason or json.dumps(o.detail)))

    def _find_and_trade(self, markets, books, weights, smart_trades) -> None:
        c = self.cfg
        trend_budget = 40  # history calls per cycle, top-volume markets first
        for m in markets:
            yb, nb = books.get(m.yes_token), books.get(m.no_token)
            if not yb or not nb:
                continue
            # 1) genuine arbitrage (complete set)
            arb = evaluate_binary_arbitrage(m, yb, nb, sets_usd=c.max_order_usd, min_profit=c.arb_min_profit)
            if arb and arb.accepted:
                self._log_signal(arb, "accept")
                self._execute_arb(m, arb)
                continue
            # 2) speculative directional views
            ests = self._estimates(m, yb, weights, smart_trades, with_trend=trend_budget > 0)
            trend_budget -= 1
            for est in ests:
                opp = evaluate_directional(est, m, yb, nb, probe_usd=max(c.min_order_usd * 5, 25),
                                           min_edge=c.min_edge, min_confidence=c.min_confidence,
                                           max_spread=c.max_spread)
                if not opp:
                    continue
                if not opp.accepted:
                    self._log_signal(opp, "reject")
                    continue
                self._execute_directional(m, opp, books[opp.token_id])
                break

    def _execute_directional(self, m: Market, opp: Opportunity, book: OrderBook) -> None:
        st = self.portfolio.state({})
        usd, why = self.risk.size_order(st, m.condition_id, m.event_id, opp.p_est, opp.p_std, opp.cost_per_share)
        if usd <= 0:
            opp.reject_reason = why
            self._log_signal(opp, "reject")
            return
        est = walk_book(book.asks, "BUY", usd=usd, limit_price=opp.limit_price, fee_rate=m.fee_rate,
                        fee_exponent=m.fee_exponent)
        shares = round(est.shares, 2)
        if shares < m.min_size:
            opp.reject_reason = f"size {shares} < market min {m.min_size}"
            self._log_signal(opp, "reject")
            return
        self._log_signal(opp, "accept")
        self.orders.submit(OrderRequest(opp.token_id, "BUY", opp.limit_price, shares, "FAK", m.fee_rate,
                                        m.fee_exponent),
                           condition_id=m.condition_id, event_id=m.event_id, strategy=opp.strategy,
                           question=m.question, outcome=opp.outcome)

    def _execute_arb(self, m: Market, opp: Opportunity) -> None:
        st = self.portfolio.state({})
        budget = min(self.cfg.max_order_usd, st.cash - self.cfg.min_cash_reserve_pct * st.equity,
                     self.cfg.max_position_pct * st.equity - st.exposure_by_market.get(m.condition_id, 0.0))
        if budget <= 0:
            return
        sets = round(min(opp.legs[0]["shares"], budget / max(opp.cost_per_share, 1e-6)), 2)
        if sets < m.min_size:
            return
        filled = []
        for i, leg in enumerate(opp.legs):
            res = self.orders.submit(
                OrderRequest(leg["token_id"], "BUY", leg["limit"], sets, "FOK", m.fee_rate, m.fee_exponent),
                condition_id=m.condition_id, event_id=m.event_id, strategy="complete_set_arb",
                key=client_key(f"arb{i}", leg["token_id"], "BUY"), question=m.question,
                outcome=m.outcomes[i] if i < len(m.outcomes) else "")
            if not res or res.status != "FILLED":
                break
            filled.append(leg)
        if 0 < len(filled) < len(opp.legs):
            # leg risk materialised: unwind what we bought
            self.db.audit("arb_leg_risk_unwind", {"condition_id": m.condition_id}, level="WARNING")
            for leg in filled:
                self._sell_all(m, leg["token_id"], strategy="arb_unwind")

    def _sell_all(self, m: Market, token_id: str, strategy: str, min_price: float = 0.01) -> None:
        pos = self.db.one("SELECT size FROM positions WHERE token_id=?", (token_id,))
        if not pos or pos["size"] < 1e-6:
            return
        self.orders.submit(OrderRequest(token_id, "SELL", min_price, round(pos["size"], 2), "FAK", m.fee_rate,
                                        m.fee_exponent),
                           condition_id=m.condition_id, event_id=m.event_id, strategy=strategy,
                           question=m.question, outcome="exit")

    def _manage_exits(self, held_books: dict[str, OrderBook], weights, smart_trades) -> None:
        """Sell when the executable bid (net of fees) exceeds our current estimate of value.
        Positions with no current estimate are held to resolution."""
        for p in self.portfolio.positions():
            book = held_books.get(p["token_id"])
            mrow = self.db.one("SELECT * FROM markets WHERE condition_id=?", (p["condition_id"],))
            if not book or not book.bids or not mrow or mrow["closed"]:
                continue
            m = Market(condition_id=mrow["condition_id"], question=mrow["question"], slug=mrow["slug"],
                       event_id=mrow["event_id"], event_slug=mrow["event_slug"], neg_risk=bool(mrow["neg_risk"]),
                       end_date=mrow["end_date"], active=True, closed=False,
                       outcomes=json.loads(mrow["outcomes"] or "[]"), outcome_prices=[],
                       token_ids=[mrow["yes_token"], mrow["no_token"]], fee_rate=mrow["fee_rate"] or 0.0,
                       fee_exponent=mrow["fee_exponent"] or 1.0, tick_size=mrow["tick_size"] or 0.01,
                       min_size=mrow["min_size"] or 5, liquidity=mrow["liquidity"] or 0,
                       volume24h=mrow["volume24h"] or 0)
            is_yes = p["token_id"] == m.yes_token
            yes_book = book if is_yes else None
            if yes_book is None:
                try:
                    yes_book = self.api.order_book(m.yes_token)
                except Exception:  # noqa: BLE001
                    continue
            ests = self._estimates(m, yes_book, weights, smart_trades, with_trend=True)
            if not ests:
                continue
            p_yes = sum(e.p / e.std ** 2 for e in ests) / sum(1 / e.std ** 2 for e in ests)
            value = p_yes if is_yes else 1 - p_yes
            bid = book.best_bid
            net_bid = bid - taker_fee(1, bid, m.fee_rate, m.fee_exponent)
            if net_bid > value + 0.01:
                self.db.audit("exit_signal", {"token_id": p["token_id"], "value": value, "net_bid": net_bid})
                self._sell_all(m, p["token_id"], strategy="exit", min_price=round(max(value, 0.01), 2))
