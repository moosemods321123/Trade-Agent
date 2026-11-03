"""Command-line entry point: python -m polyagent <command>."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path

from .config import Config


def _setup_logging(level: str) -> None:
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)


def cmd_run(cfg: Config, args) -> None:
    from .agent import Agent

    agent = Agent(cfg)
    if args.once:
        agent.cycle()
        print(json.dumps(agent.db.one("SELECT * FROM equity_snapshots ORDER BY ts DESC LIMIT 1"), indent=2))
    else:
        agent.run_forever()


def cmd_mm(cfg: Config, args) -> None:
    from .mm.maker import MarketMaker

    mm = MarketMaker(cfg)
    if args.once:
        mm.cycle()
        for r in mm.db.query("SELECT * FROM mm_quotes"):
            print(r)
    else:
        mm.run_forever()


def cmd_dashboard(cfg: Config, args) -> None:
    import uvicorn

    from .dashboard.app import create_app

    uvicorn.run(create_app(cfg), host=args.host or cfg.dashboard_host, port=args.port or cfg.dashboard_port,
                log_level="warning")


def cmd_wallets(cfg: Config, args) -> None:
    from .api.polymarket import PolymarketData
    from .db import Database
    from .wallets.intelligence import WalletRanker

    db = Database(cfg.db_path)
    cfg.wallet_seed_count = args.n
    ranker = WalletRanker(PolymarketData(cfg), db, cfg)
    res = ranker.refresh(args.wallet or None)
    res.sort(key=lambda m: (m.reliable, m.score), reverse=True)
    print(f"{'wallet':44} {'rel':>3} {'trades':>6} {'mkts':>5} {'win%':>6} {'winLCB':>6} {'ROI%':>7} "
          f"{'retLCB':>7} {'realized$':>11}  flags")
    for m in res[: args.top]:
        print(f"{m.wallet:44} {'Y' if m.reliable else '-':>3} {m.n_trades:>6} {m.n_resolved:>5} "
              f"{100 * m.win_rate:>6.1f} {100 * m.win_rate_lcb:>6.1f} {100 * m.roi:>7.1f} {m.return_lcb:>7.3f} "
              f"{m.realized_pnl:>11,.0f}  {','.join(m.flags)}")
    print(f"\n{sum(m.reliable for m in res)}/{len(res)} wallets passed reliability gates")


def cmd_backtest(cfg: Config, args) -> None:
    from .api.polymarket import PolymarketData
    from .backtest.engine import Backtester, BTParams, build_dataset, market_calibration

    cache = Path(args.cache)
    data = build_dataset(PolymarketData(cfg), args.markets, args.start, args.end, cache)
    print(f"dataset: {len(data)} resolved markets ({cache})")
    print("market calibration 24h before resolution:", json.dumps(market_calibration(data)))
    bt = Backtester(data, start_equity=args.equity)
    base = BTParams(strategy=args.strategy)
    grid = {"min_edge": [0.03, 0.06], "min_confidence": [0.6, 0.8]}  # 72 combos for smart_money
    if args.strategy == "smart_money":
        grid["prior_concentration"] = [20.0, 60.0, 200.0]
        grid["per_wallet_pseudo"] = [6.0, 20.0, 60.0]
        grid["min_smart_wallets"] = [1, 2]
    wf = bt.walk_forward(grid, base)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(wf, indent=2, default=str))
    print(json.dumps({k: wf.get(k) for k in ("train_markets", "test_markets", "chosen", "train", "test", "note")},
                     indent=2, default=str))
    print(f"full results: {out}")


def cmd_study(cfg: Config, args) -> None:
    """Signal study on a cached backtest dataset: smart-wallet flow vs all-wallet flow."""
    from .backtest.engine import Backtester, BTParams, MarketRecord, flow_signal_study

    data = [MarketRecord.from_json(d) for d in json.loads(Path(args.cache).read_text())]
    bt = Backtester(data)
    p = BTParams()
    print("smart wallets:", json.dumps(flow_signal_study(bt, p, smart_only=True)))
    print("all wallets:  ", json.dumps(flow_signal_study(bt, p, smart_only=False)))


def cmd_scan(cfg: Config, args) -> None:
    """One-shot read-only scan: shows eligible markets and any complete-set arbitrage."""
    from .api.polymarket import PolymarketData
    from .db import Database
    from .market.scanner import MarketScanner
    from .strategy.engine import evaluate_binary_arbitrage

    db = Database(cfg.db_path)
    api = PolymarketData(cfg)
    markets, books = MarketScanner(api, db, cfg).scan()
    print(f"{len(markets)} eligible markets")
    for m in markets[: args.top]:
        yb, nb = books.get(m.yes_token), books.get(m.no_token)
        if not yb or not nb:
            continue
        arb = evaluate_binary_arbitrage(m, yb, nb, 100, cfg.arb_min_profit)
        gap = f"arb {arb.edge:+.4f}/set" if arb else "arb n/a"
        print(f"{m.question[:70]:70} bid {yb.best_bid} ask {yb.best_ask} | {gap}")


def cmd_report(cfg: Config, args) -> None:
    from .db import Database
    from .reports import write_report

    p = write_report(Database(cfg.db_path))
    print(p.read_text())


def cmd_kill(cfg: Config, args) -> None:
    from .db import Database

    db = Database(cfg.db_path)
    db.set_state("kill", True)
    db.set_state("kill_handled", False)
    Path(cfg.kill_file).touch()
    db.audit("kill_requested", {"via": "cli"}, level="CRITICAL")
    print("kill switch set: the agent will cancel all orders and halt on its next cycle")


def cmd_resume(cfg: Config, args) -> None:
    from .db import Database

    db = Database(cfg.db_path)
    for k in ("kill", "kill_handled", "halt"):
        db.set_state(k, None)
    Path(cfg.kill_file).unlink(missing_ok=True)
    db.audit("resume_requested", {"via": "cli"}, level="WARNING")
    print("kill switch and halts cleared")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="polyagent")
    ap.add_argument("--env", default=".env")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--once", action="store_true")
    mmp = sub.add_parser("mm", help="paper liquidity-reward market maker")
    mmp.add_argument("--once", action="store_true")
    d = sub.add_parser("dashboard")
    d.add_argument("--host")
    d.add_argument("--port", type=int)
    w = sub.add_parser("wallets")
    w.add_argument("--n", type=int, default=60)
    w.add_argument("--top", type=int, default=25)
    w.add_argument("--wallet", action="append")
    b = sub.add_parser("backtest")
    b.add_argument("--strategy", choices=["smart_money", "trend"], default="smart_money")
    b.add_argument("--markets", type=int, default=120)
    b.add_argument("--start", default="2026-08-01")
    b.add_argument("--end", default="2026-10-31")
    b.add_argument("--equity", type=float, default=1000.0)
    b.add_argument("--cache", default="data/backtest_cache.json")
    b.add_argument("--out", default="data/backtest_result.json")
    st = sub.add_parser("study")
    st.add_argument("--cache", default="data/backtest_cache.json")
    s = sub.add_parser("scan")
    s.add_argument("--top", type=int, default=30)
    sub.add_parser("report")
    sub.add_parser("kill")
    sub.add_parser("resume")
    args = ap.parse_args(argv)
    cfg = Config.load(args.env)
    _setup_logging(cfg.log_level)
    {"run": cmd_run, "mm": cmd_mm, "dashboard": cmd_dashboard, "wallets": cmd_wallets, "backtest": cmd_backtest,
     "scan": cmd_scan, "study": cmd_study, "report": cmd_report, "kill": cmd_kill, "resume": cmd_resume}[args.cmd](cfg, args)


if __name__ == "__main__":
    main(sys.argv[1:])
