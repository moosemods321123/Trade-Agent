# PolyAgent

An autonomous Polymarket intelligence and trading agent: wallet analytics, market monitoring,
cost-aware signal evaluation, uncertainty-adjusted Kelly sizing, paper and live execution through
the official CLOB client, hard risk limits, and a small live dashboard.

> **Read this first.** Nothing here guarantees profit, and no software can. The agent is built to
> trade only when it measures an edge that survives fees, spread and estimation error. Most of the
> time it measures none and does nothing, which is the correct result. It ships in **paper mode**.
> Live trading takes two explicit switches and your own keys, and it is your responsibility to
> check that Polymarket's terms allow you to trade from your jurisdiction. This code does not
> bypass geoblocking and must not be used to.

---

## Contents

1. [Architecture](#architecture)
2. [Install](#install)
3. [Configuration](#configuration)
4. [Running](#running)
5. [Strategies and the maths](#strategies-and-the-maths)
6. [Backtesting](#backtesting)
7. [Measured results](#measured-results)
8. [Risk controls](#risk-controls)
9. [Database schema](#database-schema)
10. [Deployment (Linux VPS)](#deployment-linux-vps)
11. [Going live](#going-live)
12. [Assumptions](#assumptions)
13. [Limitations](#limitations)

---

## Architecture

```
polyagent/
  config.py              every tunable, loaded from POLY_* env vars / .env
  db.py                  SQLite (WAL) schema + audit log + key/value state
  api/http.py            retrying HTTP client (exponential backoff, jitter, rate limit)
  api/polymarket.py      Gamma (markets), Data API v2 (wallets/trades), CLOB public (books, history)
  analytics/stats.py     OLS slope / R^2 / slope SE, Beta-Binomial updating, Wilson bounds, drawdown
  analytics/costs.py     documented taker-fee formula, order-book walking (VWAP, slippage, depth)
  wallets/intelligence.py  FIFO P&L reconstruction, risk metrics, suspicious-pattern flags, ranking
  market/scanner.py      universe filter, batch order books, price history, RSS news matcher
  strategy/estimators.py smart-money Bayesian estimator, regression-trend estimator
  strategy/engine.py     executable-price EV, confidence gates, complete-set arbitrage
  risk/manager.py        fractional Kelly with uncertainty shrinkage, exposure caps, halts
  execution/brokers.py   PaperBroker (fills against live books) and LiveBroker (py-clob-client)
  execution/orders.py    idempotent submission, fill verification, stale cancels, reconciliation
  execution/portfolio.py cash/position ledger, settlement, equity snapshots
  backtest/engine.py     point-in-time, event-driven backtester with walk-forward selection
  agent.py               the autonomous loop
  dashboard/             FastAPI + one-page UI (P&L $ and %, equity curve, positions, trades, kill switch)
  reports.py             performance reports from recorded data only
  cli.py                 python -m polyagent {run,dashboard,backtest,wallets,scan,report,kill,resume}
deploy/                  systemd units, watchdog, daily report timer, installer
tests/                   41 automated tests
```

Each cycle of the loop (default every 60 s):

1. Heartbeat. Check the kill switch (file or DB flag). If it is set, cancel every order, halt, and only mark to market.
2. Scan active markets, keep the eligible binary ones, batch-fetch their order books.
3. Poll open orders for fills and cancel stale ones. Settle resolved positions (paper) or reconcile with the exchange (live).
4. Mark to market, snapshot equity, check drawdown and daily-loss limits. A breach cancels everything and halts.
5. Refresh wallet rankings in a background thread when they are stale. Poll news feeds.
6. Exits: sell a position when the executable bid, net of fees, is worth more than the current estimate.
7. Entries: complete-set arbitrage first, then directional estimates. Every candidate is logged to `signals` with an accept or reject reason.

---

## Install

Python 3.11+.

```bash
git clone <this repo> polyagent && cd polyagent
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # defaults are paper mode
python -m pytest              # 41 tests, no network needed
```

## Configuration

All settings are environment variables prefixed `POLY_`. They are listed with defaults in
`.env.example` and documented in `polyagent/config.py`. The ones that matter most:

| Variable | Default | Meaning |
|---|---|---|
| `POLY_MODE` | `paper` | `paper` or `live` |
| `POLY_LIVE_TRADING_CONFIRMED` | `false` | second switch required for live |
| `POLY_STARTING_CAPITAL` | `1000` | paper bankroll (live reads the real balance) |
| `POLY_KELLY_FRACTION` | `0.25` | fraction of full Kelly |
| `POLY_MAX_POSITION_PCT` | `0.05` | max cost basis per market, as a share of equity |
| `POLY_MAX_EVENT_PCT` | `0.10` | max per correlated event group |
| `POLY_MAX_TOTAL_EXPOSURE_PCT` | `0.60` | max deployed |
| `POLY_MIN_CASH_RESERVE_PCT` | `0.20` | collateral that is never spent |
| `POLY_MAX_DRAWDOWN_PCT` | `0.20` | from peak: cancel all and halt (manual resume) |
| `POLY_DAILY_LOSS_LIMIT_PCT` | `0.08` | since UTC midnight: cancel all and halt until tomorrow |
| `POLY_MIN_EDGE` | `0.03` | required p_est minus all-in cost per share |
| `POLY_MIN_CONFIDENCE` | `0.80` | required P(true prob > cost) |

## Running

```bash
python -m polyagent scan              # read-only: eligible markets + arbitrage gaps
python -m polyagent wallets --n 60    # score wallets, print ranking with flags
python -m polyagent run --once        # one full cycle
python -m polyagent run               # run forever
python -m polyagent dashboard         # http://127.0.0.1:8080
python -m polyagent backtest --strategy smart_money
python -m polyagent report            # writes data/reports/report-*.md
python -m polyagent kill              # emergency stop: cancel all + halt
python -m polyagent resume            # clear kill/halts
```

The **emergency stop** works three ways, and each one cancels all open orders on the next cycle and halts:
the dashboard button, `python -m polyagent kill`, or `touch data/KILL`.

---

## Strategies and the maths

### Probability estimates

**Smart-money estimator** (`strategy/estimators.py`). The prior is a Beta distribution centred on the
market mid, with concentration *K*. The market is treated as informative and is never ignored. Each
wallet that passed the reliability gates adds pseudo-observations toward YES or NO, in proportion to
its weight and its net 24-hour flow, capped per wallet so no single whale dominates. The estimate is
the posterior mean, and its uncertainty is the posterior standard deviation. At least two
independent reliable wallets must agree before it gives a view.

**Trend estimator.** OLS of logit(price) on time over 72 h. It gives a view only if R² ≥ 0.5 and
|t(slope)| ≥ 3. The projection is damped, and the uncertainty is propagated from the slope standard
error, with a model-risk floor.

**News** (optional RSS). A fresh headline that matches a market widens the uncertainty of the
estimate (×1.5). Headlines are never turned into a direction. Doing that without a validated model
would invent an edge.

### From estimate to trade

For each side the engine walks the *actual ask book* for the probe size, including the market's
taker fee at every price level. That gives an all-in cost per share *c*. It then requires:

- `edge = p − c ≥ min_edge`
- `confidence = P(p_true > c) ≥ min_confidence`, under N(p, σ)
- spread ≤ `max_spread`, and enough depth

The order is a fill-and-kill limit at the highest price that still leaves `min_edge` after fees.

**Fees** follow Polymarket's documented formula `fee = shares × rate × p × (1 − p)`, using each
market's own `feeSchedule.rate`. Makers pay nothing.

### Arbitrage vs speculation

- **Complete-set arbitrage**: YES ask + NO ask + fees < $1. One leg pays $1 at resolution. It is executed with FOK legs, and if the second leg fails the first is unwound. The remaining risks are leg risk, oracle disputes and capital locked until resolution.
- **Multi-outcome (neg-risk) gaps** are detected only (`multi_outcome_arbitrage_gap`). Gamma event listings are not guaranteed to be exhaustive, so this is never auto-executed.
- **Everything else is speculative** and is labelled that way in `signals.strategy`.

### Position sizing

Full Kelly for a $1-payout contract bought at all-in cost *c* is `f* = (p − c)/(1 − c)`. The agent stakes

```
f = kelly_fraction × f* × edge² / (edge² + σ²)
```

The last factor shrinks the stake when estimation error is comparable to the edge. The stake is then
capped by the per-market, per-event and total-exposure limits, the cash reserve and `max_order_usd`.
It can never be the whole account: the reserve and the caps are validated at startup.

### Wallet intelligence

For each wallet (`wallets/intelligence.py`):

- **P&L.** Realized P&L comes from Polymarket's own per-position figures (`/v2/positions?status=CLOSED`, sorted by recency, see below). Unrealized P&L is marked from open positions. FIFO reconstruction from fills is used in the backtester, where the per-position API is not point-in-time.
- **Metrics.** ROI, win rate with its Wilson 95% lower bound, max drawdown of the realized P&L curve, trades per day, size-weighted holding period, trade size and its dispersion, average entry price, entry edge versus outcome, entry lead time before resolution, Herfindahl concentration, per-market Sharpe.
- **Score.** The one-sided 95% lower bound of the per-market mean return, shrunk toward zero (prior n₀ = 10), multiplied by log(1 + n). A short lucky streak cannot rank high.
- **Hard reliability gates.** Too few trades, too few closed markets or too short a history (≥ 50 / 15 / 30 days by default), concentrated returns (one market > 50% of gains), implausible win rate (> 90% over 10+ markets), bot or market-maker pace (> 200 trades/day), lottery-driven (one long shot bought under 10¢ makes most of the profit), or a non-positive lower bound.
- **Survivorship.** Leaderboard seeds only contain winners, so candidates also come from the live trade tape, and the gates and shrinkage do the filtering.

> **Data trap found and fixed during build:** `/v2/positions?status=CLOSED` defaults to sorting by
> realized P&L *descending*. Any capped fetch therefore sees only a wallet's best trades, which
> produced 100% win rates on every whale. The agent requests `sort_by=TIMESTAMP` instead.

---

## Backtesting

`backtest/engine.py` replays resolved markets in event order, with these look-ahead guards:

- strategies receive only price points and trades with `ts ≤ t`
- wallet rankings at time *t* are computed from trades `≤ t`, scored only on markets whose resolution timestamp is `≤ t` (both are covered by tests)
- parameters are chosen on the earliest 60% of markets, ordered by resolution time, and reported on the untouched last 40%

The run reports trade count, win rate, return, max drawdown, a bootstrap 95% CI on mean per-trade
return, and the Brier score of our estimate compared with the market's at entry. It also reports the
market's own calibration 24 h before resolution, which sets a baseline for how much room any
independent estimate has.

```bash
python -m polyagent backtest --strategy smart_money --markets 120 --start 2026-08-01 --end 2026-10-31
```

Data is cached in `data/backtest_cache.json`, so reruns are reproducible offline.

---

## Measured results

All numbers below come from runs made on 2026-11-03 against real Polymarket data. None are
simulated or adjusted, and the JSON outputs reproduce them (`data/` is git-ignored, so rerun the
commands). **Short version: no strategy here has shown an edge after costs out of sample.**

**Dataset.** 400 resolved binary markets that closed 2026-05-01 → 2026-10-31, chosen by volume.
Each has hourly YES prices for up to 60 days before resolution, plus every public trade ≥ $200
(Data API v2). Train is the first 240 markets by resolution time, test the last 160.

**The market is hard to beat.** Brier score of the market price 24 h before resolution: **0.114**
(a coin flip scores 0.25). There is little calibration error left for an independent estimate to exploit.

| Test | Result | Verdict |
|---|---|---|
| Complete-set arbitrage, live scan of 90 eligible markets | every YES+NO pair cost **$1.005–$1.120** per set after fees and depth | none available |
| Signal study: does net flow of point-in-time "reliable" wallets predict outcome − price? (`polyagent study`) | +0.025 per decision, **t = 0.56** over 49 events | no detectable signal |
| Same study using *all* wallets | +0.001, t = 0.05 over 181 events | no signal |
| Smart-money walk-forward (72 parameter sets chosen on train) | train **+19.7%** (18 trades) → test **−14.4%** (15 trades, 13% win rate, 95% CI of mean trade return [−100%, −34%]) | **overfit**. The best in-sample settings lost money out of sample, and out of sample its probabilities were worse than the market's (Brier 0.165 vs 0.123) |
| Trend walk-forward | train +10.7% (7 trades, CI spans negative) → test +2.9% (**2 trades**) | too few trades to mean anything |
| Live paper run, 3 cycles | 34 trend views passed the R² gate, all rejected (edge after fees < 3¢). 0 smart-money views. 0 trades | the gates work as designed |

**A cautionary detour.** An early ad-hoc version of the signal study showed t ≈ 3.8. That was a
look-ahead leak in the diagnostic script: it reused a wallet ranking computed at a *later* time
whenever it moved back to an earlier market. With a strictly point-in-time ranking (the engine's
`weights_at`), the effect disappears. If you extend this code, keep the point-in-time guarantees.
Leaks are the usual source of backtests that look too good.

What this means in practice: with the shipped defaults (strict wallet gates, prior concentration
200, two-wallet agreement), the smart-money estimator almost never fires, and when it does it moves
the price by about a cent. That is too little to trade on. The agent sits in cash, which, given the
evidence above, is the right answer. Loosening the gates to "make it trade" is exactly the overfit
configuration that lost 14% out of sample.

---

## Liquidity-reward market making (paper)

The directional strategies showed no edge, so the default service is a different business:
**getting paid to provide liquidity**. Polymarket pays makers a daily reward pool per market
(about $57k/day across ~257 markets when checked). The payout depends on quoting tightly on
both sides, not on predicting outcomes.

```bash
python -m polyagent mm --once   # one cycle, prints the quote book
python -m polyagent mm          # run forever (paper only, by design)
```

How it works (`polyagent/mm/`):

- **Selection:** scans reward markets and scores hypothetical quotes against the live book with the published formula, `S = ((v − s)/v)² × size`, two-sided `Qmin` with c = 3. It keeps the markets with the best estimated reward for the capital, and allocates collateral in that order.
- **Quotes:** a YES bid at `mid − s` and a NO bid at `(1 − mid) − s` inside the reward band. A filled pair is a complete set bought for less than $1, and it is **merged** straight back into $1.
- **Fills (paper):** resting bids fill only from the real public trade tape. Trades through our price fill us. Trades at our price fill us only after the displayed queue ahead of us is consumed. Taker buys of the complement fill us via minting. Adverse fills are therefore always counted.
- **Guards:** quotes are pulled when the mid moves > 3¢ within 5 min, within 48 h of close, outside 0.10–0.90, or when the book is too wide. Once one side's inventory exceeds $150, only the side that completes sets is quoted. Drawdown, daily-loss and kill-switch handling are shared with the main agent.
- **Rewards:** accrued every cycle from our estimated share and credited at the UTC day roll, with Polymarket's $1 minimum. They appear on the dashboard as *Rewards today (accruing)* and *Rewards paid*.

**P&L = rewards + spread on merged sets − losses on unhedged inventory when prices move.**
The first two terms are measurable now. The third only shows up over time and is the one that
kills market makers.

**First paper run (2026-11-03, 50 minutes, $1,000 bankroll, ~$800 quoted across 5 markets):**
$2.38 of rewards accrued (a ~$69/day run rate, still an estimate until a daily credit).
1 fill: 81 NO shares at 60¢, marked −$0.41. Net about +$2 including accrued rewards.
Fifty minutes says nothing about adverse selection: there was no big news while it ran. The
number that matters is the P&L after a few weeks that include news shocks. Note also that
80% of the accrual came from two thin $50/day pools where the bot was the main quoter, so
the estimate will drop as soon as anyone competes.

Approximations to keep in mind:

- Competitors are scored as one aggregate maker from the visible book.
- Queue position ignores other makers' cancellations.
- Polymarket's per-minute sampling is approximated per cycle.
- Live quoting (post-only GTC orders, on-chain merge and redeem) is deliberately not wired up. The class refuses `POLY_MODE=live`. Run it in paper mode for weeks first.

## Memecoin paper trader (Solana)

```bash
python -m polyagent meme --once      # one cycle
python -m polyagent meme             # collect + paper trade forever (use its own POLY_DB_PATH)
python -m polyagent meme-backtest    # walk-forward on whatever has been collected
```

**Why it collects its own data.** Free memecoin APIs list coins that still exist. Coins that
rugged and vanished are mostly missing, so any "historical" backtest built from them is
survivorship-biased and flattering. The collector records **every** new Solana pool it sees on
GeckoTerminal, with its RugCheck report, and tracks it on DexScreener for 24 h. Young coins are
polled every cycle, older ones less often, and dead ones are tracked until they are marked dead.
That dataset is what `meme-backtest` uses.

**Strategy** (`polyagent/meme/strategy.py`, shared by the paper trader and the backtester):

- **Rug filter:** a RugCheck report is required, the normalised risk score must be ≤ 60, and any hard risk disqualifies the coin:
  - mint or freeze authority still enabled
  - a single holder or the top 10 holding most of the supply
  - LP unlocked
  - copycat token
  - a creator who has rugged before
- **Entry:** the coin is 20–240 minutes old (skipping the first-minutes snipe and dump), with liquidity ≥ $20k, FDV between $50k and $5M, 1h volume ≥ $20k, buys outnumbering sells over 5 minutes, and a +10% to +300% move over 1h. Each coin is traded at most once.
- **Execution (paper):** a real Jupiter USDC→token quote, rejected if slippage against a $1 probe quote exceeds 3%, *and* rejected unless a round-trip sell quote returns ≥ 85% of the stake (an anti-honeypot check). A $0.15 per-swap network and priority fee is charged.
- **Valuation:** every cycle, positions are valued with a Jupiter sell quote for the full amount, which is what you could actually get. No sell route for three cycles means the value goes to zero and the position is written off.
- **Exits:** sell half at 2×, then trail the rest with a 35% drawdown from peak. Stop-loss at 0.6×, time exit at 12 h.
- **Sizing:** 1% of equity per coin, at most 8 open, with the 20% cash reserve, drawdown limits and kill switch shared with the main agent. On a risk halt, everything is sold.

The paper trader refuses `POLY_MODE=live`. Signing real Jupiter swaps is deliberately not implemented.

**First run (2026-11-03, ~70 minutes, $1,000 paper bankroll):**

- 734 new Solana pools discovered. Only 46 ever reached $5k of liquidity; the rest were dust.
- Of those 46, an hour later: median price **0.01×** first seen, 72% down more than half, 48% with liquidity gone entirely, and **4% up more than 2×**.
- Most coins failed the rug filter or the liquidity and momentum gates. One trade passed everything: $10 into "Pog", stopped out at 0.54× (−$4.76). The price gapped through the 0.6× stop.
- Walk-forward backtest: **not enough data yet** (2 trades on the training half). It needs days to weeks of collection before it can say anything.

The base rates are the point. A random new coin is a lottery ticket with a poor expected value. The
question this bot answers over time is whether the filter can find the 4% often enough, and early
enough, to beat that. Nothing so far says it can.

---

## Risk controls

| Control | Where | Action |
|---|---|---|
| Max drawdown from peak | `RiskManager.check_limits` | cancel all, halt until manual resume |
| Daily loss limit | same | cancel all, halt until next UTC day |
| Kill switch (UI / CLI / file) | `Agent.cycle` | cancel all, halt |
| Per-market / per-event / total exposure caps | `RiskManager.size_order` | stake reduced or skipped |
| Cash reserve | same + config validation | never spent |
| Duplicate orders | unique `client_key` reserved *before* the exchange call | second submit refused |
| One open order per token | `OrderManager.has_open_order` | skip |
| Stale orders | `OrderManager.monitor` | cancelled after `order_stale_sec` |
| Crash between reserve and place | orphaned `PENDING` rows | cancelled by monitor, never retried blindly |
| Fill verification | `LiveBroker.get_order` → `get_trades` | ledger only moves on reported trades; `FAILED` trades ignored |
| Exchange drift | `OrderManager.reconcile_live` | adopt exchange positions/balance, audit the diff |

Every submit, update, cancel, halt, settlement, reconciliation diff and error is written to `audit_log`.

## Database schema

SQLite in WAL mode (`data/polyagent.db`). The full DDL is in `polyagent/db.py`.

| Table | Purpose |
|---|---|
| `markets` | market metadata, tokens, fee schedule, liquidity |
| `book_snapshots` | best bid/ask and top-5 depth per scan |
| `price_history` | cached hourly prices |
| `wallet_trades` | raw wallet fills used for analytics |
| `wallet_scores` | metrics, score, reliability and flags per wallet |
| `signals` | every evaluated opportunity with accept/reject reason |
| `orders` | order lifecycle, unique `client_key`, fills/fees summary |
| `fills` | individual fills and settlements with realized P&L |
| `positions` | size, average cost (fees capitalised), realized P&L |
| `equity_snapshots` | cash, positions value, equity, P&L $ and %, drawdown |
| `audit_log` | append-only event log |
| `news_items` | polled headlines and matched markets |
| `system_state` | kill switch, halt, heartbeat, ledger cash, high-water mark |

---

## Deployment (Linux VPS)

```bash
sudo ./deploy/install.sh            # creates user, venv, systemd units, starts everything
sudo nano /opt/polyagent/.env       # configure; keep POLY_MODE=paper initially
sudo systemctl restart polyagent
journalctl -u polyagent -f          # logs
ssh -L 8080:127.0.0.1:8080 you@vps  # then open http://localhost:8080
```

- `polyagent.service` restarts automatically (`Restart=always`, with crash-loop limits).
- `polyagent-watchdog.timer` restarts the agent if its heartbeat goes stale (via `/healthz`).
- `polyagent-report.timer` writes a daily report to `data/reports/`.
- State lives in SQLite, so a restart resumes where it stopped. Open orders are re-polled, and orphaned pending rows are resolved.
- The dashboard binds to localhost. If you expose it, set `POLY_DASHBOARD_TOKEN` and put it behind TLS.

## Going live

1. Run paper mode for weeks and read the `signals` table and the reports. If paper does not make money, live will not either.
2. Confirm that you are legally allowed to trade on Polymarket where you are.
3. Fund a dedicated wallet with an amount you can lose entirely.
4. Set `POLY_MODE=live`, `POLY_LIVE_TRADING_CONFIRMED=true`, `POLY_PRIVATE_KEY`, `POLY_FUNDER_ADDRESS` and `POLY_SIGNATURE_TYPE`. Start with `POLY_MAX_ORDER_USD=5`.
5. Watch the first fills against polymarket.com and check that reconciliation reports no diffs.

`LiveBroker` has been tested only against a fake client. No real order has been placed by this
code. Polymarket is migrating to Protocol V2 / pUSD collateral, and `py-clob-client` may need an
upgrade for V2 markets.

## Assumptions

- Taker fee is `shares × rate × (p(1−p))^exponent`. This equals the documented `shares × rate × p × (1−p)` for exponent 1, the only value seen. Makers pay nothing.
- The agent trades as taker (FAK/FOK), so maker rebates and queue position are ignored.
- Paper fills walk the live book at the moment of submission. The agent's own orders have no market impact beyond consuming the displayed depth.
- A position is marked at mid for P&L display. Liquidation value would be at the bid, which is lower.
- A closed position priced at exactly 0 or 1 is treated as resolved.
- Markets in the same Gamma event are treated as one correlated exposure group. Cross-event correlation, such as several elections that share a driver, is not modelled.

## Limitations

- **There is no evidence yet of profit after costs.** See [Measured results](#measured-results). The sample is small, comes from one period, and is concentrated in a few event types.
- No historical order books are public, so backtest execution uses the last observed price plus an assumed half-spread and slippage. Real fills will be worse in thin markets.
- The Data API serves market trades newest-first with no time filter. The backtest uses only trades ≥ $200 so that the cap covers each market's life. Smaller flow is invisible to it.
- Wallet histories through the API are capped (2,000 fills, 1,000 closed positions, most recent first), so these are recent-performance metrics, not lifetime ones.
- Ranking thousands of wallets with a 95% bound still lets a few through by chance (multiple testing). The live sample of reliable wallets is small, and each one is noise-prone.
- The trend estimator rarely fires. When it does, its edge after fees has so far been below threshold.
- Exits are value-based only. Without a fresh estimate, a position is held to resolution.
- Copying wallets can be front-run, or exploited by wallets that know they are being copied.
- Live redemption of resolved positions on-chain is not automated. The agent records it, and reconciliation picks up the balance once you redeem.
