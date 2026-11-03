"""Configuration loaded from environment variables (and an optional .env file).

Every risk-relevant number lives here so it can be audited in one place.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from pathlib import Path


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def _cast(raw: str, typ):
    if typ is bool:
        return raw.strip().lower() in ("1", "true", "yes", "on")
    if typ is int:
        return int(raw)
    if typ is float:
        return float(raw)
    if typ == list[str]:
        return [s.strip() for s in raw.split(",") if s.strip()]
    return raw


@dataclass
class Config:
    # --- mode -------------------------------------------------------------
    mode: str = "paper"  # "paper" | "live"
    live_trading_confirmed: bool = False  # second key required for live mode
    db_path: str = "data/polyagent.db"
    kill_file: str = "data/KILL"
    log_level: str = "INFO"

    # --- endpoints ----------------------------------------------------------
    gamma_url: str = "https://gamma-api.polymarket.com"
    data_url: str = "https://data-api.polymarket.com"
    clob_url: str = "https://clob.polymarket.com"
    chain_id: int = 137

    # --- live credentials (only used when mode=live) -------------------------
    private_key: str = ""
    funder_address: str = ""
    signature_type: int = 1  # 0=EOA, 1=email/magic proxy, 2=browser proxy

    # --- capital & risk -----------------------------------------------------
    starting_capital: float = 1000.0  # paper bankroll in USDC
    kelly_fraction: float = 0.25  # fraction of full Kelly
    max_position_pct: float = 0.05  # max cost basis per market as % of equity
    max_event_pct: float = 0.10  # max exposure per correlated event group
    max_total_exposure_pct: float = 0.60  # max cost basis deployed overall
    min_cash_reserve_pct: float = 0.20  # collateral always left unspent
    max_drawdown_pct: float = 0.20  # equity drawdown from peak -> halt + cancel
    daily_loss_limit_pct: float = 0.08  # loss since UTC midnight -> halt for day
    min_order_usd: float = 2.0
    max_order_usd: float = 250.0

    # --- signal thresholds --------------------------------------------------
    min_edge: float = 0.03  # p_est minus all-in cost per share
    min_confidence: float = 0.80  # P(true prob > breakeven) under posterior
    max_spread: float = 0.06
    min_liquidity_usd: float = 5000.0
    min_volume24h_usd: float = 1000.0
    min_hours_to_close: float = 6.0
    max_days_to_close: float = 120.0
    price_floor: float = 0.04  # avoid lottery tickets / near-certain contracts
    price_ceiling: float = 0.96
    min_r_squared: float = 0.5  # trend estimator gate
    arb_min_profit: float = 0.005  # per $1 of complete set, after fees

    # --- wallet intelligence ------------------------------------------------
    wallet_min_trades: int = 50
    wallet_min_resolved_markets: int = 15
    wallet_min_days_active: int = 30
    wallet_max_market_share: float = 0.5  # >50% of P&L from one market = flagged
    wallet_top_n: int = 50
    wallet_refresh_minutes: int = 360
    wallet_seed_count: int = 200  # leaderboard wallets to analyse per refresh
    smart_flow_lookback_hours: int = 24
    # smart-money estimator hyperparameters: set these from a walk-forward backtest
    smart_prior_concentration: float = 200.0
    smart_per_wallet_pseudo: float = 6.0
    smart_min_wallets: int = 2

    # --- liquidity-reward market making (paper) --------------------------------
    mm_capital_per_market: float = 250.0  # YES bid + NO bid notional per market
    mm_max_markets: int = 8
    mm_spread_frac: float = 0.5  # quote at this fraction of the reward band from mid
    mm_max_inventory_usd: float = 150.0  # stop bidding a side once it holds this much
    mm_vol_window_sec: int = 300
    mm_vol_pull: float = 0.03  # pull quotes if mid ranges more than this inside the window
    mm_min_hours_to_close: float = 48.0
    mm_mid_min: float = 0.10
    mm_mid_max: float = 0.90
    mm_cycle_sec: int = 15
    mm_reselect_minutes: int = 15
    mm_universe: int = 400  # markets scanned for reward pools

    # --- memecoin paper trader ----------------------------------------------------
    meme_cycle_sec: int = 60
    meme_discovery_pages: int = 2
    meme_max_new_per_cycle: int = 25
    meme_track_hours: float = 24.0
    meme_tx_cost_usd: float = 0.15  # Solana base + priority fee per swap (competitive fees can be higher)

    # --- loop cadence -------------------------------------------------------
    scan_interval_sec: int = 60
    markets_per_scan: int = 150
    order_stale_sec: int = 120
    http_timeout_sec: float = 15.0
    http_max_retries: int = 5

    # --- news ---------------------------------------------------------------
    news_feeds: list[str] = field(default_factory=list)
    news_uncertainty_boost: float = 1.5  # widen p_std when fresh news matches

    # --- dashboard ----------------------------------------------------------
    dashboard_host: str = "127.0.0.1"
    dashboard_port: int = 8080
    dashboard_token: str = ""  # required for POST /api/kill when set

    @classmethod
    def load(cls, env_file: str | None = ".env") -> "Config":
        if env_file:
            _load_dotenv(Path(env_file))
        cfg = cls()
        for f in fields(cls):
            raw = os.environ.get("POLY_" + f.name.upper())
            if raw is not None:
                typ = f.type if not isinstance(f.type, str) else eval(f.type)  # noqa: S307
                setattr(cfg, f.name, _cast(raw, typ))
        cfg.validate()
        return cfg

    def validate(self) -> None:
        if self.mode not in ("paper", "live"):
            raise ValueError("POLY_MODE must be 'paper' or 'live'")
        if self.mode == "live":
            if not self.live_trading_confirmed:
                raise ValueError("Live mode requires POLY_LIVE_TRADING_CONFIRMED=true")
            if not self.private_key or not self.funder_address:
                raise ValueError("Live mode requires POLY_PRIVATE_KEY and POLY_FUNDER_ADDRESS")
        if not 0 < self.kelly_fraction <= 1:
            raise ValueError("kelly_fraction must be in (0, 1]")
        if self.max_position_pct >= 1 or self.max_total_exposure_pct >= 1:
            raise ValueError("Exposure limits must be < 100% of equity")
        if self.min_cash_reserve_pct <= 0:
            raise ValueError("A positive cash reserve is mandatory")
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
