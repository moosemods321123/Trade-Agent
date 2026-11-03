"""Data sources: GeckoTerminal (new pools), DexScreener (pairs), RugCheck (risk), Jupiter (quotes)."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from ..api.http import ApiError, HttpClient

log = logging.getLogger(__name__)

USDC = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
SOL = "So11111111111111111111111111111111111111112"
STABLES = {USDC, SOL, "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB"}


def _f(x, default=0.0) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


@dataclass
class PairInfo:
    mint: str
    symbol: str
    name: str
    pair: str
    dex: str
    price_usd: float
    liquidity_usd: float
    fdv: float
    vol_m5: float
    vol_h1: float
    buys_m5: int
    sells_m5: int
    buys_h1: int
    sells_h1: int
    chg_m5: float
    chg_h1: float
    created_ms: int


@dataclass
class RugReport:
    score_normalised: float
    lp_locked_pct: float
    risks: list[dict] = field(default_factory=list)

    @property
    def danger(self) -> list[str]:
        return [r.get("name", "") for r in self.risks if r.get("level") == "danger"]


@dataclass
class Quote:
    in_amount: int
    out_amount: int
    price_impact: float  # fraction, e.g. 0.01 = 1%


class MemeSources:
    def __init__(self, timeout: float = 15.0):
        self.gecko = HttpClient("https://api.geckoterminal.com/api/v2", timeout, 3, min_interval=2.2)  # ~30/min
        self.dex = HttpClient("https://api.dexscreener.com", timeout, 3, min_interval=0.25)
        self.rug = HttpClient("https://api.rugcheck.xyz/v1", timeout, 2, min_interval=1.0)
        self.jup = HttpClient("https://lite-api.jup.ag/swap/v1", timeout, 2, min_interval=1.1)

    def new_pools(self, pages: int = 2) -> list[dict]:
        out = []
        for page in range(1, pages + 1):
            try:
                d = self.gecko.get("/networks/solana/new_pools", {"page": page, "include": "base_token"})
            except ApiError as e:
                log.warning("gecko new_pools failed: %s", e)
                break
            for p in d.get("data", []):
                rel = p.get("relationships", {})
                base = rel.get("base_token", {}).get("data", {}).get("id", "")
                mint = base.split("_", 1)[1] if "_" in base else ""
                if not mint or mint in STABLES:
                    continue
                a = p.get("attributes", {})
                out.append({"mint": mint, "pool": a.get("address"), "created": a.get("pool_created_at"),
                            "dex": rel.get("dex", {}).get("data", {}).get("id", "")})
        return out

    def pairs(self, mints: list[str]) -> dict[str, PairInfo]:
        """Best (highest-liquidity) pair per mint, batched 30 at a time."""
        out: dict[str, PairInfo] = {}
        for i in range(0, len(mints), 30):
            chunk = mints[i:i + 30]
            try:
                rows = self.dex.get("/tokens/v1/solana/" + ",".join(chunk)) or []
            except ApiError as e:
                log.warning("dexscreener failed: %s", e)
                continue
            for r in rows:
                mint = r.get("baseToken", {}).get("address")
                if mint not in chunk:
                    continue
                liq = _f((r.get("liquidity") or {}).get("usd"))
                if mint in out and out[mint].liquidity_usd >= liq:
                    continue
                tx = r.get("txns") or {}
                vol = r.get("volume") or {}
                chg = r.get("priceChange") or {}
                out[mint] = PairInfo(
                    mint=mint, symbol=r.get("baseToken", {}).get("symbol", ""), name=r.get("baseToken", {}).get("name", ""),
                    pair=r.get("pairAddress", ""), dex=r.get("dexId", ""), price_usd=_f(r.get("priceUsd")),
                    liquidity_usd=liq, fdv=_f(r.get("fdv")), vol_m5=_f(vol.get("m5")), vol_h1=_f(vol.get("h1")),
                    buys_m5=int((tx.get("m5") or {}).get("buys", 0)), sells_m5=int((tx.get("m5") or {}).get("sells", 0)),
                    buys_h1=int((tx.get("h1") or {}).get("buys", 0)), sells_h1=int((tx.get("h1") or {}).get("sells", 0)),
                    chg_m5=_f(chg.get("m5")), chg_h1=_f(chg.get("h1")), created_ms=int(r.get("pairCreatedAt") or 0))
        return out

    def rugcheck(self, mint: str) -> RugReport | None:
        try:
            d = self.rug.get(f"/tokens/{mint}/report/summary")
        except ApiError as e:
            log.debug("rugcheck %s failed: %s", mint, e)
            return None
        return RugReport(_f(d.get("score_normalised"), 100.0), _f(d.get("lpLockedPct")), d.get("risks") or [])

    def quote(self, input_mint: str, output_mint: str, amount: int, slippage_bps: int = 500) -> Quote | None:
        """Executable route via Jupiter. None means no route (untradeable / rugged)."""
        if amount <= 0:
            return None
        try:
            d = self.jup.get("/quote", {"inputMint": input_mint, "outputMint": output_mint, "amount": int(amount),
                                        "slippageBps": slippage_bps})
        except ApiError as e:
            log.debug("jupiter quote failed: %s", e)
            return None
        if not d or "outAmount" not in d:
            return None
        return Quote(int(d["inAmount"]), int(d["outAmount"]), _f(d.get("priceImpactPct")) / 100.0)
