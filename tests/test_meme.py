import time

import pytest

from polyagent.meme.backtest import Coin, Obs, simulate
from polyagent.meme.sources import USDC, PairInfo, Quote, RugReport
from polyagent.meme.strategy import MemeParams, TokenState, entry_signal, exit_signal, rug_ok
from polyagent.meme.trader import MemeTrader


def good_state(**kw):
    base = dict(age_min=60, liquidity=50_000, fdv=500_000, vol_h1=80_000, buys_m5=30, sells_m5=10,
                chg_m5=2.0, chg_h1=40.0, rug_score=20.0, risks=[])
    base.update(kw)
    return TokenState(**base)


def test_rug_filter_blocks_hard_risks_and_missing_reports():
    p = MemeParams()
    assert rug_ok(good_state(), p)[0]
    assert not rug_ok(good_state(risks=["Mint Authority still enabled"]), p)[0]
    assert not rug_ok(good_state(risks=["Creator history of rugged tokens"]), p)[0]
    assert not rug_ok(good_state(rug_score=None), p)[0]
    assert not rug_ok(good_state(rug_score=90), p)[0]


def test_entry_gates():
    p = MemeParams()
    assert entry_signal(good_state(), p)[0]
    assert entry_signal(good_state(age_min=5), p)[1] == "age"
    assert entry_signal(good_state(liquidity=5_000), p)[1] == "liquidity"
    assert entry_signal(good_state(buys_m5=5, sells_m5=20), p)[1] == "buy pressure"
    assert entry_signal(good_state(chg_h1=900), p)[1] == "momentum"


def test_exit_ladder():
    p = MemeParams()
    assert exit_signal(2.1, 2.1, 1, False, p) == ("take_profit", p.take_profit_frac)
    assert exit_signal(0.5, 1.0, 1, False, p)[0] == "stop_loss"
    assert exit_signal(1.9, 3.0, 1, True, p)[0] == "trailing_stop"  # 1.9 <= 3.0 * 0.65
    assert exit_signal(1.2, 1.3, 13, False, p)[0] == "time_exit"
    assert exit_signal(1.2, 1.3, 1, False, p)[0] == ""


def _coin(mint, prices, liq=50_000, start=0.0, created=-3600.0):
    obs = [Obs(start + i * 60, p, (liq if p > 0 else 0), 500_000, 80_000, 30, 10, 2.0, 40.0)
           for i, p in enumerate(prices)]
    return Coin(mint, mint, start, start + created, 20.0, [], obs)


def test_backtest_rug_goes_to_zero_and_winner_takes_profit():
    p = MemeParams(min_age_min=0)
    rug = _coin("rug", [1.0, 1.0, 0.0, 0.0])
    moon = _coin("moon", [1.0, 1.5, 2.5, 3.0, 1.5], start=1.0)
    r = simulate([rug, moon], p, 1000.0, tx_cost=0.0)
    reasons = r["exit_reasons"]
    assert reasons.get("rugged") == 1 or reasons.get("stop_loss") == 1
    assert reasons.get("take_profit") == 1
    rug_trade = next(t for t in r["trades_detail"] if t["mint"] == "rug")
    assert rug_trade["proceeds"] == 0.0


def test_backtest_does_not_peek_at_future_prices():
    # coin only qualifies (momentum) after a big move; we must enter at the later price, not before
    p = MemeParams(min_age_min=0)
    c = _coin("c", [1.0, 1.0, 5.0, 5.0])
    for o in c.obs[:2]:
        o.chg_h1 = 0.0  # fails momentum early
    r = simulate([c], p, 1000.0, tx_cost=0.0)
    assert r["trades"] == 0 or all(t["proceeds"] <= t["cost"] * 1.05 for t in r["trades_detail"])


class FakeSrc:
    def __init__(self):
        self.pools = [{"mint": "M1", "pool": "P1", "created": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(time.time() - 3600)),
                       "dex": "pumpswap"}]
        self.rate = 1000.0  # tokens (raw) per USDC micro-unit... simple linear book
        self.route = True

    def new_pools(self, pages=2):
        return self.pools

    def rugcheck(self, mint):
        return RugReport(20.0, 100.0, [])

    def pairs(self, mints):
        return {m: PairInfo(m, "MEME", "Meme", "P1", "pumpswap", 0.001, 50_000, 500_000, 5_000, 80_000, 30, 10,
                            300, 100, 2.0, 40.0, int((time.time() - 3600) * 1000)) for m in mints}

    def quote(self, inp, out, amount, slippage_bps=500):
        if not self.route:
            return None
        if inp == USDC:
            return Quote(amount, int(amount * self.rate), 0.0)
        return Quote(amount, int(amount / self.rate), 0.0)


def test_trader_buys_values_and_takes_profit(cfg, db):
    src = FakeSrc()
    t = MemeTrader(cfg, src=src, db=db)
    t.cycle()
    pos = db.one("SELECT * FROM meme_positions")
    assert pos and pos["cost_usd"] == pytest.approx(10 + cfg.meme_tx_cost_usd)
    src.rate = 400.0  # token value up 2.5x
    t.cycle()
    pos = db.one("SELECT * FROM meme_positions")
    assert pos["took_profit"] == 1
    sells = db.query("SELECT * FROM fills WHERE side='SELL'")
    assert sells and sells[0]["outcome"] == "take_profit" and sells[0]["realized_pnl"] > 0


def test_trader_writes_off_when_liquidity_vanishes(cfg, db, monkeypatch):
    src = FakeSrc()
    t = MemeTrader(cfg, src=src, db=db)
    t.cycle()
    src.route = False
    later = time.time() + 7200
    for k in range(4):
        t.cycle(now=later + k * 60)
    assert not db.query("SELECT * FROM meme_positions")
    w = db.one("SELECT * FROM fills WHERE outcome='written_off'")
    assert w and w["realized_pnl"] < 0


def test_kill_switch_flattens(cfg, db):
    src = FakeSrc()
    t = MemeTrader(cfg, src=src, db=db)
    t.cycle()
    db.set_state("kill", True)
    t.cycle()
    assert not db.query("SELECT * FROM meme_positions")
    assert db.get_state("halt")


def test_meme_trader_refuses_live(cfg):
    cfg.mode = "live"
    with pytest.raises(RuntimeError):
        MemeTrader(cfg, src=FakeSrc())
