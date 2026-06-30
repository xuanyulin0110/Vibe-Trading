"""Tests for TWEquityEngine market rules.

Validates:
  - Day trading allowed: shares bought today CAN be sold today (unlike
    ChinaAEngine's T+1 lock)
  - No short selling
  - Flat +-10% price limit enforcement
  - 1,000-share board-lot rounding
  - Commission structure (brokerage fee bilateral, tax sell-only)
  - Slippage
"""

from __future__ import annotations

import pandas as pd
import pytest

from backtest.engines.tw_equity import PRICE_LIMIT, TWEquityEngine, _calc_pct_change
from backtest.models import Position


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_bar(
    close: float = 600.0,
    pre_close: float | None = None,
    pct_chg: float | None = None,
    trade_date: str | None = None,
    open_: float | None = None,
) -> pd.Series:
    """Build a minimal bar Series for testing."""
    d: dict = {"close": close, "open": open_ or close}
    if pre_close is not None:
        d["pre_close"] = pre_close
    if pct_chg is not None:
        d["pct_chg"] = pct_chg
    if trade_date is not None:
        d["trade_date"] = pd.Timestamp(trade_date)
    return pd.Series(d)


def _make_engine(**overrides) -> TWEquityEngine:
    config = {"initial_cash": 1_000_000}
    config.update(overrides)
    return TWEquityEngine(config)


# ---------------------------------------------------------------------------
# can_execute: no short selling
# ---------------------------------------------------------------------------


class TestNoShortSelling:
    def test_short_blocked(self) -> None:
        engine = _make_engine()
        bar = _make_bar()
        assert engine.can_execute("2330.TW", -1, bar) is False

    def test_long_allowed(self) -> None:
        engine = _make_engine()
        bar = _make_bar()
        assert engine.can_execute("2330.TW", 1, bar) is True

    def test_close_allowed_when_no_position(self) -> None:
        engine = _make_engine()
        bar = _make_bar()
        assert engine.can_execute("2330.TW", 0, bar) is True


# ---------------------------------------------------------------------------
# can_execute: day trading (no T+1 lock, unlike China A)
# ---------------------------------------------------------------------------


class TestDayTradingAllowed:
    def test_sell_same_day_allowed(self) -> None:
        """Unlike ChinaAEngine, TW allows selling same-day-bought shares (當沖)."""
        engine = _make_engine()
        engine.positions["2330.TW"] = Position(
            symbol="2330.TW",
            direction=1,
            entry_price=600.0,
            entry_time=pd.Timestamp("2025-06-10"),
            size=1000.0,
        )
        bar = _make_bar(trade_date="2025-06-10")
        assert engine.can_execute("2330.TW", 0, bar) is True


# ---------------------------------------------------------------------------
# can_execute: price limits (flat +-10%, no tiered board)
# ---------------------------------------------------------------------------


class TestPriceLimits:
    def test_limit_up_buy_blocked(self) -> None:
        engine = _make_engine()
        bar = _make_bar(close=660.0, pre_close=600.0)  # +10%
        assert engine.can_execute("2330.TW", 1, bar) is False

    def test_limit_down_sell_blocked(self) -> None:
        engine = _make_engine()
        engine.positions["2330.TW"] = Position(
            "2330.TW", 1, 600.0, pd.Timestamp("2025-06-09"), 1000.0,
        )
        bar = _make_bar(close=540.0, pre_close=600.0, trade_date="2025-06-10")
        assert engine.can_execute("2330.TW", 0, bar) is False

    def test_within_limit_allowed(self) -> None:
        engine = _make_engine()
        bar = _make_bar(close=620.0, pre_close=600.0)  # +3.3%
        assert engine.can_execute("2330.TW", 1, bar) is True

    def test_pct_chg_field_used(self) -> None:
        engine = _make_engine()
        bar = _make_bar(pct_chg=10.0)  # 10% -> +0.10
        assert engine.can_execute("2330.TW", 1, bar) is False

    def test_no_tiered_board_20pct_still_blocked_at_10pct(self) -> None:
        """TW has no ChiNext-style 20% tier -- every listed stock uses the same 10% limit."""
        engine = _make_engine()
        bar = _make_bar(close=720.0, pre_close=600.0)  # +20%, would pass under China A's ChiNext tier
        assert engine.can_execute("3008.TW", 1, bar) is False


# ---------------------------------------------------------------------------
# round_size: 1,000-share board lots
# ---------------------------------------------------------------------------


class TestRoundSize:
    def test_exact_lots(self) -> None:
        engine = _make_engine()
        assert engine.round_size(3000.0, 600.0) == 3000

    def test_rounds_down(self) -> None:
        engine = _make_engine()
        assert engine.round_size(3500.0, 600.0) == 3000
        assert engine.round_size(1999.0, 600.0) == 1000
        assert engine.round_size(999.0, 600.0) == 0

    def test_zero_size(self) -> None:
        engine = _make_engine()
        assert engine.round_size(0.0, 600.0) == 0

    def test_negative_clamps_to_zero(self) -> None:
        engine = _make_engine()
        assert engine.round_size(-500.0, 600.0) == 0


# ---------------------------------------------------------------------------
# calc_commission: fee structure
# ---------------------------------------------------------------------------


class TestCommission:
    def test_minimum_commission(self) -> None:
        """Small trades hit the NT$20 minimum."""
        engine = _make_engine()
        # 1 share x NT$10 = NT$10 notional -> 0.1425% = NT$0.01 -> min NT$20
        comm = engine.calc_commission(1, 10.0, 1, is_open=True)
        assert comm >= 20.0

    def test_buy_no_transaction_tax(self) -> None:
        """Buy side: no transaction tax."""
        engine = _make_engine()
        comm_buy = engine.calc_commission(1000, 600.0, 1, is_open=True)
        comm_sell = engine.calc_commission(1000, 600.0, 1, is_open=False)
        assert comm_sell > comm_buy

    def test_real_trade_fee_math(self) -> None:
        """Hand-computed example: 1,000 shares @ NT$600 (notional NT$600,000).

        Brokerage fee 0.1425% bilateral, transaction tax 0.3% sell-only --
        verify the engine reproduces TWSE's published fee schedule to the cent.
        """
        engine = _make_engine()
        size, price = 1000, 600.0
        notional = size * price  # 600,000

        comm_buy = engine.calc_commission(size, price, 1, is_open=True)
        expected_buy = notional * 0.001425  # NT$855.0
        assert comm_buy == pytest.approx(expected_buy, abs=0.01)

        comm_sell = engine.calc_commission(size, price, 1, is_open=False)
        expected_sell = notional * 0.001425 + notional * 0.003  # NT$855 + NT$1,800
        assert comm_sell == pytest.approx(expected_sell, abs=0.01)

    def test_custom_commission_rate(self) -> None:
        """Brokers commonly discount the headline 0.1425% rate -- must be configurable."""
        engine = _make_engine(commission_rate=0.001425 * 0.6)  # 6-fold discount
        size, price = 10000, 600.0
        notional = size * price
        comm = engine.calc_commission(size, price, 1, is_open=True)
        assert comm == pytest.approx(notional * 0.001425 * 0.6, abs=0.01)

    def test_leverage_forced_one(self) -> None:
        """TW equity engine forces leverage=1 by default."""
        engine = _make_engine(leverage=10.0)
        assert engine.default_leverage == 1.0


# ---------------------------------------------------------------------------
# apply_slippage
# ---------------------------------------------------------------------------


class TestSlippage:
    def test_buy_slippage_increases_price(self) -> None:
        engine = _make_engine()
        assert engine.apply_slippage(600.0, 1) > 600.0

    def test_sell_slippage_decreases_price(self) -> None:
        engine = _make_engine()
        assert engine.apply_slippage(600.0, -1) < 600.0

    def test_custom_slippage_rate(self) -> None:
        engine = _make_engine(slippage=0.005)
        assert engine.apply_slippage(600.0, 1) == pytest.approx(603.0)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


class TestPriceLimitConstant:
    def test_flat_ten_percent(self) -> None:
        assert PRICE_LIMIT == 0.10


class TestCalcPctChange:
    def test_from_pct_chg_field(self) -> None:
        bar = _make_bar(pct_chg=5.0)
        assert _calc_pct_change(bar) == pytest.approx(0.05)

    def test_from_close_and_pre_close(self) -> None:
        bar = _make_bar(close=660.0, pre_close=600.0)
        assert _calc_pct_change(bar) == pytest.approx(0.1)

    def test_none_when_no_data(self) -> None:
        bar = pd.Series({"close": 600.0})
        assert _calc_pct_change(bar) is None
