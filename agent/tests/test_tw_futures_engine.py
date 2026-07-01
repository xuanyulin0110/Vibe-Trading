"""Tests for TWFuturesEngine (TAIFEX index futures: TXF / MXF / TMF).

Validates:
  - Product code extraction from .TWF symbols (TXFR1 -> TXF)
  - Contract multiplier lookup (TXF=200, MXF=50, TMF=20)
  - Fixed per-contract margin (decoupled from sizing leverage)
  - ±10% price limit, T+0, both directions allowed
  - Integer contract rounding
  - Commission (per-contract broker fee + bilateral transaction tax)
  - Market detection routing (.TWF -> tw_futures, no collision)
"""

from __future__ import annotations

import pandas as pd
import pytest

from backtest.engines.tw_futures import (
    _MARGIN_PER_CONTRACT,
    _MULTIPLIER,
    _TRANSACTION_TAX_RATE,
    TWFuturesEngine,
    _extract_product,
)
from backtest.models import Position


def _make_bar(
    close: float = 20000.0,
    pre_close: float | None = None,
    pct_chg: float | None = None,
    open_: float | None = None,
) -> pd.Series:
    d: dict = {"close": close, "open": open_ or close}
    if pre_close is not None:
        d["pre_close"] = pre_close
    if pct_chg is not None:
        d["pct_chg"] = pct_chg
    return pd.Series(d)


def _make_engine(**overrides) -> TWFuturesEngine:
    config = {"initial_cash": 1_000_000, "codes": ["TXFR1.TWF"]}
    config.update(overrides)
    return TWFuturesEngine(config)


class TestExtractProduct:
    @pytest.mark.parametrize(
        "symbol, expected",
        [
            ("TXFR1.TWF", "TXF"),
            ("TXFR2.TWF", "TXF"),
            ("MXFR1.TWF", "MXF"),
            ("TMFR1.TWF", "TMF"),
            ("TMF202503.TWF", "TMF"),
            ("TXFG5.TWF", "TXF"),
            ("TXFR1", "TXF"),  # suffix optional for extraction
        ],
    )
    def test_extract(self, symbol: str, expected: str) -> None:
        assert _extract_product(symbol) == expected


class TestContractMultiplier:
    @pytest.mark.parametrize(
        "symbol, expected",
        [("TXFR1.TWF", 200), ("MXFR1.TWF", 50), ("TMFR1.TWF", 20)],
    )
    def test_multiplier(self, symbol: str, expected: int) -> None:
        engine = _make_engine()
        assert engine.get_contract_multiplier(symbol) == expected

    def test_unknown_product_falls_back_to_txf(self) -> None:
        engine = _make_engine()
        assert engine.get_contract_multiplier("ZZZ1.TWF") == 200


class TestMargin:
    @pytest.mark.parametrize(
        "symbol, expected",
        [("TXFR1.TWF", 636_000.0), ("MXFR1.TWF", 159_000.0), ("TMFR1.TWF", 31_800.0)],
    )
    def test_per_contract_margin(self, symbol: str, expected: float) -> None:
        engine = _make_engine()
        assert engine.get_margin_per_contract(symbol) == expected

    def test_calc_margin_uses_fixed_amount_not_leverage(self) -> None:
        """_calc_margin must return contracts x fixed margin, ignoring price/leverage."""
        engine = _make_engine()
        # 2 TXF contracts, whatever price/leverage -> 2 x 636,000
        margin = engine._calc_margin("TXFR1.TWF", size=2, price=20000.0, leverage=10.0)
        assert margin == pytest.approx(2 * 636_000.0)

    def test_margin_override_single_number(self) -> None:
        engine = _make_engine(margin_per_contract=500_000.0)
        assert engine.get_margin_per_contract("TXFR1.TWF") == 500_000.0
        assert engine.get_margin_per_contract("MXFR1.TWF") == 500_000.0

    def test_margin_override_per_product_dict(self) -> None:
        engine = _make_engine(margin_per_contract={"TXF": 700_000.0})
        assert engine.get_margin_per_contract("TXFR1.TWF") == 700_000.0
        # Products absent from the override dict fall back to the built-in table.
        assert engine.get_margin_per_contract("MXFR1.TWF") == 159_000.0


class TestDirectionAndT0:
    def test_long_allowed(self) -> None:
        engine = _make_engine()
        assert engine.can_execute("TXFR1.TWF", 1, _make_bar()) is True

    def test_short_allowed(self) -> None:
        """Unlike equities, futures allow opening shorts."""
        engine = _make_engine()
        assert engine.can_execute("TXFR1.TWF", -1, _make_bar()) is True

    def test_same_day_close_allowed(self) -> None:
        """T+0: a position opened today can be closed today (no bar-date check)."""
        engine = _make_engine()
        engine.positions["TXFR1.TWF"] = Position(
            "TXFR1.TWF", 1, 20000.0, pd.Timestamp("2025-06-10"), 1.0,
        )
        assert engine.can_execute("TXFR1.TWF", 0, _make_bar()) is True


class TestPriceLimits:
    def test_limit_up_blocks_long(self) -> None:
        engine = _make_engine()
        bar = _make_bar(close=22000.0, pre_close=20000.0)  # +10%
        assert engine.can_execute("TXFR1.TWF", 1, bar) is False

    def test_limit_down_blocks_short(self) -> None:
        engine = _make_engine()
        bar = _make_bar(close=18000.0, pre_close=20000.0)  # -10%
        assert engine.can_execute("TXFR1.TWF", -1, bar) is False

    def test_within_limit_allowed(self) -> None:
        engine = _make_engine()
        bar = _make_bar(close=20600.0, pre_close=20000.0)  # +3%
        assert engine.can_execute("TXFR1.TWF", 1, bar) is True

    def test_no_reference_price_skips_limit(self) -> None:
        """A bar with only close (Shioaji daily bars) can't compute pct change -> allowed."""
        engine = _make_engine()
        assert engine.can_execute("TXFR1.TWF", 1, _make_bar(close=20000.0)) is True

    def test_cannot_close_long_at_limit_down(self) -> None:
        engine = _make_engine()
        engine.positions["TXFR1.TWF"] = Position(
            "TXFR1.TWF", 1, 20000.0, pd.Timestamp("2025-06-09"), 1.0,
        )
        bar = _make_bar(close=18000.0, pre_close=20000.0)  # -10%
        assert engine.can_execute("TXFR1.TWF", 0, bar) is False


class TestRoundSize:
    def test_integer_contracts(self) -> None:
        engine = _make_engine()
        assert engine.round_size(2.9, 20000.0) == 2
        assert engine.round_size(0.5, 20000.0) == 0

    def test_negative_clamped(self) -> None:
        engine = _make_engine()
        assert engine.round_size(-3.0, 20000.0) == 0


class TestCommission:
    def test_commission_and_tax(self) -> None:
        """1 TXF contract @ 20,000 pts: broker fee NT$50 + tax = 1*20000*200*0.00002."""
        engine = _make_engine(commission_per_contract=50.0)
        engine._active_symbol = "TXFR1.TWF"
        comm = engine.calc_commission(1, 20000.0, 1, is_open=True)
        expected_tax = 1 * 20000.0 * 200 * _TRANSACTION_TAX_RATE  # = 80.0
        assert comm == pytest.approx(50.0 + expected_tax)

    def test_commission_scales_with_contracts(self) -> None:
        engine = _make_engine(commission_per_contract=50.0)
        engine._active_symbol = "TXFR1.TWF"
        one = engine.calc_commission(1, 20000.0, 1, is_open=True)
        three = engine.calc_commission(3, 20000.0, 1, is_open=True)
        assert three == pytest.approx(3 * one)

    def test_default_leverage_is_one(self) -> None:
        engine = _make_engine()
        assert engine.default_leverage == 1.0


class TestSlippage:
    def test_buy_raises_sell_lowers(self) -> None:
        engine = _make_engine()
        assert engine.apply_slippage(20000.0, 1) > 20000.0
        assert engine.apply_slippage(20000.0, -1) < 20000.0


class TestTables:
    def test_multiplier_and_margin_products_match(self) -> None:
        assert set(_MULTIPLIER) == set(_MARGIN_PER_CONTRACT) == {"TXF", "MXF", "TMF"}


class TestMarketDetection:
    @pytest.mark.parametrize(
        "symbol, market",
        [
            ("TXFR1.TWF", "tw_futures"),
            ("MXFR2.TWF", "tw_futures"),
            ("TMF202503.TWF", "tw_futures"),
            ("2330.TW", "tw_equity"),
            ("IF2406.CFFEX", "futures"),
            ("ESZ4", "futures"),
            ("ES.CME", "futures"),
        ],
    )
    def test_detect_market(self, symbol: str, market: str) -> None:
        from backtest.engines._market_hooks import _detect_market

        assert _detect_market(symbol) == market

    def test_is_tw_futures_helper(self) -> None:
        from backtest.engines._market_hooks import _is_tw_futures

        assert _is_tw_futures("TXFR1.TWF") is True
        assert _is_tw_futures("IF2406.CFFEX") is False
        assert _is_tw_futures("2330.TW") is False

    def test_detect_source_routes_twf_to_shioaji_futures(self) -> None:
        from src.market_data import detect_source

        assert detect_source("TXFR1.TWF") == "shioaji_futures"
        assert detect_source("MXFR1.TWF") == "shioaji_futures"


class TestEngineRouting:
    def test_create_market_engine_returns_tw_futures_engine(self) -> None:
        from backtest.runner import _create_market_engine

        engine = _create_market_engine("shioaji_futures", {"initial_cash": 1_000_000}, ["TXFR1.TWF"])
        assert isinstance(engine, TWFuturesEngine)
