"""Taiwan index futures (TAIFEX) backtest engine.

Covers the TAIEX index futures family traded on TAIFEX:
  - TXF 大台 (臺股期貨): NT$200 per index point
  - MXF 小型臺指:        NT$50  per index point
  - TMF 微型臺指:        NT$20  per index point

Market rules:
  - T+0 intraday trading, both long and short allowed (no equity-style
    short-sale restriction, no T+1 hold).
  - Price limit: ±10% from the previous day's close (TAIFEX dynamic price
    stabilisation baseline). Only enforced when the bar carries a
    prior-price reference (``pre_close``/``settle``/``pct_chg``); Shioaji
    daily bars resampled from 1-minute K-bars do not, so the limit check is
    a no-op there — same limitation as ``tw_equity``.
  - Minimum trading unit: 1 contract (integer lots).
  - Fees: broker commission per contract (NT$, configurable) + futures
    transaction tax 0.00002 per side of notional (0.002% each of buy/sell;
    the plan's 0.04‰ round-trip is 0.00002 per side).

Margin design (see plan Phase 2b): TAIFEX margin is a fixed NT$ amount per
contract that the exchange revises with index level / volatility — NOT a
clean percentage of notional. So this engine does NOT derive leverage from a
margin rate (the ChinaFuturesEngine ``leverage = 1/margin_rate`` trick would
be wrong here). Instead:
  - ``default_leverage`` (config ``leverage``, default 1.0) purely controls
    position *sizing* intent (how much notional a target weight maps to).
  - ``_calc_margin()`` is overridden to return the real fixed per-contract
    margin from a lookup table, decoupled from sizing leverage.

The per-contract margin figures below are the TAIFEX 原始保證金 (initial
margin) as published at implementation time (2026). They drift as TAIFEX
revises them — override via config ``margin_per_contract`` (a dict keyed by
product, or a single number applied to all) and re-check against
taifex.com.tw/cht/5/indexMarging periodically.
"""

from __future__ import annotations

import re

import pandas as pd

from backtest.engines.futures_base import FuturesBaseEngine

# ── Contract multiplier (NT$ per index point) ──
_MULTIPLIER: dict[str, int] = {
    "TXF": 200,  # 大台
    "MXF": 50,   # 小型臺指
    "TMF": 20,   # 微型臺指
}

# ── Initial margin (NT$ per contract), TAIFEX 原始保證金, 2026 baseline ──
_MARGIN_PER_CONTRACT: dict[str, float] = {
    "TXF": 636_000.0,
    "MXF": 159_000.0,
    "TMF": 31_800.0,
}

# ── Price limit (fraction, ± from prior close) ──
_PRICE_LIMIT = 0.10

# ── Fees ──
_DEFAULT_COMMISSION_PER_CONTRACT = 50.0  # NT$ per contract per side (broker-dependent)
_TRANSACTION_TAX_RATE = 0.00002  # per side of notional (0.04‰ round-trip)

_KNOWN_PRODUCTS = tuple(_MULTIPLIER)  # ("TXF", "MXF", "TMF")


def _extract_product(symbol: str) -> str:
    """Extract the TAIFEX product code from a futures symbol.

    Examples:
        'TXFR1.TWF'  -> 'TXF'   (continuous near-month)
        'MXFR2.TWF'  -> 'MXF'
        'TMF202503.TWF' -> 'TMF'

    Continuous-contract suffixes (``R1``/``R2``) start with a letter, so a
    naive leading-letters regex would return ``TXFR`` for ``TXFR1``. Match
    against the known product prefixes first; fall back to leading letters.
    """
    code = symbol.split(".")[0].upper()
    for product in _KNOWN_PRODUCTS:
        if code.startswith(product):
            return product
    m = re.match(r"([A-Z]+)", code)
    return m.group(1) if m else code


class TWFuturesEngine(FuturesBaseEngine):
    """TAIFEX index futures engine (TXF / MXF / TMF).

    Config keys:
      - leverage: sizing leverage (default 1.0; see margin design in module docstring)
      - slippage: default 0.0005
      - commission_per_contract: NT$ per contract per side (default 50)
      - margin_per_contract: override initial margin; a number (applied to all
        products) or a dict keyed by product code
    """

    def __init__(self, config: dict):
        leverage = config.get("leverage", 1.0)
        config = {**config, "leverage": leverage}
        super().__init__(config)
        self.slippage_rate: float = config.get("slippage", 0.0005)
        self._commission_per_contract: float = config.get(
            "commission_per_contract", _DEFAULT_COMMISSION_PER_CONTRACT
        )
        self._margin_override = config.get("margin_per_contract")

    # ── Market rules ──

    def can_execute(self, symbol: str, direction: int, bar: pd.Series) -> bool:
        """TAIFEX futures: T+0, both directions, ±10% price limit."""
        pct_chg = _calc_pct_change(bar)
        if pct_chg is not None:
            if direction == 1 and pct_chg >= _PRICE_LIMIT - 0.001:
                return False  # limit-up: can't open long
            if direction == -1 and pct_chg <= -_PRICE_LIMIT + 0.001:
                return False  # limit-down: can't open short
            if direction == 0:
                pos = self.positions.get(symbol)
                if pos is not None:
                    if pos.direction == 1 and pct_chg <= -_PRICE_LIMIT + 0.001:
                        return False  # can't close long at limit-down
                    if pos.direction == -1 and pct_chg >= _PRICE_LIMIT - 0.001:
                        return False  # can't close short at limit-up
        return True

    def round_size(self, raw_size: float, price: float) -> float:
        """Integer contracts, minimum 1."""
        return max(int(raw_size), 0)

    def calc_commission(self, size: float, price: float, _direction: int, is_open: bool) -> float:
        """Broker commission per contract + TAIFEX transaction tax (both sides).

        ``_direction`` is unused — reserved for future asymmetric fee schedules.
        """
        product = _extract_product(self._active_symbol)
        cm = _MULTIPLIER.get(product, 200)
        commission = size * self._commission_per_contract
        tax = size * price * cm * _TRANSACTION_TAX_RATE
        return commission + tax

    def apply_slippage(self, price: float, direction: int) -> float:
        """Futures slippage."""
        return price * (1 + direction * self.slippage_rate)

    # ── Futures multiplier / margin ──

    def get_contract_multiplier(self, symbol: str) -> float:
        """Look up NT$-per-point multiplier from the product code."""
        return float(_MULTIPLIER.get(_extract_product(symbol), 200))

    def get_margin_per_contract(self, symbol: str) -> float:
        """Fixed initial margin (NT$) for one contract of ``symbol``.

        Honours the ``margin_per_contract`` config override (number or
        per-product dict) before falling back to the built-in table.
        """
        product = _extract_product(symbol)
        override = self._margin_override
        if isinstance(override, dict):
            if product in override:
                return float(override[product])
        elif override is not None:
            return float(override)
        return _MARGIN_PER_CONTRACT.get(product, 636_000.0)

    def _calc_margin(
        self, symbol: str, size: float, price: float, leverage: float,
    ) -> float:
        """Real capital tied up = contracts × fixed per-contract initial margin.

        Overrides ``FuturesBaseEngine._calc_margin`` (which is
        ``size*price*cm/leverage``) because TAIFEX margin is a fixed NT$ amount
        per contract, not a percentage of notional. ``leverage`` is ignored
        here on purpose — it only governs sizing via ``_calc_raw_size``.
        """
        return size * self.get_margin_per_contract(symbol)


# ── Helpers ──


def _calc_pct_change(bar: pd.Series):
    """Bar price change fraction. Priority: close/pre_close > settle/pre_settle > pct_chg.

    Returns None when the bar carries no prior-price reference (e.g. Shioaji
    daily bars resampled from 1-minute K-bars), in which case the price-limit
    check is skipped.
    """
    close = bar.get("close")
    pre_close = bar.get("pre_close")
    if close is not None and pre_close is not None and pre_close > 0:
        return (float(close) - float(pre_close)) / float(pre_close)

    settle = bar.get("settle")
    pre_settle = bar.get("pre_settle")
    if settle is not None and pre_settle is not None and pre_settle > 0:
        return (float(settle) - float(pre_settle)) / float(pre_settle)

    if "pct_chg" in bar.index:
        val = bar["pct_chg"]
        if pd.notna(val):
            return float(val) / 100.0

    return None
