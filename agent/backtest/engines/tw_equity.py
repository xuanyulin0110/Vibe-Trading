"""Taiwan equity (TWSE/TPEx) backtest engine.

Market rules:
  - Day trading (當沖) is allowed: unlike China A's T+1, shares bought today
    may be sold the same day. No same-day-sell restriction is enforced here.
  - Short selling is blocked by default (matches ChinaAEngine's conservative
    default) — real TW short selling requires 融券 availability and is
    subject to the uptick rule, neither of which is modelled here.
  - Price limit: flat ±10% from previous close for all listed stocks (no
    separate ChiNext/STAR-style tiered board).
  - Minimum lot: 1,000 shares (一張). Odd-lot (零股) trading is not modelled.
  - Brokerage fee: 0.1425% bilateral (headline rate; brokers commonly
    discount this — configurable via ``commission_rate``).
  - Securities transaction tax: 0.3% sell-side only.
"""

from __future__ import annotations

import pandas as pd

from backtest.engines.base import BaseEngine

BOARD_LOT_SIZE = 1000
PRICE_LIMIT = 0.10


class TWEquityEngine(BaseEngine):
    """Taiwan equity market engine.

    Config keys:
      - commission_rate: default 0.001425 (0.1425%, bilateral)
      - commission_min: default 20.0 (TWD; broker-dependent, configurable)
      - transaction_tax: default 0.003 (0.3%, sell-only)
      - slippage: default 0.001
    """

    def __init__(self, config: dict):
        config = {**config, "leverage": 1.0}  # TW equities: no leverage by default
        super().__init__(config)
        self.commission_rate: float = config.get("commission_rate", 0.001425)
        self.commission_min: float = config.get("commission_min", 20.0)
        self.transaction_tax: float = config.get("transaction_tax", 0.003)
        self.slippage_rate: float = config.get("slippage", 0.001)

    def can_execute(self, symbol: str, direction: int, bar: pd.Series) -> bool:
        """TW equity execution rules.

        Args:
            symbol: Stock code (e.g. 2330.TW).
            direction: 1 (buy), -1 (short — blocked), 0 (sell/close).
            bar: Current bar (needs 'close', 'pre_close' or 'pct_chg').

        Returns:
            True if the trade is allowed.
        """
        # 1. No short selling (simplification — see module docstring)
        if direction == -1:
            return False

        # 2. Price limits (flat +-10%, no same-day-sell restriction)
        pct_chg = _calc_pct_change(bar)
        if pct_chg is not None:
            if direction == 1 and pct_chg >= PRICE_LIMIT - 0.001:
                return False  # limit-up: can't buy
            if direction == 0 and pct_chg <= -PRICE_LIMIT + 0.001:
                return False  # limit-down: can't sell

        return True

    def round_size(self, raw_size: float, price: float) -> float:
        """Round down to 1,000-share board lots."""
        return max(int(raw_size / BOARD_LOT_SIZE) * BOARD_LOT_SIZE, 0)

    def calc_commission(self, size: float, price: float, _direction: int, is_open: bool) -> float:
        """TW fee structure: brokerage fee (bilateral) + transaction tax (sell-only).

        ``_direction`` is unused today — reserved for future asymmetric
        long/short fee schedules (margin trading, securities lending).
        """
        notional = size * price
        comm = max(notional * self.commission_rate, self.commission_min)
        if not is_open:
            comm += notional * self.transaction_tax
        return comm

    def apply_slippage(self, price: float, direction: int) -> float:
        """TW equity slippage."""
        return price * (1 + direction * self.slippage_rate)


# -- Helpers --


def _calc_pct_change(bar: pd.Series):
    """Calculate price change percentage from bar data."""
    if "pct_chg" in bar.index:
        val = bar["pct_chg"]
        if pd.notna(val):
            return float(val) / 100.0

    close = bar.get("close")
    pre_close = bar.get("pre_close")
    if close is not None and pre_close is not None and pre_close > 0:
        return (float(close) - float(pre_close)) / float(pre_close)
    return None
