"""finlab chip/fundamental data provider: 三大法人/融資融券/月營收.

finlab's date index for these tables is already the public
disclosure/trading date -- e.g. ``monthly_revenue:當月營收`` is indexed by
its actual announcement date (commonly the 10th of the following month,
per TWSE's monthly-revenue disclosure deadline), not the period-end date.
``institutional_investors_trading_summary`` and ``margin_transactions`` are
daily end-of-day statistics published the same trading day. A simple
backward as-of merge against price frames is therefore already
point-in-time safe -- no extra announcement-date offset is needed, unlike
Tushare's statement tables (see ``tushare_fundamentals.py``) which carry a
separate ``ann_date``/``f_ann_date`` column for exactly that reason.

Field-key strings were confirmed against the installed ``finlab`` package's
``finlab.data.search(...)`` catalogue at the time this module was written;
re-confirm if finlab renames a dataset.
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional

import pandas as pd

from backtest.loaders._symbol_utils import _strip_tw_suffix

_TABLE_FIELDS: Dict[str, Dict[str, str]] = {
    "institutional": {
        "foreign_net": "institutional_investors_trading_summary:外資自營商買賣超股數",
        "foreign_ex_dealer_net": "institutional_investors_trading_summary:外陸資買賣超股數(不含外資自營商)",
        "trust_net": "institutional_investors_trading_summary:投信買賣超股數",
        "dealer_self_net": "institutional_investors_trading_summary:自營商買賣超股數(自行買賣)",
        "dealer_hedge_net": "institutional_investors_trading_summary:自營商買賣超股數(避險)",
    },
    "margin": {
        "margin_balance": "margin_transactions:融資今日餘額",
        "margin_buy": "margin_transactions:融資買進",
        "margin_sell": "margin_transactions:融資賣出",
        "margin_usage_rate": "margin_transactions:融資使用率",
        "short_balance": "margin_transactions:融券今日餘額",
        "short_buy": "margin_transactions:融券買進",
        "short_sell": "margin_transactions:融券賣出",
    },
    "monthly_revenue": {
        "revenue": "monthly_revenue:當月營收",
        "revenue_yoy_pct": "monthly_revenue:去年同月增減(%)",
        "revenue_mom_pct": "monthly_revenue:上月比較增減(%)",
    },
}


class FinlabFundamentalProvider:
    """Whole-market wide-table provider for finlab chip/fundamental data."""

    def __init__(self) -> None:
        self._field_cache: Dict[str, pd.DataFrame] = {}

    def list_tables(self) -> list[str]:
        """Return supported fundamental tables in stable order."""
        return sorted(_TABLE_FIELDS)

    def describe_table(self, table: str) -> Dict[str, str]:
        """Return the alias -> finlab field-key mapping for a supported table."""
        try:
            return _TABLE_FIELDS[table]
        except KeyError as exc:
            raise ValueError(f"Unsupported finlab fundamental table: {table}") from exc

    def _field_table(self, field_key: str) -> pd.DataFrame:
        """Return the whole-market wide table for one field, fetched once per instance."""
        if field_key not in self._field_cache:
            from finlab import data

            self._field_cache[field_key] = data.get(field_key)
        return self._field_cache[field_key]

    def query_fundamentals(
        self,
        table: str,
        codes: Iterable[str],
        *,
        fields: Optional[Iterable[str]] = None,
    ) -> Dict[str, pd.DataFrame]:
        """Return ``{code: DataFrame(date-indexed, columns=requested aliases)}`` for one table.

        Args:
            table: One of ``list_tables()`` (institutional/margin/monthly_revenue).
            codes: Stock codes (e.g. ``2330.TW``).
            fields: Alias names from ``describe_table(table)``; defaults to all.

        Returns:
            Mapping code -> DataFrame, omitting codes with no data in this table.
        """
        field_map = self.describe_table(table)
        field_list = list(fields or field_map.keys())
        unknown = [f for f in field_list if f not in field_map]
        if unknown:
            raise ValueError(f"Unknown fields for table {table!r}: {unknown}")

        result: Dict[str, pd.DataFrame] = {}
        for code in codes:
            stock_id = _strip_tw_suffix(code)
            columns: Dict[str, pd.Series] = {}
            for alias in field_list:
                wide = self._field_table(field_map[alias])
                if stock_id in wide.columns:
                    columns[alias] = wide[stock_id]
            if columns:
                result[code] = pd.DataFrame(columns)
        return result


def enrich_price_frames_with_finlab_fundamentals(
    data_map: Dict[str, pd.DataFrame],
    provider: FinlabFundamentalProvider,
    fields_by_table: Dict[str, Iterable[str]],
) -> Dict[str, pd.DataFrame]:
    """Attach PIT-safe finlab chip/fundamental columns to daily price frames.

    Columns are prefixed with their table name, e.g. ``institutional_trust_net``,
    ``margin_margin_balance``, ``monthly_revenue_revenue``.
    """
    if not data_map or not fields_by_table:
        return data_map

    enriched = {code: frame.copy() for code, frame in data_map.items()}

    for table, fields in fields_by_table.items():
        field_list = list(fields or [])
        per_code = provider.query_fundamentals(table, list(enriched), fields=field_list)

        for code, frame in enriched.items():
            rows = per_code.get(code)
            if rows is None or rows.empty or frame.empty:
                continue

            right = rows.rename(columns={c: f"{table}_{c}" for c in rows.columns})
            right.index.name = "_pit_date"
            right = right.reset_index()

            left = frame.copy()
            original_index = left.index
            left["_trade_date"] = pd.to_datetime(left.index).normalize()
            left["_original_order"] = range(len(left))

            merged = pd.merge_asof(
                left.sort_values("_trade_date"),
                right.sort_values("_pit_date"),
                left_on="_trade_date",
                right_on="_pit_date",
                direction="backward",
            )
            merged = merged.sort_values("_original_order").drop(
                columns=["_trade_date", "_original_order", "_pit_date"]
            )
            merged.index = original_index
            enriched[code] = merged

    return enriched
