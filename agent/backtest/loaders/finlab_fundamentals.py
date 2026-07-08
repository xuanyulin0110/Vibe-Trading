"""finlab chip/fundamental data provider.

Covers 三大法人/融資融券/月營收 (the original three tables) plus the Phase 2d
expansion: 財報科目 (financial_statement), 財務比率 (fundamental_features),
外資持股 (foreign_investors_shareholding), 興櫃月營收 (rotc_monthly_revenue),
董監持股不足 (internal_equity_insufficient), and 期貨三大法人
(futures_institutional_investors_trading_summary). Every dataset prefix here
was confirmed to exist against the installed ``finlab`` package's
``finlab.data.search(...)`` catalogue at the time this module was written;
re-confirm if finlab renames a dataset.

Date-index semantics differ per table -- do NOT assume they're all alike:

  - ``institutional`` / ``margin`` / ``monthly_revenue`` / ``rotc_monthly_revenue``
    / ``foreign_shareholding``: finlab's date index IS the public
    disclosure/trading date (e.g. ``monthly_revenue:當月營收`` is indexed by its
    actual announcement date, commonly the 10th of the following month). A
    plain backward as-of merge against price frames is already
    point-in-time safe.
  - ``financial_statement`` / ``fundamental_features``: finlab indexes these
    by **quarter-period label** (``'2025-Q1'``), not a date at all -- using
    the index directly would be wrong AND wouldn't even merge_asof
    (non-comparable to a DatetimeIndex). The real per-stock disclosure date
    per quarter lives in a separate table, ``etl:financial_statements_disclosure_dates``
    (same quarter-label index/columns shape, values = actual filing date).
    Confirmed empirically: TSMC's (2330) 2025-Q1 statement wasn't disclosed
    until 2025-05-15, ~45 days after quarter-end -- using the quarter-end as
    the PIT date would leak ~45 days of look-ahead. ``query_fundamentals``
    resolves the quarter label -> real date via that companion table before
    handing back a date-indexed series, so the point-in-time merge in
    ``enrich_price_frames_with_finlab_fundamentals`` stays correct.
  - ``director_shareholding``: finlab's index here already looks like a real
    date (not a quarter label), but no companion disclosure-date table was
    found for it (unlike financial_statement). Treated as-is, same as the
    other already-dated tables; this dataset is sparse (only populated when
    a company actually has an insufficient-holding flag) so most codes will
    simply have no rows.
  - ``futures_institutional``: a whole different shape. Columns are
    ``"{TAIFEX product name}_{investor type}"`` (e.g. ``臺股期貨_外資及陸資``
    for TXF foreign net), not stock codes -- there's no ``.TW`` symbol to
    strip. Codes here are ``.TWF`` futures symbols; the product code
    (TXF/MXF/TMF) is extracted and mapped to its Chinese column prefix, and
    the investor type (外資及陸資/投信/自營商) is baked into the alias itself
    since one measure spans three investor-type columns.
"""

from __future__ import annotations

import os
from typing import Dict, Iterable, Optional, Tuple

import pandas as pd

from backtest.loaders._symbol_utils import _strip_tw_suffix
from backtest.loaders.finlab_loader import FINLAB_TOKEN_PLACEHOLDERS

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
    "rotc_monthly_revenue": {
        "revenue": "rotc_monthly_revenue:當月營收",
        "revenue_yoy_pct": "rotc_monthly_revenue:去年同月增減(%)",
        "revenue_mom_pct": "rotc_monthly_revenue:上月比較增減(%)",
    },
    "foreign_shareholding": {
        "shares_held": "foreign_investors_shareholding:全體外資及陸資持有股數",
        "holding_pct": "foreign_investors_shareholding:全體外資及陸資持股比率",
        "remaining_investable_pct": "foreign_investors_shareholding:外資及陸資尚可投資比率",
    },
    "director_shareholding": {
        "director_insufficient_shares": "internal_equity_insufficient:全體董事(不包含獨立董事)不足股數",
        "supervisor_insufficient_shares": "internal_equity_insufficient:監察人應持有股數不足股數",
    },
    # Quarter-period-indexed -- see module docstring. query_fundamentals()
    # resolves these through _DISCLOSURE_DATE_KEY before returning.
    "financial_statement": {
        "total_assets": "financial_statement:資產總額",
        "total_liabilities": "financial_statement:負債總額",
        "total_equity": "financial_statement:股東權益總額",
        "revenue": "financial_statement:營業收入淨額",
        "gross_profit": "financial_statement:營業毛利",
        "operating_income": "financial_statement:營業利益",
        "net_income": "financial_statement:歸屬母公司淨利_損",
        "eps": "financial_statement:每股盈餘",
        "operating_cash_flow": "financial_statement:營業活動之淨現金流入_流出",
    },
    "fundamental_features": {
        "roe": "fundamental_features:ROE稅後",
        "roa": "fundamental_features:ROA稅後息前",
        "gross_margin": "fundamental_features:營業毛利率",
        "operating_margin": "fundamental_features:營業利益率",
        "net_margin": "fundamental_features:稅後淨利率",
        "revenue_growth": "fundamental_features:營收成長率",
        "current_ratio": "fundamental_features:流動比率",
        "debt_ratio": "fundamental_features:負債比率",
        "free_cash_flow": "fundamental_features:自由現金流量",
    },
    # Display-only entry -- actual querying goes through
    # _FUTURES_INSTITUTIONAL_FIELDS (needs the per-alias investor-type suffix
    # that a flat alias -> field-key string can't carry). Kept in sync with
    # _FUTURES_INSTITUTIONAL_FIELDS by the TestTables consistency test.
    "futures_institutional": {
        "foreign_net_oi": "futures_institutional_investors_trading_summary:多空未平倉口數淨額 [外資及陸資]",
        "trust_net_oi": "futures_institutional_investors_trading_summary:多空未平倉口數淨額 [投信]",
        "dealer_net_oi": "futures_institutional_investors_trading_summary:多空未平倉口數淨額 [自營商]",
        "foreign_net_volume": "futures_institutional_investors_trading_summary:多空交易口數淨額 [外資及陸資]",
        "trust_net_volume": "futures_institutional_investors_trading_summary:多空交易口數淨額 [投信]",
        "dealer_net_volume": "futures_institutional_investors_trading_summary:多空交易口數淨額 [自營商]",
    },
}

_QUARTER_INDEXED_TABLES = frozenset({"financial_statement", "fundamental_features"})
_DISCLOSURE_DATE_KEY = "etl:financial_statements_disclosure_dates"

_FUTURES_TABLE = "futures_institutional"

# alias -> (finlab field key, investor-type column suffix)
_FUTURES_INSTITUTIONAL_FIELDS: Dict[str, Tuple[str, str]] = {
    "foreign_net_oi": ("futures_institutional_investors_trading_summary:多空未平倉口數淨額", "外資及陸資"),
    "trust_net_oi": ("futures_institutional_investors_trading_summary:多空未平倉口數淨額", "投信"),
    "dealer_net_oi": ("futures_institutional_investors_trading_summary:多空未平倉口數淨額", "自營商"),
    "foreign_net_volume": ("futures_institutional_investors_trading_summary:多空交易口數淨額", "外資及陸資"),
    "trust_net_volume": ("futures_institutional_investors_trading_summary:多空交易口數淨額", "投信"),
    "dealer_net_volume": ("futures_institutional_investors_trading_summary:多空交易口數淨額", "自營商"),
}

# TAIFEX product code -> Chinese column-name prefix used by
# futures_institutional_investors_trading_summary (large-contract TXF is
# "臺股期貨" in finlab's naming, not "臺指期貨").
_FUTURES_PRODUCT_PREFIX: Dict[str, str] = {
    "TXF": "臺股期貨",
    "MXF": "小型臺指期貨",
    "TMF": "微型臺指期貨",
}
_KNOWN_TW_FUTURES_PRODUCTS = tuple(_FUTURES_PRODUCT_PREFIX)


def _extract_tw_futures_product(code: str) -> str:
    """Extract the TAIFEX product code from a ``.TWF`` symbol (``TXFR1.TWF`` -> ``TXF``)."""
    bare = code.split(".")[0].upper()
    for product in _KNOWN_TW_FUTURES_PRODUCTS:
        if bare.startswith(product):
            return product
    return bare[:3]


class FinlabFundamentalProvider:
    """Whole-market wide-table provider for finlab chip/fundamental data.

    Unlike ``FinlabLoader`` (the price loader), this provider has no login of
    its own by construction -- it previously assumed some other code path in
    the same process (typically the price loader, when finlab is the active
    price source) had already called ``finlab.login()``. That assumption
    breaks whenever Shioaji is the price source (the tw_equity/tw_futures
    default) and a caller still requests ``fundamental_fields`` enrichment:
    this provider would be the first and only finlab touch-point in that
    process. An empty/placeholder token then reaching ``finlab.login()``
    falls back to an interactive browser-auth flow that prints to stdout and
    blocks -- the same MCP-stdio-corruption failure mode fixed in
    ``finlab_loader.py`` (see its ``__init__`` docstring), just not
    previously guarded here too.
    """

    def __init__(self) -> None:
        self._field_cache: Dict[str, pd.DataFrame] = {}
        self._logged_in = False

    def _ensure_logged_in(self) -> None:
        if self._logged_in:
            return
        token = os.getenv("FINLAB_API_TOKEN", "")
        if token.strip() in FINLAB_TOKEN_PLACEHOLDERS:
            raise RuntimeError("FINLAB_API_TOKEN is not configured")
        import finlab

        finlab.login(token)
        self._logged_in = True

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
        """Return the whole-market wide table for one field, fetched once per instance.

        Drops any row whose index is NaT before caching. Confirmed live
        against ``futures_institutional_investors_trading_summary:多空未平倉口數淨額``
        (2026-07-09): finlab's own table carries exactly one such row --
        a genuine upstream data-quality glitch, not something our loader
        introduces. ``enrich_price_frames_with_finlab_fundamentals()``'s
        ``pd.merge_asof`` hard-requires non-null merge keys, so a single NaT
        anywhere in a wide table crashed enrichment for every code and
        every field ever selected from it, at every interval -- this
        symptom happened to surface via a 5m futures backtest, but the
        crash has nothing to do with 5m vs daily; the same NaT row would
        just as surely poison a daily-bar merge querying this table.
        """
        if field_key not in self._field_cache:
            self._ensure_logged_in()
            from finlab import data

            table = data.get(field_key)
            self._field_cache[field_key] = table[table.index.notna()]
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
            table: One of ``list_tables()``.
            codes: Stock codes (e.g. ``2330.TW``) for stock-oriented tables, or
                ``.TWF`` futures codes (e.g. ``TXFR1.TWF``) for
                ``futures_institutional``.
            fields: Alias names from ``describe_table(table)``; defaults to all.

        Returns:
            Mapping code -> DataFrame, omitting codes with no data in this table.
        """
        if table == _FUTURES_TABLE:
            return self._query_futures_institutional(codes, fields)

        field_map = self.describe_table(table)
        field_list = list(fields or field_map.keys())
        unknown = [f for f in field_list if f not in field_map]
        if unknown:
            raise ValueError(f"Unknown fields for table {table!r}: {unknown}")

        disclosure: Optional[pd.DataFrame] = None
        if table in _QUARTER_INDEXED_TABLES:
            disclosure = self._field_table(_DISCLOSURE_DATE_KEY)

        result: Dict[str, pd.DataFrame] = {}
        for code in codes:
            stock_id = _strip_tw_suffix(code)
            columns: Dict[str, pd.Series] = {}
            for alias in field_list:
                wide = self._field_table(field_map[alias])
                if stock_id not in wide.columns:
                    continue
                series = wide[stock_id]
                if disclosure is not None:
                    series = _resolve_quarter_index_to_dates(series, disclosure, stock_id)
                columns[alias] = series
            if columns:
                result[code] = pd.DataFrame(columns)
        return result

    def _query_futures_institutional(
        self, codes: Iterable[str], fields: Optional[Iterable[str]],
    ) -> Dict[str, pd.DataFrame]:
        field_list = list(fields or _FUTURES_INSTITUTIONAL_FIELDS.keys())
        unknown = [f for f in field_list if f not in _FUTURES_INSTITUTIONAL_FIELDS]
        if unknown:
            raise ValueError(f"Unknown fields for table {_FUTURES_TABLE!r}: {unknown}")

        result: Dict[str, pd.DataFrame] = {}
        for code in codes:
            product = _extract_tw_futures_product(code)
            prefix = _FUTURES_PRODUCT_PREFIX.get(product)
            if prefix is None:
                continue
            columns: Dict[str, pd.Series] = {}
            for alias in field_list:
                field_key, investor_suffix = _FUTURES_INSTITUTIONAL_FIELDS[alias]
                wide = self._field_table(field_key)
                column = f"{prefix}_{investor_suffix}"
                if column in wide.columns:
                    columns[alias] = wide[column]
            if columns:
                result[code] = pd.DataFrame(columns)
        return result


def _resolve_quarter_index_to_dates(
    series: pd.Series, disclosure: pd.DataFrame, stock_id: str,
) -> pd.Series:
    """Convert a quarter-period-indexed series (``'2025-Q1'``) to a real,
    PIT-safe date index using the per-stock disclosure date table.

    The field table and the disclosure-date table don't necessarily cover the
    same quarter range (the disclosure table has deeper history), so alignment
    is by quarter-label (``reindex``), not position. Quarters with no known
    disclosure date (not yet filed, or before the disclosure table's history)
    are dropped rather than guessed.
    """
    series = series.dropna()
    if stock_id not in disclosure.columns:
        return series.iloc[0:0]
    dates = disclosure[stock_id].reindex(series.index)
    valid = dates.notna()
    return pd.Series(series[valid].to_numpy(), index=dates[valid].to_numpy()).sort_index()


def enrich_price_frames_with_finlab_fundamentals(
    data_map: Dict[str, pd.DataFrame],
    provider: FinlabFundamentalProvider,
    fields_by_table: Dict[str, Iterable[str]],
) -> Dict[str, pd.DataFrame]:
    """Attach PIT-safe finlab chip/fundamental columns to daily price frames.

    Columns are prefixed with their table name, e.g. ``institutional_trust_net``,
    ``margin_margin_balance``, ``monthly_revenue_revenue``,
    ``financial_statement_net_income``, ``futures_institutional_foreign_net_oi``.
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
