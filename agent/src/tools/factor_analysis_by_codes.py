"""codes/factor_name/date-range adapter for the MCP ``factor_analysis`` tool.

``mcp_server.py``'s ``factor_analysis()`` tool has always advertised a
``codes``/``factor_name``/``start_date``/``end_date``/``source``/``top_n``/
``bottom_n`` interface, but the registered ``FactorAnalysisTool`` only ever
accepted precomputed ``factor_csv``/``return_csv``/``output_dir`` -- calling
``registry.execute("factor_analysis", {"codes": ..., ...})`` always raised
``KeyError: 'factor_csv'``. Confirmed identical in upstream HKUDS/Vibe-Trading
(git diff against ``upstream/main`` is empty for both files) -- this is not a
fork-local regression.

This module implements the interface the MCP tool actually promises: fetch
OHLCV price data for ``codes``, resolve ``factor_name`` against whichever
fundamentals provider matches the codes' market (reusing the *existing*,
already-tested point-in-time merge helpers -- ``enrich_price_frames_with_finlab_fundamentals``
for tw_equity/tw_futures, ``enrich_price_frames_with_fundamentals`` for
everything else -- rather than reshaping each provider's very different raw
output shape by hand), then runs the same IC/IR math
``factor_analysis_tool.py`` uses.
"""

from __future__ import annotations

import json
from typing import Any, Optional

import pandas as pd

from backtest.engines._market_hooks import _detect_market
from src.factors.factor_analysis_core import compute_ic_series
from src.market_data import detect_source, get_loader


def run_factor_analysis_by_codes(
    codes: list[str],
    factor_name: str,
    start_date: str,
    end_date: str,
    source: str = "auto",
    top_n: int = 10,
    bottom_n: int = 10,
) -> str:
    """Compute factor IC/IR and a top-N vs bottom-N return spread for ``codes``.

    Args:
        codes: Stock/futures codes (e.g. ``["000001.SZ", "600519.SH"]`` or
            ``["2330.TW", "2317.TW"]``).
        factor_name: Alias name from a fundamentals table (e.g. ``"roe"``,
            ``"pe_ttm"``-style Tushare fields where supported). See the
            ``available_fields`` list in the error response when a name
            isn't found.
        start_date: Start date (YYYY-MM-DD).
        end_date: End date (YYYY-MM-DD).
        source: Data source for prices ("auto" detects per code).
        top_n: Number of top-ranked codes per day for the spread.
        bottom_n: Number of bottom-ranked codes per day for the spread.

    Returns:
        JSON-formatted result string.
    """
    if not codes:
        return _error("codes must be a non-empty list")

    try:
        price_map = _fetch_prices(codes, start_date, end_date, source)
    except Exception as exc:
        return _error(f"failed to fetch price data: {exc}")
    if not price_map:
        return _error("no price data returned for the given codes/date range")

    market = _detect_market(codes[0])
    try:
        table, available_fields = _resolve_factor_table(market, factor_name)
    except Exception as exc:
        return _error(f"failed to resolve factor_name: {exc}")
    if table is None:
        return json.dumps({
            "status": "error",
            "error": f"factor_name {factor_name!r} not found in any fundamentals table for market {market!r}",
            "available_fields": available_fields,
        }, ensure_ascii=False)

    try:
        enriched = _enrich_with_factor(market, price_map, table, factor_name, end_date)
    except Exception as exc:
        return _error(f"failed to fetch factor data: {exc}")

    factor_col = f"{table}_{factor_name}"
    factor_df, return_df = _build_wide_tables(enriched, factor_col)
    if factor_df.empty or return_df.empty:
        return _error("no overlapping factor/price data across the requested codes and date range")

    ic_series = compute_ic_series(factor_df, return_df)
    if ic_series.empty:
        return _error(
            "IC computation failed: insufficient shared dates/codes "
            "(need at least 5 codes with both factor and return data per day)"
        )

    ic_mean = float(ic_series.mean())
    ic_std = float(ic_series.std())
    ir = ic_mean / ic_std if ic_std > 0 else 0.0
    ic_positive_ratio = float((ic_series > 0).mean())

    top_mean, bottom_mean = _top_bottom_mean_returns(factor_df, return_df, top_n, bottom_n)

    return json.dumps({
        "status": "ok",
        "factor_name": factor_name,
        "codes_used": sorted(factor_df.columns),
        "ic_mean": round(ic_mean, 6),
        "ic_std": round(ic_std, 6),
        "ir": round(ir, 4),
        "ic_positive_ratio": round(ic_positive_ratio, 4),
        "ic_count": int(len(ic_series)),
        "top_n": top_n,
        "bottom_n": bottom_n,
        "top_mean_return": round(top_mean, 6) if top_mean is not None else None,
        "bottom_mean_return": round(bottom_mean, 6) if bottom_mean is not None else None,
        "top_minus_bottom_spread": (
            round(top_mean - bottom_mean, 6) if top_mean is not None and bottom_mean is not None else None
        ),
    }, ensure_ascii=False)


def _error(message: str) -> str:
    return json.dumps({"status": "error", "error": message}, ensure_ascii=False)


def _fetch_prices(
    codes: list[str], start_date: str, end_date: str, source: str,
) -> dict[str, pd.DataFrame]:
    """Fetch raw OHLCV DataFrames (date-indexed) per code, grouped by data source."""
    groups: dict[str, list[str]] = {}
    if source == "auto":
        for code in codes:
            groups.setdefault(detect_source(code), []).append(code)
    else:
        groups[source] = list(codes)

    price_map: dict[str, pd.DataFrame] = {}
    for src, src_codes in groups.items():
        loader = get_loader(src)()
        data_map = loader.fetch(src_codes, start_date, end_date, interval="1D")
        price_map.update(data_map)
    return price_map


def _resolve_factor_table(market: str, factor_name: str) -> tuple[Optional[str], list[str]]:
    """Find which fundamentals table has ``factor_name`` as an alias for ``market``.

    Returns (table_name_or_None, all_available_field_names_across_tables).
    """
    provider = _provider_for_market(market)
    available: list[str] = []
    for table in provider.list_tables():
        schema = provider.describe_table(table)
        if isinstance(schema, dict):
            fields = list(schema)
        else:
            # Tushare TableSchema.columns includes identity columns
            # (ts_code/ann_date/end_date) alongside real factor fields --
            # exclude the required ones so e.g. factor_name="ts_code"
            # can't be "resolved" into a nonsensical numeric factor.
            fields = [c.name for c in schema.columns if not c.required]
        available.extend(f"{table}.{f}" for f in fields)
        if factor_name in fields:
            return table, available
    return None, available


def _provider_for_market(market: str) -> Any:
    if market in ("tw_equity", "tw_futures"):
        from backtest.loaders.finlab_fundamentals import FinlabFundamentalProvider

        return FinlabFundamentalProvider()
    from backtest.loaders.tushare_fundamentals import TushareFundamentalProvider

    return TushareFundamentalProvider()


def _enrich_with_factor(
    market: str,
    price_map: dict[str, pd.DataFrame],
    table: str,
    factor_name: str,
    end_date: str,
) -> dict[str, pd.DataFrame]:
    if market in ("tw_equity", "tw_futures"):
        from backtest.loaders.finlab_fundamentals import (
            FinlabFundamentalProvider,
            enrich_price_frames_with_finlab_fundamentals,
        )

        provider = FinlabFundamentalProvider()
        return enrich_price_frames_with_finlab_fundamentals(
            price_map, provider, {table: [factor_name]},
        )

    from backtest.loaders.tushare_fundamentals import (
        TushareFundamentalProvider,
        enrich_price_frames_with_fundamentals,
    )

    provider = TushareFundamentalProvider()
    return enrich_price_frames_with_fundamentals(
        price_map, provider, {table: [factor_name]}, as_of=end_date,
    )


def _build_wide_tables(
    enriched: dict[str, pd.DataFrame], factor_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reshape {code: DataFrame(date-indexed, has close + factor_col)} into
    two wide tables (index=date, columns=codes): factor values, and next-bar
    forward returns computed from close."""
    factor_cols: dict[str, pd.Series] = {}
    return_cols: dict[str, pd.Series] = {}
    for code, frame in enriched.items():
        if factor_col not in frame.columns or "close" not in frame.columns:
            continue
        idx = pd.to_datetime(frame.index)
        factor_cols[code] = pd.Series(frame[factor_col].to_numpy(), index=idx)
        forward_return = frame["close"].pct_change().shift(-1)
        return_cols[code] = pd.Series(forward_return.to_numpy(), index=idx)

    factor_df = pd.DataFrame(factor_cols).sort_index() if factor_cols else pd.DataFrame()
    return_df = pd.DataFrame(return_cols).sort_index() if return_cols else pd.DataFrame()
    return factor_df, return_df


def _top_bottom_mean_returns(
    factor_df: pd.DataFrame, return_df: pd.DataFrame, top_n: int, bottom_n: int,
) -> tuple[Optional[float], Optional[float]]:
    """Rank codes by factor value each day, average the top-N and bottom-N
    codes' forward returns across all days."""
    common_dates = factor_df.index.intersection(return_df.index)
    top_returns: list[float] = []
    bottom_returns: list[float] = []
    for date in common_dates:
        f = factor_df.loc[date].dropna()
        r = return_df.loc[date].dropna()
        shared = f.index.intersection(r.index)
        if len(shared) < 2:
            continue
        ranked = f[shared].sort_values(ascending=False)
        top_codes = ranked.index[:top_n]
        bottom_codes = ranked.index[-bottom_n:]
        if len(top_codes) > 0:
            top_returns.append(float(r[top_codes].mean()))
        if len(bottom_codes) > 0:
            bottom_returns.append(float(r[bottom_codes].mean()))

    top_mean = sum(top_returns) / len(top_returns) if top_returns else None
    bottom_mean = sum(bottom_returns) / len(bottom_returns) if bottom_returns else None
    return top_mean, bottom_mean
