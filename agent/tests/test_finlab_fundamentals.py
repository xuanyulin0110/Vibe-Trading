from __future__ import annotations

import pandas as pd
import pytest

from backtest.loaders.finlab_fundamentals import (
    FinlabFundamentalProvider,
    enrich_price_frames_with_finlab_fundamentals,
)


def _provider_with_canned_tables(**field_key_to_wide_table: pd.DataFrame) -> FinlabFundamentalProvider:
    """Build a provider whose ``_field_table`` cache is pre-populated, bypassing the live finlab import."""
    provider = FinlabFundamentalProvider()
    provider._field_cache.update(field_key_to_wide_table)
    return provider


class TestProviderMetadata:
    def test_list_tables(self) -> None:
        provider = FinlabFundamentalProvider()
        assert provider.list_tables() == ["institutional", "margin", "monthly_revenue"]

    def test_describe_table_returns_alias_map(self) -> None:
        provider = FinlabFundamentalProvider()
        schema = provider.describe_table("monthly_revenue")
        assert schema["revenue"] == "monthly_revenue:當月營收"

    def test_describe_unknown_table_raises(self) -> None:
        provider = FinlabFundamentalProvider()
        with pytest.raises(ValueError, match="Unsupported finlab fundamental table"):
            provider.describe_table("nope")


class TestQueryFundamentals:
    def test_slices_per_code_from_wide_table(self) -> None:
        dates = pd.to_datetime(["2026-06-10", "2026-07-13"])
        wide = pd.DataFrame({"2330": [1000.0, 1100.0], "2317": [500.0, 550.0]}, index=dates)
        provider = _provider_with_canned_tables(**{"monthly_revenue:當月營收": wide})

        result = provider.query_fundamentals(
            "monthly_revenue", ["2330.TW", "2317.TW"], fields=["revenue"],
        )

        assert list(result["2330.TW"]["revenue"]) == [1000.0, 1100.0]
        assert list(result["2317.TW"]["revenue"]) == [500.0, 550.0]

    def test_omits_codes_with_no_data(self) -> None:
        dates = pd.to_datetime(["2026-06-10"])
        wide = pd.DataFrame({"2330": [1000.0]}, index=dates)
        provider = _provider_with_canned_tables(**{"monthly_revenue:當月營收": wide})

        result = provider.query_fundamentals("monthly_revenue", ["9999.TW"], fields=["revenue"])
        assert "9999.TW" not in result

    def test_unknown_field_raises(self) -> None:
        provider = FinlabFundamentalProvider()
        with pytest.raises(ValueError, match="Unknown fields"):
            provider.query_fundamentals("monthly_revenue", ["2330.TW"], fields=["not_a_field"])

    def test_defaults_to_all_fields(self) -> None:
        dates = pd.to_datetime(["2026-06-10"])
        wide = pd.DataFrame({"2330": [1000.0]}, index=dates)
        provider = _provider_with_canned_tables(**{
            "monthly_revenue:當月營收": wide,
            "monthly_revenue:去年同月增減(%)": pd.DataFrame({"2330": [5.0]}, index=dates),
            "monthly_revenue:上月比較增減(%)": pd.DataFrame({"2330": [2.0]}, index=dates),
        })

        result = provider.query_fundamentals("monthly_revenue", ["2330.TW"])
        assert set(result["2330.TW"].columns) == {"revenue", "revenue_yoy_pct", "revenue_mom_pct"}


class TestEnrichPriceFramesWithFinlabFundamentals:
    def test_point_in_time_merge(self) -> None:
        """finlab's monthly_revenue index IS the announcement date -- a backward
        as-of merge means a price bar only sees revenue announced on or before it."""
        revenue_dates = pd.to_datetime(["2026-05-11", "2026-06-10"])
        wide = pd.DataFrame({"2330": [50_000.0, 55_000.0]}, index=revenue_dates)
        provider = _provider_with_canned_tables(**{"monthly_revenue:當月營收": wide})

        price_dates = pd.to_datetime(["2026-05-05", "2026-05-15", "2026-06-15"])
        bars = pd.DataFrame(
            {
                "open": [600.0, 610.0, 620.0],
                "high": [605.0, 615.0, 625.0],
                "low": [595.0, 605.0, 615.0],
                "close": [602.0, 612.0, 622.0],
                "volume": [1000, 1100, 1200],
            },
            index=price_dates,
        )

        enriched = enrich_price_frames_with_finlab_fundamentals(
            {"2330.TW": bars}, provider, {"monthly_revenue": ["revenue"]},
        )

        result = enriched["2330.TW"]
        # Before any announcement: NaN (no look-ahead).
        assert pd.isna(result.loc[pd.Timestamp("2026-05-05"), "monthly_revenue_revenue"])
        # On/after the 2026-05-11 announcement, sees that revenue.
        assert result.loc[pd.Timestamp("2026-05-15"), "monthly_revenue_revenue"] == 50_000.0
        # On/after the 2026-06-10 announcement, sees the updated revenue.
        assert result.loc[pd.Timestamp("2026-06-15"), "monthly_revenue_revenue"] == 55_000.0

    def test_column_prefixed_with_table_name(self) -> None:
        dates = pd.to_datetime(["2026-06-01"])
        wide = pd.DataFrame({"2330": [100_000.0]}, index=dates)
        provider = _provider_with_canned_tables(**{
            "institutional_investors_trading_summary:投信買賣超股數": wide,
        })

        bars = pd.DataFrame(
            {"open": [600.0], "high": [605.0], "low": [595.0], "close": [602.0], "volume": [1000]},
            index=dates,
        )
        enriched = enrich_price_frames_with_finlab_fundamentals(
            {"2330.TW": bars}, provider, {"institutional": ["trust_net"]},
        )
        assert "institutional_trust_net" in enriched["2330.TW"].columns

    def test_no_op_when_no_fields_requested(self) -> None:
        provider = FinlabFundamentalProvider()
        data_map = {"2330.TW": pd.DataFrame({"close": [600.0]})}
        result = enrich_price_frames_with_finlab_fundamentals(data_map, provider, {})
        assert result is data_map
