from __future__ import annotations

import sys

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


class TestLazyLogin:
    """FinlabFundamentalProvider has no login of its own by default -- it used
    to assume some other code path (typically FinlabLoader) had already
    called finlab.login() in-process. That's false when Shioaji is the price
    source and this provider is the only finlab touch-point (see class
    docstring in finlab_fundamentals.py)."""

    def test_init_does_not_log_in(self) -> None:
        provider = FinlabFundamentalProvider()
        assert provider._logged_in is False

    def test_cache_hit_never_needs_login(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A pre-populated cache (this test file's usual pattern) must not
        require a token at all -- _ensure_logged_in() only runs on a cache miss."""
        monkeypatch.delenv("FINLAB_API_TOKEN", raising=False)
        wide = pd.DataFrame({"2330": [1.0]}, index=pd.to_datetime(["2026-06-10"]))
        provider = _provider_with_canned_tables(**{"monthly_revenue:當月營收": wide})

        result = provider.query_fundamentals("monthly_revenue", ["2330.TW"], fields=["revenue"])
        assert result["2330.TW"]["revenue"].iloc[0] == 1.0

    @pytest.mark.parametrize("token", ["", "your-finlab-token"])
    def test_cache_miss_with_placeholder_token_raises_instead_of_calling_finlab(
        self, monkeypatch: pytest.MonkeyPatch, token: str,
    ) -> None:
        monkeypatch.setenv("FINLAB_API_TOKEN", token)
        provider = FinlabFundamentalProvider()
        with pytest.raises(RuntimeError, match="FINLAB_API_TOKEN is not configured"):
            provider._field_table("monthly_revenue:當月營收")
        assert provider._logged_in is False

    def test_cache_miss_with_real_token_logs_in_once(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls: list[str] = []

        class _FakeData:
            @staticmethod
            def get(field_key: str) -> pd.DataFrame:
                return pd.DataFrame({"2330": [1.0]})

        class _FakeFinlabModule:
            data = _FakeData()

            @staticmethod
            def login(token: str) -> None:
                calls.append(token)

        monkeypatch.setenv("FINLAB_API_TOKEN", "real-token")
        monkeypatch.setitem(sys.modules, "finlab", _FakeFinlabModule)
        monkeypatch.setitem(sys.modules, "finlab.data", _FakeData)

        provider = FinlabFundamentalProvider()
        provider._field_table("monthly_revenue:當月營收")
        provider._field_table("margin_transactions:融資今日餘額")  # second, different field

        assert calls == ["real-token"]  # login happens once, not once per field
        assert provider._logged_in is True


class TestFieldTableDropsNaTIndexRows:
    """Found live 2026-07-09: a real futures_institutional_investors_trading_
    summary table fetch crashed enrich_price_frames_with_finlab_fundamentals
    with "ValueError: Merge keys contain null values on right side" --
    confirmed against the actual finlab data that exactly one row of
    finlab's own wide table has a NaT index (a genuine upstream data-quality
    glitch, not something this loader introduces). pd.merge_asof hard-
    requires non-null merge keys, so this crashed for every code and field
    selected from the table, at every interval -- discovered via a 5m
    futures backtest, but not actually specific to 5m vs daily."""

    def test_nat_indexed_row_is_dropped_from_cached_table(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class _FakeData:
            @staticmethod
            def get(field_key: str) -> pd.DataFrame:
                return pd.DataFrame(
                    {"TXF": [100.0, 200.0, 300.0]},
                    index=pd.DatetimeIndex([pd.NaT, pd.Timestamp("2026-07-08"), pd.Timestamp("2026-07-09")]),
                )

        class _FakeFinlabModule:
            data = _FakeData()

            @staticmethod
            def login(token: str) -> None:
                pass

        monkeypatch.setenv("FINLAB_API_TOKEN", "real-token")
        monkeypatch.setitem(sys.modules, "finlab", _FakeFinlabModule)
        monkeypatch.setitem(sys.modules, "finlab.data", _FakeData)

        provider = FinlabFundamentalProvider()
        table = provider._field_table("some_field_key")

        assert len(table) == 2
        assert table.index.isna().sum() == 0

    def test_enrichment_no_longer_crashes_with_a_nat_row_in_the_table(self) -> None:
        wide = pd.DataFrame(
            {"臺股期貨_外資及陸資": [100.0, 200.0]},
            index=pd.DatetimeIndex([pd.NaT, pd.Timestamp("2026-04-01")]),
        )
        # _field_table() would have dropped the NaT row on fetch; simulate
        # that directly since this provider is cache-pre-populated in tests.
        provider = _provider_with_canned_tables(**{
            "futures_institutional_investors_trading_summary:多空未平倉口數淨額": wide[wide.index.notna()],
        })
        price = pd.DataFrame(
            {"open": [1.0, 1.0], "high": [1.0, 1.0], "low": [1.0, 1.0], "close": [1.0, 1.0], "volume": [1.0, 1.0]},
            index=pd.date_range("2026-04-01 00:00:00", periods=2, freq="5min"),
        )
        enriched = enrich_price_frames_with_finlab_fundamentals(
            {"TXFR1.TWF": price}, provider, {"futures_institutional": ["foreign_net_oi"]},
        )
        assert enriched["TXFR1.TWF"]["futures_institutional_foreign_net_oi"].iloc[0] == 200.0


class TestProviderMetadata:
    def test_list_tables(self) -> None:
        provider = FinlabFundamentalProvider()
        assert provider.list_tables() == [
            "director_shareholding",
            "financial_statement",
            "foreign_shareholding",
            "fundamental_features",
            "futures_institutional",
            "institutional",
            "margin",
            "monthly_revenue",
            "rotc_monthly_revenue",
        ]

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


class TestQuarterIndexedTables:
    """financial_statement / fundamental_features are indexed by quarter-period
    label ('2025-Q1'), not a date -- query_fundamentals must resolve them
    through the etl:financial_statements_disclosure_dates companion table."""

    def test_resolves_quarter_label_to_disclosure_date(self) -> None:
        quarters = ["2024-Q4", "2025-Q1", "2025-Q2"]
        assets = pd.DataFrame({"2330": [100.0, 110.0, 120.0]}, index=quarters)
        disclosure = pd.DataFrame(
            {"2330": pd.to_datetime(["2025-02-27", "2025-05-15", "2025-08-14"])},
            index=quarters,
        )
        provider = _provider_with_canned_tables(**{
            "financial_statement:資產總額": assets,
            "etl:financial_statements_disclosure_dates": disclosure,
        })

        result = provider.query_fundamentals(
            "financial_statement", ["2330.TW"], fields=["total_assets"],
        )

        series = result["2330.TW"]["total_assets"]
        assert list(series.index) == list(pd.to_datetime(["2025-02-27", "2025-05-15", "2025-08-14"]))
        assert list(series) == [100.0, 110.0, 120.0]

    def test_drops_quarters_with_no_known_disclosure_date(self) -> None:
        quarters = ["2024-Q4", "2025-Q1"]
        assets = pd.DataFrame({"2330": [100.0, 110.0]}, index=quarters)
        disclosure = pd.DataFrame(
            {"2330": [pd.NaT, pd.Timestamp("2025-05-15")]}, index=quarters,
        )
        provider = _provider_with_canned_tables(**{
            "financial_statement:資產總額": assets,
            "etl:financial_statements_disclosure_dates": disclosure,
        })

        result = provider.query_fundamentals(
            "financial_statement", ["2330.TW"], fields=["total_assets"],
        )

        series = result["2330.TW"]["total_assets"]
        assert list(series) == [110.0]

    def test_disclosure_table_may_have_different_quarter_range(self) -> None:
        """The disclosure-date table has deeper history than any one field --
        alignment must be by quarter label, not position."""
        field_quarters = ["2025-Q1", "2025-Q2"]
        assets = pd.DataFrame({"2330": [110.0, 120.0]}, index=field_quarters)
        disclosure_quarters = ["2024-Q2", "2024-Q3", "2025-Q1", "2025-Q2"]
        disclosure = pd.DataFrame(
            {"2330": pd.to_datetime(["2024-08-14", "2024-11-14", "2025-05-15", "2025-08-14"])},
            index=disclosure_quarters,
        )
        provider = _provider_with_canned_tables(**{
            "financial_statement:資產總額": assets,
            "etl:financial_statements_disclosure_dates": disclosure,
        })

        result = provider.query_fundamentals(
            "financial_statement", ["2330.TW"], fields=["total_assets"],
        )

        series = result["2330.TW"]["total_assets"]
        assert list(series) == [110.0, 120.0]
        assert list(series.index) == list(pd.to_datetime(["2025-05-15", "2025-08-14"]))

    def test_end_to_end_pit_merge_uses_disclosure_date_not_quarter_end(self) -> None:
        quarters = ["2025-Q1"]
        assets = pd.DataFrame({"2330": [110.0]}, index=quarters)
        disclosure = pd.DataFrame({"2330": pd.to_datetime(["2025-05-15"])}, index=quarters)
        provider = _provider_with_canned_tables(**{
            "financial_statement:資產總額": assets,
            "etl:financial_statements_disclosure_dates": disclosure,
        })

        price_dates = pd.to_datetime(["2025-03-31", "2025-05-20"])
        bars = pd.DataFrame(
            {
                "open": [600.0, 610.0], "high": [605.0, 615.0],
                "low": [595.0, 605.0], "close": [602.0, 612.0], "volume": [1000, 1100],
            },
            index=price_dates,
        )

        enriched = enrich_price_frames_with_finlab_fundamentals(
            {"2330.TW": bars}, provider, {"financial_statement": ["total_assets"]},
        )

        result = enriched["2330.TW"]
        # Quarter-end (2025-03-31) is BEFORE the real 2025-05-15 disclosure: no look-ahead.
        assert pd.isna(result.loc[pd.Timestamp("2025-03-31"), "financial_statement_total_assets"])
        # After the real disclosure date, the figure is visible.
        assert result.loc[pd.Timestamp("2025-05-20"), "financial_statement_total_assets"] == 110.0


class TestFuturesInstitutional:
    """futures_institutional_investors_trading_summary uses TAIFEX product-name
    columns ("臺股期貨_外資及陸資"), not stock codes -- codes here are .TWF symbols."""

    def test_resolves_product_prefix_and_investor_type(self) -> None:
        dates = pd.to_datetime(["2026-06-30"])
        oi = pd.DataFrame(
            {"臺股期貨_外資及陸資": [1234.0], "小型臺指期貨_外資及陸資": [99.0]}, index=dates,
        )
        provider = _provider_with_canned_tables(**{
            "futures_institutional_investors_trading_summary:多空未平倉口數淨額": oi,
        })

        result = provider.query_fundamentals(
            "futures_institutional", ["TXFR1.TWF", "MXFR1.TWF"], fields=["foreign_net_oi"],
        )

        assert result["TXFR1.TWF"]["foreign_net_oi"].iloc[0] == 1234.0
        assert result["MXFR1.TWF"]["foreign_net_oi"].iloc[0] == 99.0

    def test_unknown_product_omitted(self) -> None:
        dates = pd.to_datetime(["2026-06-30"])
        oi = pd.DataFrame({"臺股期貨_外資及陸資": [1234.0]}, index=dates)
        provider = _provider_with_canned_tables(**{
            "futures_institutional_investors_trading_summary:多空未平倉口數淨額": oi,
        })

        result = provider.query_fundamentals(
            "futures_institutional", ["ZZZR1.TWF"], fields=["foreign_net_oi"],
        )
        assert "ZZZR1.TWF" not in result

    def test_unknown_field_raises(self) -> None:
        provider = FinlabFundamentalProvider()
        with pytest.raises(ValueError, match="Unknown fields"):
            provider.query_fundamentals("futures_institutional", ["TXFR1.TWF"], fields=["nope"])

    def test_defaults_to_all_fields(self) -> None:
        dates = pd.to_datetime(["2026-06-30"])
        provider = _provider_with_canned_tables(**{
            "futures_institutional_investors_trading_summary:多空未平倉口數淨額": pd.DataFrame(
                {"臺股期貨_外資及陸資": [1.0], "臺股期貨_投信": [2.0], "臺股期貨_自營商": [3.0]}, index=dates,
            ),
            "futures_institutional_investors_trading_summary:多空交易口數淨額": pd.DataFrame(
                {"臺股期貨_外資及陸資": [4.0], "臺股期貨_投信": [5.0], "臺股期貨_自營商": [6.0]}, index=dates,
            ),
        })

        result = provider.query_fundamentals("futures_institutional", ["TXFR1.TWF"])
        assert set(result["TXFR1.TWF"].columns) == {
            "foreign_net_oi", "trust_net_oi", "dealer_net_oi",
            "foreign_net_volume", "trust_net_volume", "dealer_net_volume",
        }


class TestTableConsistency:
    def test_all_table_fields_keys_are_valid_finlab_dataset_prefixes(self) -> None:
        """Every _TABLE_FIELDS field-key string (barring the display-only
        futures_institutional entries) must look like 'dataset:column'."""
        from backtest.loaders.finlab_fundamentals import _TABLE_FIELDS

        for table, fields in _TABLE_FIELDS.items():
            if table == "futures_institutional":
                continue
            for alias, key in fields.items():
                assert ":" in key, f"{table}.{alias} field key {key!r} missing ':' separator"

    def test_futures_institutional_display_and_query_aliases_match(self) -> None:
        from backtest.loaders.finlab_fundamentals import (
            _FUTURES_INSTITUTIONAL_FIELDS,
            _TABLE_FIELDS,
        )

        assert set(_TABLE_FIELDS["futures_institutional"]) == set(_FUTURES_INSTITUTIONAL_FIELDS)
