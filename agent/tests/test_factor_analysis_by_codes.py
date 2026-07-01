from __future__ import annotations

import json
import sys
from typing import Dict

import pandas as pd
import pytest

from src.tools.factor_analysis_by_codes import (
    _build_wide_tables,
    _resolve_factor_table,
    _top_bottom_mean_returns,
    run_factor_analysis_by_codes,
)


def _install_fake_finlab(monkeypatch: pytest.MonkeyPatch, tables: Dict[str, pd.DataFrame]) -> None:
    """Install a fake ``finlab`` module so FinlabFundamentalProvider (constructed
    fresh inside the adapter) logs in and reads canned data instead of hitting
    the network. A real-looking token is required: _ensure_logged_in() raises
    on an empty/placeholder FINLAB_API_TOKEN (see finlab_fundamentals.py)."""

    class _FakeData:
        @staticmethod
        def get(field_key: str) -> pd.DataFrame:
            return tables[field_key]

    class _FakeFinlabModule:
        data = _FakeData()

        @staticmethod
        def login(token: str) -> None:
            pass

    monkeypatch.setenv("FINLAB_API_TOKEN", "fake-token-for-tests")
    monkeypatch.setitem(sys.modules, "finlab", _FakeFinlabModule)
    monkeypatch.setitem(sys.modules, "finlab.data", _FakeData)


def _install_fake_tushare(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set a real-looking TUSHARE_TOKEN and stub ts.pro_api so
    TushareFundamentalProvider() constructs cleanly without network access."""
    monkeypatch.setenv("TUSHARE_TOKEN", "fake-token-for-tests")
    fake_module = type(sys)("tushare")
    fake_module.pro_api = lambda token="": object()
    monkeypatch.setitem(sys.modules, "tushare", fake_module)


class TestResolveFactorTable:
    def test_finds_alias_in_finlab_table(self) -> None:
        table, available = _resolve_factor_table("tw_equity", "roe")
        assert table == "fundamental_features"
        assert "fundamental_features.roe" in available

    def test_unknown_alias_returns_none_with_full_field_list(self) -> None:
        table, available = _resolve_factor_table("tw_equity", "not_a_real_field")
        assert table is None
        assert len(available) > 20  # every finlab table's fields, flattened

    def test_tushare_excludes_identity_columns(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_fake_tushare(monkeypatch)
        table, available = _resolve_factor_table("a_share", "roe")
        assert table == "fina_indicator"
        assert "fina_indicator.ts_code" not in available
        assert "fina_indicator.ann_date" not in available
        assert "fina_indicator.roe" in available

    def test_tushare_identity_column_name_is_not_resolvable_as_factor(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _install_fake_tushare(monkeypatch)
        table, _ = _resolve_factor_table("a_share", "ts_code")
        assert table is None


class TestBuildWideTables:
    def test_factor_and_forward_return_alignment(self) -> None:
        dates = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"])
        frame = pd.DataFrame(
            {
                "close": [100.0, 110.0, 121.0],
                "fundamental_features_roe": [5.0, 5.0, 6.0],
            },
            index=dates,
        )
        factor_df, return_df = _build_wide_tables(
            {"2330.TW": frame}, "fundamental_features_roe",
        )
        assert list(factor_df["2330.TW"]) == [5.0, 5.0, 6.0]
        # forward return = pct_change().shift(-1): day1->day2 = 10%, day2->day3 = 10%, last is NaN
        assert return_df["2330.TW"].iloc[0] == pytest.approx(0.10)
        assert return_df["2330.TW"].iloc[1] == pytest.approx(0.10)
        assert pd.isna(return_df["2330.TW"].iloc[2])

    def test_skips_codes_missing_factor_column(self) -> None:
        dates = pd.to_datetime(["2024-01-01"])
        frame = pd.DataFrame({"close": [100.0]}, index=dates)
        factor_df, _ = _build_wide_tables({"2330.TW": frame}, "fundamental_features_roe")
        assert "2330.TW" not in factor_df.columns


class TestTopBottomMeanReturns:
    def test_ranks_and_averages_extremes(self) -> None:
        dates = pd.to_datetime(["2024-01-01", "2024-01-02"])
        factor_df = pd.DataFrame(
            {"A": [1.0, 1.0], "B": [2.0, 2.0], "C": [3.0, 3.0], "D": [4.0, 4.0]}, index=dates,
        )
        return_df = pd.DataFrame(
            {"A": [0.01, 0.01], "B": [0.02, 0.02], "C": [0.03, 0.03], "D": [0.04, 0.04]}, index=dates,
        )
        top_mean, bottom_mean = _top_bottom_mean_returns(factor_df, return_df, top_n=1, bottom_n=1)
        assert top_mean == pytest.approx(0.04)  # D has the highest factor value
        assert bottom_mean == pytest.approx(0.01)  # A has the lowest

    def test_returns_none_when_no_overlapping_dates(self) -> None:
        factor_df = pd.DataFrame({"A": [1.0]}, index=pd.to_datetime(["2024-01-01"]))
        return_df = pd.DataFrame({"A": [0.01]}, index=pd.to_datetime(["2024-06-01"]))
        top_mean, bottom_mean = _top_bottom_mean_returns(factor_df, return_df, top_n=1, bottom_n=1)
        assert top_mean is None
        assert bottom_mean is None


class TestRunFactorAnalysisByCodes:
    def test_end_to_end_with_canned_finlab_data(self, monkeypatch: pytest.MonkeyPatch) -> None:
        dates = pd.to_datetime(pd.date_range("2024-01-01", periods=20, freq="D"))
        rng_close = {
            "2330.TW": [600 + i for i in range(20)],
            "2317.TW": [100 - i * 0.5 for i in range(20)],
            "2454.TW": [800 + (i % 3) for i in range(20)],
            "2412.TW": [120 + i * 0.2 for i in range(20)],
            "1301.TW": [90 + (i % 2) for i in range(20)],
        }

        class _FakeLoader:
            def fetch(self, codes, start_date, end_date, interval="1D"):
                return {
                    code: pd.DataFrame(
                        {"open": rng_close[code], "high": rng_close[code],
                         "low": rng_close[code], "close": rng_close[code],
                         "volume": [1000] * 20},
                        index=dates,
                    )
                    for code in codes
                }

        monkeypatch.setattr(
            "src.tools.factor_analysis_by_codes.get_loader", lambda source: _FakeLoader,
        )
        monkeypatch.setattr(
            "src.tools.factor_analysis_by_codes.detect_source", lambda code: "finlab",
        )

        # finlab wide tables are columned by bare stock id (no .TW suffix). Uses
        # "institutional" (already date-indexed) rather than the quarter-indexed
        # "fundamental_features" table -- the quarter -> disclosure-date
        # resolution path has its own dedicated coverage in
        # test_finlab_fundamentals.py; this test is about the adapter, not that.
        # Varies by both code and day -- compute_ic_series needs cross-sectional
        # variance in the factor (identical values across codes give zero-variance
        # ranks, so Pearson-on-ranks IC is NaN for that day).
        foreign_net = pd.DataFrame(
            {
                code.split(".")[0]: [10.0 + day + code_idx * 5 for day in range(20)]
                for code_idx, code in enumerate(rng_close)
            },
            index=dates,
        )
        _install_fake_finlab(
            monkeypatch,
            {"institutional_investors_trading_summary:外資自營商買賣超股數": foreign_net},
        )

        result = json.loads(run_factor_analysis_by_codes(
            codes=list(rng_close), factor_name="foreign_net",
            start_date="2024-01-01", end_date="2024-01-20",
            source="auto", top_n=2, bottom_n=2,
        ))

        assert result["status"] == "ok"
        assert result["factor_name"] == "foreign_net"
        assert result["ic_count"] > 0
        assert result["top_mean_return"] is not None
        assert result["bottom_mean_return"] is not None

    def test_empty_codes_returns_clean_error(self) -> None:
        result = json.loads(run_factor_analysis_by_codes(
            codes=[], factor_name="roe", start_date="2024-01-01", end_date="2024-01-20",
        ))
        assert result["status"] == "error"
        assert "non-empty" in result["error"]

    def test_unresolvable_factor_name_returns_available_fields(self, monkeypatch: pytest.MonkeyPatch) -> None:
        dates = pd.to_datetime(["2024-01-01", "2024-01-02"])

        class _FakeLoader:
            def fetch(self, codes, start_date, end_date, interval="1D"):
                return {
                    code: pd.DataFrame(
                        {"open": [1, 1], "high": [1, 1], "low": [1, 1], "close": [1, 2], "volume": [1, 1]},
                        index=dates,
                    )
                    for code in codes
                }

        monkeypatch.setattr(
            "src.tools.factor_analysis_by_codes.get_loader", lambda source: _FakeLoader,
        )
        monkeypatch.setattr(
            "src.tools.factor_analysis_by_codes.detect_source", lambda code: "finlab",
        )

        result = json.loads(run_factor_analysis_by_codes(
            codes=["2330.TW"], factor_name="totally_bogus_field",
            start_date="2024-01-01", end_date="2024-01-02",
        ))
        assert result["status"] == "error"
        assert "not found" in result["error"]
        assert "available_fields" in result
        assert len(result["available_fields"]) > 0
