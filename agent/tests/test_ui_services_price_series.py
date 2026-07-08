"""OHLCV-artifact date-column recognition: _load_ohlcv_artifacts / _normalize_price_rows.

Regression coverage for a bug found via real-browser verification: finlab's
native data carries an index named "date" (confirmed against a live token),
which flows straight through backtest/engines/base.py's `df.to_csv()` (no
index_label override) into `ohlcv_{code}.csv`. yfinance_loader.py explicitly
normalizes its index to "trade_date", and Shioaji-sourced frames are unnamed
(empty CSV header) -- both already matched. "date" did not, so every row of
a finlab-sourced backtest's price chart was silently dropped (price_series
came back `{}`, RunDetail showed "Pick a symbol to load chart data" despite
a 200 response) even though the equity curve and trade log rendered fine.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.ui_services import _load_ohlcv_artifacts, _normalize_price_rows, build_indicator_series


def _write_ohlcv_csv(path: Path, header: str, rows: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(header + "\n" + "\n".join(rows) + "\n", encoding="utf-8")


class TestLoadOhlcvArtifactsDateColumnNames:
    def test_finlab_style_date_column_is_recognized(self, tmp_path: Path) -> None:
        _write_ohlcv_csv(
            tmp_path / "artifacts" / "ohlcv_2330.TW.csv",
            "date,open,high,low,close,volume",
            ["2024-01-02,600.0,605.0,595.0,602.0,1000"],
        )
        rows = _load_ohlcv_artifacts(tmp_path)
        assert len(rows) == 1
        assert rows[0]["code"] == "2330.TW"
        assert rows[0]["time"] == "2024-01-02"
        assert rows[0]["close"] == 602.0

    def test_yfinance_style_trade_date_column_still_works(self, tmp_path: Path) -> None:
        _write_ohlcv_csv(
            tmp_path / "artifacts" / "ohlcv_AAPL.US.csv",
            "trade_date,open,high,low,close,volume",
            ["2024-01-02,180.0,182.0,179.0,181.0,5000"],
        )
        rows = _load_ohlcv_artifacts(tmp_path)
        assert len(rows) == 1
        assert rows[0]["code"] == "AAPL.US"

    def test_shioaji_style_unnamed_index_column_still_works(self, tmp_path: Path) -> None:
        _write_ohlcv_csv(
            tmp_path / "artifacts" / "ohlcv_TXFR1.TWF.csv",
            ",open,high,low,close,volume",
            ["2024-01-02,17800.0,17900.0,17700.0,17850.0,120000"],
        )
        rows = _load_ohlcv_artifacts(tmp_path)
        assert len(rows) == 1
        assert rows[0]["code"] == "TXFR1.TWF"

    def test_timestamp_column_still_works(self, tmp_path: Path) -> None:
        _write_ohlcv_csv(
            tmp_path / "artifacts" / "ohlcv_000001.SZ.csv",
            "timestamp,open,high,low,close,volume",
            ["2024-01-02,10.0,10.5,9.8,10.2,80000"],
        )
        rows = _load_ohlcv_artifacts(tmp_path)
        assert len(rows) == 1

    def test_no_recognized_date_column_drops_rows_not_crashes(self, tmp_path: Path) -> None:
        _write_ohlcv_csv(
            tmp_path / "artifacts" / "ohlcv_X.csv",
            "totally_unknown_column,open,high,low,close,volume",
            ["2024-01-02,1.0,1.0,1.0,1.0,1"],
        )
        rows = _load_ohlcv_artifacts(tmp_path)
        assert rows == []

    def test_no_artifacts_dir_returns_empty(self, tmp_path: Path) -> None:
        assert _load_ohlcv_artifacts(tmp_path) == []


class TestNormalizePriceRowsDateColumn:
    def test_date_key_is_recognized(self) -> None:
        rows = _normalize_price_rows([
            {"date": "2024-01-02", "code": "2330.TW", "open": "600", "high": "605",
             "low": "595", "close": "602", "volume": "1000"},
        ])
        assert len(rows) == 1
        assert rows[0]["time"] == "2024-01-02"

    def test_timestamp_key_still_takes_priority(self) -> None:
        rows = _normalize_price_rows([
            {"timestamp": "2024-01-02", "date": "2024-01-01", "code": "X",
             "open": "1", "high": "1", "low": "1", "close": "1", "volume": "1"},
        ])
        assert rows[0]["time"] == "2024-01-02"


class TestIntradayTimestampsPreserveTimeOfDay:
    """Found live 2026-07-08 reviewing a 5m Granville strategy's chart: every
    bar's "time" was truncated to just its date, so multiple bars sharing a
    day collapsed onto one x-axis label. The frontend's indicator-overlay
    Map lookup (keyed by this exact string) then resolved every same-day bar
    to whichever bar's indicator value was written last for that key --
    a proper moving average degenerated into one flat value per day, only
    stepping at day boundaries. Daily-bar runs must keep their existing
    date-only "YYYY-MM-DD" format unchanged."""

    def test_intraday_bar_keeps_time_of_day(self) -> None:
        rows = _normalize_price_rows([
            {"timestamp": "2026-04-01 09:05:00", "code": "MXFR1.TWF",
             "open": "1", "high": "1", "low": "1", "close": "1", "volume": "1"},
        ])
        assert rows[0]["time"] == "2026-04-01 09:05:00"

    def test_daily_bar_still_gets_plain_date(self) -> None:
        rows = _normalize_price_rows([
            {"timestamp": "2026-04-01", "code": "2330.TW",
             "open": "1", "high": "1", "low": "1", "close": "1", "volume": "1"},
        ])
        assert rows[0]["time"] == "2026-04-01"

    def test_consecutive_intraday_bars_get_distinct_time_keys(self) -> None:
        """The actual bug: two 5m bars on the same day must not collide."""
        rows = _normalize_price_rows([
            {"timestamp": "2026-04-01 09:00:00", "code": "MXFR1.TWF",
             "open": "1", "high": "1", "low": "1", "close": "1", "volume": "1"},
            {"timestamp": "2026-04-01 09:05:00", "code": "MXFR1.TWF",
             "open": "1", "high": "1", "low": "1", "close": "2", "volume": "1"},
        ])
        assert {r["time"] for r in rows} == {"2026-04-01 09:00:00", "2026-04-01 09:05:00"}

    def test_load_ohlcv_artifacts_preserves_intraday_time(self, tmp_path: Path) -> None:
        _write_ohlcv_csv(
            tmp_path / "artifacts" / "ohlcv_MXFR1.TWF.csv",
            ",open,high,low,close,volume",
            [
                "2026-04-01 00:00:00,32846.2,32914.4,32833.0,32893.0,1315",
                "2026-04-01 00:05:00,32892.0,32923.6,32880.8,32918.5,730",
            ],
        )
        rows = _load_ohlcv_artifacts(tmp_path)
        assert {r["time"] for r in rows} == {"2026-04-01 00:00:00", "2026-04-01 00:05:00"}

    def test_indicator_series_points_keep_distinct_intraday_time_keys(self) -> None:
        """The frontend joins price bars to indicator overlay points by an
        exact "time" string match (Map-keyed lookup) -- if two bars in the
        same day shared a key here, one indicator value would silently stand
        in for both, which is exactly how the flat/day-stepped dashed MA
        line happened."""
        price_rows = [
            {"time": "2026-04-01 09:00:00", "code": "MXFR1.TWF", "close": 100.0},
            {"time": "2026-04-01 09:05:00", "code": "MXFR1.TWF", "close": 101.0},
            {"time": "2026-04-01 09:10:00", "code": "MXFR1.TWF", "close": 102.0},
        ]
        series = build_indicator_series(price_rows, periods=[2])
        times = [pt["time"] for pt in series["MXFR1.TWF"]["ma2"]]
        assert times == ["2026-04-01 09:00:00", "2026-04-01 09:05:00", "2026-04-01 09:10:00"]
        assert len(set(times)) == 3
