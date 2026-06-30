"""Tests for shioaji_loader's date-chunking, minute-to-daily resampling, and
lazy-login behavior.

The chunking/resampling tests exercise pure logic without needing real
Shioaji credentials/network access. The lazy-login test constructs a real
``DataLoader()`` but asserts it does NOT touch the network/SDK at
construction time -- see ``_ensure_logged_in``'s docstring for why eager
login in ``__init__`` caused a deadlock (two near-simultaneous logins per
backtest run racing on the SDK's on-disk contract-cache lock files).
"""

from __future__ import annotations

import time

import pandas as pd

from backtest.loaders.shioaji_loader import (
    _CHUNK_DAYS,
    DataLoader,
    _clear_stale_shioaji_locks,
    _date_chunks,
    _resample_minute_kbars_to_daily,
)


class TestClearStaleShioajiLocks:
    def test_removes_old_lock_files(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setenv("SJ_HOME_PATH", str(tmp_path))
        old_lock = tmp_path / "contracts-1.5.4-STK-TW.parquet.lock"
        old_lock.touch()
        old_time = time.time() - 999
        import os
        os.utime(old_lock, (old_time, old_time))

        _clear_stale_shioaji_locks(max_age_seconds=120.0)

        assert not old_lock.exists()

    def test_preserves_recent_lock_files(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setenv("SJ_HOME_PATH", str(tmp_path))
        recent_lock = tmp_path / "contracts-1.5.4-STK-TW.parquet.lock"
        recent_lock.touch()

        _clear_stale_shioaji_locks(max_age_seconds=120.0)

        assert recent_lock.exists()

    def test_no_op_when_home_dir_missing(self, tmp_path, monkeypatch) -> None:
        missing = tmp_path / "does-not-exist"
        monkeypatch.setenv("SJ_HOME_PATH", str(missing))
        _clear_stale_shioaji_locks()  # must not raise


class TestLazyLogin:
    def test_init_does_not_log_in(self) -> None:
        """Constructing a loader must not touch the network -- login is deferred to fetch()."""
        loader = DataLoader()
        assert loader.api is None


class TestDateChunks:
    def test_within_one_chunk(self) -> None:
        chunks = list(_date_chunks("2024-01-01", "2024-01-10"))
        assert chunks == [("2024-01-01", "2024-01-10")]

    def test_exact_chunk_boundary(self) -> None:
        chunks = list(_date_chunks("2024-01-01", "2024-01-29"))
        assert chunks == [("2024-01-01", "2024-01-29")]

    def test_spans_two_chunks(self) -> None:
        chunks = list(_date_chunks("2024-01-01", "2024-01-30"))
        assert chunks == [
            ("2024-01-01", "2024-01-29"),
            ("2024-01-30", "2024-01-30"),
        ]

    def test_no_chunk_exceeds_29_days(self) -> None:
        for start, end in _date_chunks("2020-01-01", "2025-12-31"):
            span_days = (pd.Timestamp(end) - pd.Timestamp(start)).days + 1
            assert span_days <= _CHUNK_DAYS

    def test_chunks_cover_full_range_contiguously(self) -> None:
        chunks = list(_date_chunks("2024-01-01", "2024-03-15"))
        # Each chunk's end+1 day equals the next chunk's start -- no gaps, no overlap.
        for (_, end), (next_start, _) in zip(chunks, chunks[1:]):
            assert pd.Timestamp(end) + pd.Timedelta(days=1) == pd.Timestamp(next_start)
        assert chunks[0][0] == "2024-01-01"
        assert chunks[-1][1] == "2024-03-15"

    def test_single_day_range(self) -> None:
        chunks = list(_date_chunks("2024-01-01", "2024-01-01"))
        assert chunks == [("2024-01-01", "2024-01-01")]


class TestResampleMinuteKbarsToDaily:
    def test_aggregates_one_day(self) -> None:
        idx = pd.to_datetime([
            "2024-01-02 09:00:00", "2024-01-02 09:01:00",
            "2024-01-02 09:02:00", "2024-01-02 13:30:00",
        ])
        minute_df = pd.DataFrame({
            "open":  [600.0, 601.0, 602.0, 605.0],
            "high":  [601.0, 602.0, 603.0, 606.0],
            "low":   [599.0, 600.0, 601.0, 604.0],
            "close": [600.5, 601.5, 602.5, 605.5],
            "volume": [100, 200, 300, 400],
        }, index=idx)

        daily = _resample_minute_kbars_to_daily(minute_df)

        assert len(daily) == 1
        row = daily.iloc[0]
        assert row["open"] == 600.0     # first bar's open
        assert row["high"] == 606.0     # max across the day
        assert row["low"] == 599.0      # min across the day
        assert row["close"] == 605.5    # last bar's close
        assert row["volume"] == 1000    # summed

    def test_aggregates_multiple_days_separately(self) -> None:
        idx = pd.to_datetime([
            "2024-01-02 09:00:00", "2024-01-02 13:30:00",
            "2024-01-03 09:00:00", "2024-01-03 13:30:00",
        ])
        minute_df = pd.DataFrame({
            "open":  [600.0, 605.0, 610.0, 615.0],
            "high":  [606.0, 606.0, 616.0, 616.0],
            "low":   [599.0, 599.0, 609.0, 609.0],
            "close": [605.0, 605.5, 615.0, 615.5],
            "volume": [100, 100, 200, 200],
        }, index=idx)

        daily = _resample_minute_kbars_to_daily(minute_df)

        assert len(daily) == 2
        assert daily.index[0].date() == pd.Timestamp("2024-01-02").date()
        assert daily.index[1].date() == pd.Timestamp("2024-01-03").date()
        assert daily.iloc[0]["volume"] == 200
        assert daily.iloc[1]["volume"] == 400

    def test_drops_days_with_no_valid_ohlc(self) -> None:
        idx = pd.to_datetime(["2024-01-02 09:00:00"])
        minute_df = pd.DataFrame({
            "open": [None], "high": [None], "low": [None],
            "close": [None], "volume": [0],
        }, index=idx)

        daily = _resample_minute_kbars_to_daily(minute_df)
        assert daily.empty
