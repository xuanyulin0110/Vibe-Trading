"""Tests for the shared Shioaji K-bar helper (chunking, resampling, lock
cleanup) and the equity loader's lazy-login behavior.

The chunking/resampling tests exercise pure logic without needing real
Shioaji credentials/network access. The lazy-login test constructs a real
``DataLoader()`` but asserts it does NOT touch the network/SDK at construction
time -- see ``_ensure_logged_in``'s docstring for why eager login in
``__init__`` caused a deadlock (two near-simultaneous logins per backtest run
racing on the SDK's on-disk contract-cache lock files).
"""

from __future__ import annotations

import os
import time

import pandas as pd
import pytest

from backtest.loaders._shioaji_kbars import (
    _CHUNK_DAYS,
    clear_stale_shioaji_locks,
    date_chunks,
    is_supported_interval,
    normalize_interval,
    resample_kbars,
)
from backtest.loaders.shioaji_loader import DataLoader


class TestClearStaleShioajiLocks:
    def test_removes_old_lock_files(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setenv("SJ_HOME_PATH", str(tmp_path))
        old_lock = tmp_path / "contracts-1.5.4-STK-TW.parquet.lock"
        old_lock.touch()
        old_time = time.time() - 999
        os.utime(old_lock, (old_time, old_time))

        clear_stale_shioaji_locks(max_age_seconds=120.0)

        assert not old_lock.exists()

    def test_preserves_recent_lock_files(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setenv("SJ_HOME_PATH", str(tmp_path))
        recent_lock = tmp_path / "contracts-1.5.4-STK-TW.parquet.lock"
        recent_lock.touch()

        clear_stale_shioaji_locks(max_age_seconds=120.0)

        assert recent_lock.exists()

    def test_no_op_when_home_dir_missing(self, tmp_path, monkeypatch) -> None:
        missing = tmp_path / "does-not-exist"
        monkeypatch.setenv("SJ_HOME_PATH", str(missing))
        clear_stale_shioaji_locks()  # must not raise


class TestLazyLogin:
    def test_init_does_not_log_in(self) -> None:
        """Constructing a loader must not touch the network -- login is deferred to fetch()."""
        loader = DataLoader()
        assert loader.api is None


class TestNormalizeInterval:
    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("1D", "1d"), ("1d", "1d"),
            ("1H", "1h"), ("1h", "1h"),
            ("4H", "4h"),
            ("5m", "5m"), ("30m", "30m"), ("1m", "1m"),
        ],
    )
    def test_normalizes(self, raw: str, expected: str) -> None:
        assert normalize_interval(raw) == expected

    def test_supported_intervals(self) -> None:
        for token in ("1m", "5m", "15m", "30m", "1H", "4H", "1D", "1h", "1d"):
            assert is_supported_interval(token)
        assert not is_supported_interval("2m")
        assert not is_supported_interval("1w")


class TestDateChunks:
    def test_within_one_chunk(self) -> None:
        chunks = list(date_chunks("2024-01-01", "2024-01-10"))
        assert chunks == [("2024-01-01", "2024-01-10")]

    def test_exact_chunk_boundary(self) -> None:
        chunks = list(date_chunks("2024-01-01", "2024-01-29"))
        assert chunks == [("2024-01-01", "2024-01-29")]

    def test_spans_two_chunks(self) -> None:
        chunks = list(date_chunks("2024-01-01", "2024-01-30"))
        assert chunks == [
            ("2024-01-01", "2024-01-29"),
            ("2024-01-30", "2024-01-30"),
        ]

    def test_no_chunk_exceeds_29_days(self) -> None:
        for start, end in date_chunks("2020-01-01", "2025-12-31"):
            span_days = (pd.Timestamp(end) - pd.Timestamp(start)).days + 1
            assert span_days <= _CHUNK_DAYS

    def test_chunks_cover_full_range_contiguously(self) -> None:
        chunks = list(date_chunks("2024-01-01", "2024-03-15"))
        # Each chunk's end+1 day equals the next chunk's start -- no gaps, no overlap.
        for (_, end), (next_start, _) in zip(chunks, chunks[1:]):
            assert pd.Timestamp(end) + pd.Timedelta(days=1) == pd.Timestamp(next_start)
        assert chunks[0][0] == "2024-01-01"
        assert chunks[-1][1] == "2024-03-15"

    def test_single_day_range(self) -> None:
        chunks = list(date_chunks("2024-01-01", "2024-01-01"))
        assert chunks == [("2024-01-01", "2024-01-01")]


def _one_day_minute_frame() -> pd.DataFrame:
    """Six 1-minute bars across a single hour (09:00–09:05)."""
    idx = pd.to_datetime([
        "2024-01-02 09:00:00", "2024-01-02 09:01:00", "2024-01-02 09:02:00",
        "2024-01-02 09:03:00", "2024-01-02 09:04:00", "2024-01-02 09:05:00",
    ])
    return pd.DataFrame({
        "open":  [600.0, 601.0, 602.0, 603.0, 604.0, 605.0],
        "high":  [601.0, 602.0, 603.0, 604.0, 605.0, 606.0],
        "low":   [599.0, 600.0, 601.0, 602.0, 603.0, 604.0],
        "close": [600.5, 601.5, 602.5, 603.5, 604.5, 605.5],
        "volume": [100, 100, 100, 100, 100, 100],
    }, index=idx)


class TestResampleKbars:
    def test_1m_is_passthrough(self) -> None:
        minute_df = _one_day_minute_frame()
        out = resample_kbars(minute_df, "1m")
        assert out is minute_df  # no copy, no aggregation

    def test_5m_aggregates_first_bucket(self) -> None:
        minute_df = _one_day_minute_frame()
        out = resample_kbars(minute_df, "5m")
        # 09:00–09:04 form one 5-minute bucket; 09:05 starts the next.
        first = out.iloc[0]
        assert first["open"] == 600.0    # 09:00 open
        assert first["high"] == 605.0    # max high over 09:00–09:04
        assert first["low"] == 599.0     # min low over 09:00–09:04
        assert first["close"] == 604.5   # 09:04 close
        assert first["volume"] == 500    # five bars summed

    def test_1d_rolls_whole_day(self) -> None:
        minute_df = _one_day_minute_frame()
        out = resample_kbars(minute_df, "1D")
        assert len(out) == 1
        row = out.iloc[0]
        assert row["open"] == 600.0
        assert row["high"] == 606.0
        assert row["low"] == 599.0
        assert row["close"] == 605.5
        assert row["volume"] == 600

    @pytest.mark.parametrize("interval", ["1m", "5m", "15m", "30m", "1H", "4H", "1D"])
    def test_supported_intervals_do_not_raise(self, interval: str) -> None:
        resample_kbars(_one_day_minute_frame(), interval)

    def test_unsupported_interval_raises(self) -> None:
        with pytest.raises(ValueError, match="unsupported interval"):
            resample_kbars(_one_day_minute_frame(), "2m")

    def test_case_insensitive_day_and_hour(self) -> None:
        minute_df = _one_day_minute_frame()
        assert resample_kbars(minute_df, "1d").equals(resample_kbars(minute_df, "1D"))
        assert resample_kbars(minute_df, "1h").equals(resample_kbars(minute_df, "1H"))

    def test_drops_buckets_with_no_valid_ohlc(self) -> None:
        idx = pd.to_datetime(["2024-01-02 09:00:00"])
        minute_df = pd.DataFrame({
            "open": [None], "high": [None], "low": [None],
            "close": [None], "volume": [0],
        }, index=idx)
        assert resample_kbars(minute_df, "1D").empty
