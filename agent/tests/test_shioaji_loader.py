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


def _bar(ts: str, o: float, h: float, low: float, c: float, v: int) -> dict:
    return {"ts": pd.Timestamp(ts), "open": o, "high": h, "low": low, "close": c, "volume": v}


def _taifex_frame(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows).set_index("ts")
    df.index.name = None
    return df


class TestSessionAwareResample:
    """TAIFEX futures: a trading day's bar = the night session that runs into
    its morning (Day D-1 15:00 -> Day D 05:00) plus Day D's own day session
    (08:45-13:45). Confirmed live (2026-07-02) that the naive calendar-day
    resample instead lets Day D's *own* night session start (15:00-23:59,
    which belongs to Day D+1) leak into Day D's bar -- these tests pin the
    fix against hand-built minute bars with known expected values.
    """

    def test_night_session_start_belongs_to_next_trading_day(self) -> None:
        # Monday's night session (15:00) is the start of Tuesday's trading day.
        rows = [
            _bar("2024-01-08 15:00:00", 100, 105, 99, 102, 10),   # Mon night start -> Tue
            _bar("2024-01-08 15:01:00", 102, 106, 101, 104, 10),
            _bar("2024-01-09 08:45:00", 110, 112, 108, 111, 10),  # Tue day session
            _bar("2024-01-09 13:45:00", 111, 113, 109, 112, 10),  # Tue day session close
        ]
        minute_df = _taifex_frame(rows)
        naive = resample_kbars(minute_df, "1D", session_aware=False)
        aware = resample_kbars(minute_df, "1D", session_aware=True)

        # Naive: Monday's bar wrongly captures its own night-session start.
        assert naive.loc["2024-01-08", "close"] == 104
        # Aware: that same 15:00 bar is folded into Tuesday's bar instead,
        # and Tuesday's close is the real day-session close (13:45), not
        # contaminated by anything after it.
        assert "2024-01-08" not in aware.index.strftime("%Y-%m-%d")
        tue = aware.loc["2024-01-09"]
        assert tue["open"] == 100     # Monday night session's 15:00 open
        assert tue["close"] == 112    # Tuesday's real 13:45 day-session close
        assert tue["volume"] == 40

    def test_night_session_tail_before_5am_joins_same_trading_day(self) -> None:
        # The 00:00-05:00 continuation of Monday evening's session is still
        # part of Tuesday's trading day, consistent with the 15:00 start.
        rows = [
            _bar("2024-01-08 15:00:00", 100, 100, 100, 100, 5),
            _bar("2024-01-09 04:59:00", 120, 121, 119, 120, 5),   # night tail, still Tue
            _bar("2024-01-09 08:45:00", 121, 122, 120, 121, 5),
            _bar("2024-01-09 13:45:00", 121, 123, 120, 122, 5),
        ]
        minute_df = _taifex_frame(rows)
        aware = resample_kbars(minute_df, "1D", session_aware=True)

        assert len(aware) == 1
        tue = aware.loc["2024-01-09"]
        assert tue["open"] == 100
        assert tue["close"] == 122
        assert tue["volume"] == 20

    def test_weekend_gap_uses_actual_data_not_hardcoded_calendar(self) -> None:
        # Friday's night session belongs to the next real trading day found
        # in the data (Monday), correctly skipping Sat/Sun with no external
        # holiday calendar needed.
        rows = [
            _bar("2024-01-05 15:00:00", 200, 201, 199, 200, 5),  # Fri night -> Mon
            _bar("2024-01-08 08:45:00", 205, 206, 204, 205, 5),  # Mon day session
            _bar("2024-01-08 13:45:00", 205, 207, 204, 206, 5),
        ]
        minute_df = _taifex_frame(rows)
        aware = resample_kbars(minute_df, "1D", session_aware=True)

        assert len(aware) == 1
        mon = aware.loc["2024-01-08"]
        assert mon["open"] == 200   # Friday night session's open carried in
        assert mon["close"] == 206  # Monday's real day-session close

    def test_session_aware_defaults_off(self) -> None:
        """Equities have no night session -- callers that don't pass
        session_aware must keep getting the plain calendar-day behavior."""
        rows = [
            _bar("2024-01-08 15:00:00", 100, 105, 99, 102, 10),
            _bar("2024-01-09 08:45:00", 110, 112, 108, 111, 10),
        ]
        minute_df = _taifex_frame(rows)
        assert resample_kbars(minute_df, "1D").equals(resample_kbars(minute_df, "1D", session_aware=False))

    def test_friday_night_session_spanning_saturday_lands_on_monday(self) -> None:
        # Friday 15:00 -> Saturday 05:00 is ONE overnight session belonging to
        # Monday. Includes the 05:01-stamped Saturday closing-auction bar (real
        # Shioaji stamps the 05:00:00 auction as 05:00 or 05:01) -- an earlier
        # implementation leaked these into a spurious Saturday daily bar.
        rows = [
            _bar("2024-01-05 15:01:00", 200, 202, 199, 201, 5),   # Fri night start
            _bar("2024-01-05 23:59:00", 201, 203, 200, 202, 5),
            _bar("2024-01-06 00:30:00", 202, 204, 201, 203, 5),   # Sat overnight tail
            _bar("2024-01-06 05:01:00", 203, 205, 202, 204, 5),   # Sat 05:00 closing auction
            _bar("2024-01-08 08:46:00", 210, 211, 209, 210, 5),   # Mon day session
            _bar("2024-01-08 13:45:00", 210, 212, 209, 211, 5),
        ]
        aware = resample_kbars(_taifex_frame(rows), "1D", session_aware=True)

        assert len(aware) == 1  # no Saturday bar
        mon = aware.loc["2024-01-08"]
        assert mon["open"] == 200    # Friday 15:01 open
        assert mon["close"] == 211   # Monday's real 13:45 close
        assert mon["volume"] == 30   # every bar of the session accounted for

    def test_impossible_session_bars_are_dropped_as_junk(self) -> None:
        # The Shioaji simulation feed emits dust bars in windows where TAIFEX
        # has no session at all (observed live: Sunday "day" prints, Monday
        # pre-dawn bars implying a nonexistent Sunday night session, weekday
        # 13:46-14:59 dead-zone bars). These must not fabricate daily bars.
        rows = [
            _bar("2024-01-07 09:38:00", 150, 150, 150, 150, 2),   # Sunday "day" print
            _bar("2024-01-08 03:00:00", 151, 151, 151, 151, 1),   # Mon pre-dawn (no Sun night session)
            _bar("2024-01-08 14:30:00", 152, 152, 152, 152, 1),   # weekday dead zone
            _bar("2024-01-08 08:46:00", 210, 211, 209, 210, 5),   # legit Mon day session
            _bar("2024-01-08 13:45:00", 210, 212, 209, 211, 5),
        ]
        aware = resample_kbars(_taifex_frame(rows), "1D", session_aware=True)

        assert len(aware) == 1
        mon = aware.loc["2024-01-08"]
        assert mon["open"] == 210     # junk 03:00 bar did not leak into Monday's open
        assert mon["close"] == 211
        assert mon["volume"] == 10    # only the two legit bars counted

    def test_holiday_bridge_night_session_lands_after_holiday(self) -> None:
        # Tuesday night session with Wednesday a market holiday (no Wednesday
        # day bars in the data): the session belongs to Thursday.
        rows = [
            _bar("2024-01-09 15:01:00", 300, 301, 299, 300, 5),   # Tue night start
            _bar("2024-01-10 04:59:00", 301, 302, 300, 301, 5),   # Wed pre-dawn tail (same session)
            _bar("2024-01-11 08:46:00", 305, 306, 304, 305, 5),   # Thu day session
            _bar("2024-01-11 13:45:00", 305, 307, 304, 306, 5),
        ]
        aware = resample_kbars(_taifex_frame(rows), "1D", session_aware=True)

        assert len(aware) == 1
        thu = aware.loc["2024-01-11"]
        assert thu["open"] == 300
        assert thu["close"] == 306
        assert thu["volume"] == 20


class TestRolloverBackAdjust:
    """Ratio back-adjustment of the R1/R2 continuous-contract splice.

    The splice sits at 13:30 -> 13:31 on each settlement date (3rd Wednesday,
    holiday-shifted): the expiring contract's final-settlement trade prints
    at 13:30 and the alias points at the next month from 13:31. Confirmed on
    real 2020-2026 TXFR1 data (2025-06-18: 22308 -> 21909, -1.79% dividend
    discount) where 78 raw splices embedded a compounded x0.853 phantom
    return.
    """

    @staticmethod
    def _roll_frame() -> pd.DataFrame:
        # 2024-01-17 is the 3rd Wednesday of January 2024 (settlement date).
        # Old contract trades at ~100 through 13:30; new contract at ~90
        # from 13:31 (a -10% dividend-discount roll, exaggerated for clarity).
        rows = [
            _bar("2024-01-16 08:46:00", 100, 101, 99, 100, 10),   # Tue (pre-roll day session)
            _bar("2024-01-16 13:45:00", 100, 102, 99, 100, 10),
            _bar("2024-01-17 08:46:00", 100, 101, 99, 100, 10),   # settlement Wed, old contract
            _bar("2024-01-17 13:30:00", 100, 101, 99, 100, 10),   # final settlement print
            _bar("2024-01-17 13:31:00", 90, 91, 89, 90, 10),      # alias now next month
            _bar("2024-01-17 13:45:00", 90, 92, 89, 90, 10),
            _bar("2024-01-18 08:46:00", 90, 91, 89, 90, 10),      # Thu, new contract
            _bar("2024-01-18 13:45:00", 90, 92, 89, 91, 10),
        ]
        return _taifex_frame(rows)

    def test_splice_is_scaled_away_and_returns_become_continuous(self) -> None:
        from backtest.loaders._shioaji_kbars import back_adjust_taifex_rollovers

        adjusted = back_adjust_taifex_rollovers(self._roll_frame(), "TXFR1")

        # Everything at or before the 13:30 settlement print is scaled by
        # new/old = 90/100 = 0.9; everything after is untouched.
        assert adjusted.loc["2024-01-16 08:46:00", "open"] == pytest.approx(90.0)
        assert adjusted.loc["2024-01-17 13:30:00", "close"] == pytest.approx(90.0)
        assert adjusted.loc["2024-01-17 13:31:00", "open"] == pytest.approx(90.0)  # continuous now
        assert adjusted.loc["2024-01-18 13:45:00", "close"] == pytest.approx(91.0)  # unscaled

    def test_volume_is_never_scaled(self) -> None:
        from backtest.loaders._shioaji_kbars import back_adjust_taifex_rollovers

        adjusted = back_adjust_taifex_rollovers(self._roll_frame(), "TXFR1")
        assert (adjusted["volume"] == 10).all()

    def test_dated_contracts_are_untouched(self) -> None:
        from backtest.loaders._shioaji_kbars import back_adjust_taifex_rollovers

        frame = self._roll_frame()
        adjusted = back_adjust_taifex_rollovers(frame, "TXFG6")
        assert adjusted is frame  # no copy, no scaling -- dated contracts never roll

    def test_missing_boundary_bars_leave_splice_raw(self, capsys: pytest.CaptureFixture) -> None:
        from backtest.loaders._shioaji_kbars import back_adjust_taifex_rollovers

        # Dropping the 13:31 bar leaves the 13:31-13:36 new-contract window
        # empty on settlement day (the 13:45 bar sits outside it), so no
        # ratio can be measured for this roll.
        frame = self._roll_frame().drop(pd.Timestamp("2024-01-17 13:31:00"))
        adjusted = back_adjust_taifex_rollovers(frame, "TXFR1")
        assert "splice left raw" in capsys.readouterr().out
        assert adjusted.loc["2024-01-16 08:46:00", "open"] == pytest.approx(100.0)

    def test_multiple_rolls_compound_backward(self) -> None:
        from backtest.loaders._shioaji_kbars import back_adjust_taifex_rollovers

        rows = [
            # roll 1: 2024-01-17, 100 -> 90 (ratio 0.9)
            _bar("2024-01-17 13:30:00", 100, 100, 100, 100, 1),
            _bar("2024-01-17 13:31:00", 90, 90, 90, 90, 1),
            # roll 2: 2024-02-21 (3rd Wed of Feb), 90 -> 99 (ratio 1.1)
            _bar("2024-02-21 13:30:00", 90, 90, 90, 90, 1),
            _bar("2024-02-21 13:31:00", 99, 99, 99, 99, 1),
            _bar("2024-02-22 13:45:00", 99, 99, 99, 99, 1),
        ]
        adjusted = back_adjust_taifex_rollovers(_taifex_frame(rows), "TXFR1")

        # Oldest bar carries both ratios: 100 * 0.9 * 1.1 = 99.
        assert adjusted.loc["2024-01-17 13:30:00", "close"] == pytest.approx(99.0)
        # Between the two rolls: only the later ratio applies (90 * 1.1).
        assert adjusted.loc["2024-02-21 13:30:00", "close"] == pytest.approx(99.0)
        # After the last roll: untouched.
        assert adjusted.loc["2024-02-22 13:45:00", "close"] == pytest.approx(99.0)

    def test_env_flag_disables(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from backtest.loaders._shioaji_kbars import ROLLOVER_ADJUST_ENV, rollover_adjust_enabled

        assert rollover_adjust_enabled() is True  # default on
        monkeypatch.setenv(ROLLOVER_ADJUST_ENV, "0")
        assert rollover_adjust_enabled() is False
