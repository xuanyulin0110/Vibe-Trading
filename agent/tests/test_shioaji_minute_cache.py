"""Gap-aware persistent minute-bar cache: interval algebra + fetch_minute_kbars_cached.

Built to solve a real problem hit in this project on 2026-07-01: Shioaji
enforces a hard daily byte quota (confirmed via api.usage() --
UsageOut(bytes=530316799, limit_bytes=524288000, remaining_bytes=-6028799)
after a day of repeated backtests) and exceeding it makes kbars() silently
return empty results, not an error. The general per-request cache in
backtest/loaders/base.py caches by exact (source,symbol,timeframe,start,end)
match, so re-running a backtest with a slightly different date range was a
full cache miss and re-fetched the whole range from Shioaji every time --
these tests exist to prove the replacement (gap-only fetching against a
persistent per-symbol store) actually avoids that.
"""

from __future__ import annotations

import datetime as dt
import sys
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

from backtest.loaders._shioaji_kbars import (
    MINUTE_CACHE_ENV,
    _merge_date_intervals,
    _minute_cache_path,
    _quota_has_headroom,
    _read_coverage,
    _subtract_date_intervals,
    fetch_minute_kbars_cached,
)


# ---------------------------------------------------------------- interval algebra


class TestMergeDateIntervals:
    def test_empty_input(self) -> None:
        assert _merge_date_intervals([]) == []

    def test_single_interval_unchanged(self) -> None:
        assert _merge_date_intervals([("2024-01-01", "2024-01-05")]) == [("2024-01-01", "2024-01-05")]

    def test_adjacent_intervals_merge(self) -> None:
        # 01-06 is exactly one day after 01-05 -- adjacent, must merge into one.
        result = _merge_date_intervals([("2024-01-01", "2024-01-05"), ("2024-01-06", "2024-01-10")])
        assert result == [("2024-01-01", "2024-01-10")]

    def test_overlapping_intervals_merge(self) -> None:
        result = _merge_date_intervals([("2024-01-01", "2024-01-10"), ("2024-01-05", "2024-01-15")])
        assert result == [("2024-01-01", "2024-01-15")]

    def test_disjoint_intervals_stay_separate(self) -> None:
        result = _merge_date_intervals([("2024-01-01", "2024-01-05"), ("2024-02-01", "2024-02-05")])
        assert result == [("2024-01-01", "2024-01-05"), ("2024-02-01", "2024-02-05")]

    def test_unsorted_input_still_merges_correctly(self) -> None:
        result = _merge_date_intervals([("2024-02-01", "2024-02-05"), ("2024-01-01", "2024-01-05")])
        assert result == [("2024-01-01", "2024-01-05"), ("2024-02-01", "2024-02-05")]

    def test_contained_interval_absorbed(self) -> None:
        result = _merge_date_intervals([("2024-01-01", "2024-01-31"), ("2024-01-10", "2024-01-15")])
        assert result == [("2024-01-01", "2024-01-31")]


class TestSubtractDateIntervals:
    def test_no_coverage_returns_whole_range(self) -> None:
        gaps = _subtract_date_intervals(("2024-01-01", "2024-01-10"), [])
        assert gaps == [("2024-01-01", "2024-01-10")]

    def test_full_coverage_returns_no_gaps(self) -> None:
        gaps = _subtract_date_intervals(("2024-01-01", "2024-01-10"), [("2024-01-01", "2024-01-10")])
        assert gaps == []

    def test_wider_coverage_returns_no_gaps(self) -> None:
        gaps = _subtract_date_intervals(("2024-01-05", "2024-01-08"), [("2024-01-01", "2024-01-31")])
        assert gaps == []

    def test_middle_covered_leaves_two_gaps(self) -> None:
        gaps = _subtract_date_intervals(("2024-01-01", "2024-01-20"), [("2024-01-05", "2024-01-10")])
        assert gaps == [("2024-01-01", "2024-01-04"), ("2024-01-11", "2024-01-20")]

    def test_only_start_covered_leaves_tail_gap(self) -> None:
        gaps = _subtract_date_intervals(("2024-01-01", "2024-01-20"), [("2024-01-01", "2024-01-10")])
        assert gaps == [("2024-01-11", "2024-01-20")]

    def test_coverage_outside_request_is_ignored(self) -> None:
        gaps = _subtract_date_intervals(("2024-01-10", "2024-01-20"), [("2023-01-01", "2023-12-31")])
        assert gaps == [("2024-01-10", "2024-01-20")]

    def test_this_is_the_core_scenario_iteratively_widening_a_range(self) -> None:
        """The actual failure mode this cache exists to fix: re-running the
        same backtest with a slightly different end_date must only fetch the
        new tail, not the whole range again."""
        covered = [("2023-01-01", "2024-11-30")]
        gaps = _subtract_date_intervals(("2023-01-01", "2024-12-31"), covered)
        assert gaps == [("2024-12-01", "2024-12-31")]


class TestQuotaHasHeadroom:
    def test_positive_remaining_bytes_is_headroom(self) -> None:
        api = SimpleNamespace(usage=lambda: SimpleNamespace(remaining_bytes=1000))
        assert _quota_has_headroom(api) is True

    def test_negative_remaining_bytes_is_no_headroom(self) -> None:
        # The exact real-world shape hit on 2026-07-01.
        api = SimpleNamespace(usage=lambda: SimpleNamespace(remaining_bytes=-6028799))
        assert _quota_has_headroom(api) is False

    def test_zero_remaining_bytes_is_no_headroom(self) -> None:
        api = SimpleNamespace(usage=lambda: SimpleNamespace(remaining_bytes=0))
        assert _quota_has_headroom(api) is False

    def test_usage_call_failure_assumes_headroom(self) -> None:
        def _raise():
            raise RuntimeError("usage() not supported")

        api = SimpleNamespace(usage=_raise)
        assert _quota_has_headroom(api) is True


# ---------------------------------------------------------------- fetch_minute_kbars_cached


@pytest.fixture
def fake_duckdb(monkeypatch: pytest.MonkeyPatch) -> None:
    """Install a tiny DuckDB stand-in so cache tests stay dependency-light
    (mirrors tests/test_loader_retry_helpers.py's fixture of the same name)."""

    class _Connection:
        def __init__(self) -> None:
            self._tables: dict[str, pd.DataFrame] = {}
            self._frame: pd.DataFrame | None = None

        def register(self, name: str, frame: pd.DataFrame) -> None:
            self._tables[name] = frame.copy()

        def execute(self, sql: str) -> "_Connection":
            path = _first_sql_string(sql)
            if sql.strip().upper().startswith("COPY "):
                self._tables["cache_frame"].to_pickle(path)
                return self
            self._frame = pd.read_pickle(path)
            return self

        def fetchdf(self) -> pd.DataFrame:
            assert self._frame is not None
            return self._frame.copy()

        def close(self) -> None:
            pass

    def _first_sql_string(sql: str) -> str:
        start = sql.index("'") + 1
        end = sql.index("'", start)
        return sql[start:end].replace("''", "'")

    monkeypatch.setitem(
        sys.modules, "duckdb", SimpleNamespace(connect=lambda database=":memory:": _Connection()),
    )


class _FakeKbars:
    def __init__(self, ts: list, open_: list, high: list, low: list, close: list, volume: list) -> None:
        self.ts = ts
        self.Open = open_
        self.High = high
        self.Low = low
        self.Close = close
        self.Volume = volume


class _FakeApi:
    """Records every (start, end) passed to kbars() and returns one synthetic
    09:00 bar per calendar day in that range (empty for dates in `holidays`)."""

    def __init__(self, holidays: set[str] | None = None, remaining_bytes: int = 10_000) -> None:
        self.calls: list[tuple[str, str]] = []
        self.holidays = holidays or set()
        self.remaining_bytes = remaining_bytes

    def kbars(self, contract: Any, start: str, end: str) -> _FakeKbars:
        self.calls.append((start, end))
        cur = dt.date.fromisoformat(start)
        last = dt.date.fromisoformat(end)
        ts, opens, highs, lows, closes, vols = [], [], [], [], [], []
        while cur <= last:
            if cur.isoformat() not in self.holidays:
                price = float(cur.toordinal() % 100 + 1)
                ts.append(f"{cur.isoformat()} 09:00:00")
                opens.append(price)
                highs.append(price + 1)
                lows.append(price - 1)
                closes.append(price + 0.5)
                vols.append(1000)
            cur += dt.timedelta(days=1)
        return _FakeKbars(ts, opens, highs, lows, closes, vols)

    def usage(self) -> SimpleNamespace:
        return SimpleNamespace(remaining_bytes=self.remaining_bytes)


@pytest.fixture(autouse=True)
def _isolated_cache_home(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))


class TestFetchMinuteKbarsCachedDiskFree:
    """VIBE_TRADING_SHIOAJI_MINUTE_CACHE=0 -- must behave exactly like the
    plain uncached fetch_minute_kbars(), with no disk I/O at all."""

    def test_disabled_calls_plain_fetch_every_time(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(MINUTE_CACHE_ENV, "0")
        api = _FakeApi()
        fetch_minute_kbars_cached(
            api, object(), source="shioaji", symbol="2330.TW",
            start_date="2024-01-01", end_date="2024-01-05",
        )
        fetch_minute_kbars_cached(
            api, object(), source="shioaji", symbol="2330.TW",
            start_date="2024-01-01", end_date="2024-01-05",
        )
        assert len(api.calls) == 2  # no caching -- fetched twice for the same range


class TestFetchMinuteKbarsCachedGapFetching:
    def test_first_call_fetches_full_range(self, fake_duckdb, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(MINUTE_CACHE_ENV, "1")
        api = _FakeApi()
        df = fetch_minute_kbars_cached(
            api, object(), source="shioaji", symbol="2330.TW",
            start_date="2020-01-01", end_date="2020-01-05",
        )
        assert api.calls == [("2020-01-01", "2020-01-05")]
        assert len(df) == 5

    def test_second_call_same_range_is_a_full_cache_hit(
        self, fake_duckdb, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv(MINUTE_CACHE_ENV, "1")
        api = _FakeApi()
        kwargs = dict(
            source="shioaji", symbol="2330.TW", start_date="2020-01-01", end_date="2020-01-05",
        )
        fetch_minute_kbars_cached(api, object(), **kwargs)
        n_calls_after_first = len(api.calls)
        fetch_minute_kbars_cached(api, object(), **kwargs)
        assert len(api.calls) == n_calls_after_first  # zero new kbars() calls

    def test_widened_range_only_fetches_the_new_tail(
        self, fake_duckdb, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The exact real scenario this cache was built for: re-running with a
        later end_date must only fetch the delta, not the whole range again."""
        monkeypatch.setenv(MINUTE_CACHE_ENV, "1")
        api = _FakeApi()
        fetch_minute_kbars_cached(
            api, object(), source="shioaji", symbol="2330.TW",
            start_date="2020-01-01", end_date="2020-01-10",
        )
        api.calls.clear()

        df = fetch_minute_kbars_cached(
            api, object(), source="shioaji", symbol="2330.TW",
            start_date="2020-01-01", end_date="2020-01-15",
        )
        # Only the new tail (01-11..01-15) should have been fetched from Shioaji.
        assert api.calls == [("2020-01-11", "2020-01-15")]
        assert len(df) == 15  # but the full 15-day range is still returned

    def test_narrowed_and_shifted_range_only_fetches_missing_middle(
        self, fake_duckdb, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv(MINUTE_CACHE_ENV, "1")
        api = _FakeApi()
        fetch_minute_kbars_cached(
            api, object(), source="shioaji", symbol="2330.TW",
            start_date="2020-01-05", end_date="2020-01-10",
        )
        api.calls.clear()

        df = fetch_minute_kbars_cached(
            api, object(), source="shioaji", symbol="2330.TW",
            start_date="2020-01-01", end_date="2020-01-20",
        )
        assert api.calls == [("2020-01-01", "2020-01-04"), ("2020-01-11", "2020-01-20")]
        assert len(df) == 20

    def test_different_symbol_does_not_share_cache(
        self, fake_duckdb, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv(MINUTE_CACHE_ENV, "1")
        api = _FakeApi()
        fetch_minute_kbars_cached(
            api, object(), source="shioaji", symbol="2330.TW",
            start_date="2020-01-01", end_date="2020-01-05",
        )
        api.calls.clear()
        fetch_minute_kbars_cached(
            api, object(), source="shioaji", symbol="2317.TW",
            start_date="2020-01-01", end_date="2020-01-05",
        )
        assert api.calls == [("2020-01-01", "2020-01-05")]  # different symbol, full fetch

    def test_different_source_does_not_share_cache(
        self, fake_duckdb, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv(MINUTE_CACHE_ENV, "1")
        api = _FakeApi()
        fetch_minute_kbars_cached(
            api, object(), source="shioaji", symbol="TXFR1.TWF",
            start_date="2020-01-01", end_date="2020-01-05",
        )
        api.calls.clear()
        fetch_minute_kbars_cached(
            api, object(), source="shioaji_futures", symbol="TXFR1.TWF",
            start_date="2020-01-01", end_date="2020-01-05",
        )
        assert api.calls == [("2020-01-01", "2020-01-05")]  # different source, full fetch


class TestFetchMinuteKbarsCachedHolidays:
    def test_holiday_gap_inside_covered_range_is_not_refetched(
        self, fake_duckdb, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A date with zero rows (real holiday) inside an already-fetched
        range must not look like 'never fetched' on a later call -- coverage
        is tracked by requested range, not by which dates have data rows."""
        monkeypatch.setenv(MINUTE_CACHE_ENV, "1")
        api = _FakeApi(holidays={"2020-01-03"})
        fetch_minute_kbars_cached(
            api, object(), source="shioaji", symbol="2330.TW",
            start_date="2020-01-01", end_date="2020-01-05",
        )
        api.calls.clear()

        df = fetch_minute_kbars_cached(
            api, object(), source="shioaji", symbol="2330.TW",
            start_date="2020-01-01", end_date="2020-01-05",
        )
        assert api.calls == []  # full cache hit despite the internal holiday gap
        assert len(df) == 4  # 5 days minus the 1 holiday


class TestFetchMinuteKbarsCachedQuotaExhausted:
    def test_exhausted_quota_serves_cached_only_and_does_not_mark_gap_covered(
        self, fake_duckdb, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv(MINUTE_CACHE_ENV, "1")
        api = _FakeApi(remaining_bytes=-100)  # already exhausted
        df = fetch_minute_kbars_cached(
            api, object(), source="shioaji", symbol="2330.TW",
            start_date="2020-01-01", end_date="2020-01-05",
        )
        assert api.calls == []  # never even attempted kbars() with no headroom
        assert df.empty

        cache_path = _minute_cache_path("shioaji", "2330.TW")
        assert _read_coverage(cache_path) == []  # not marked covered -- must retry later

    def test_quota_recovers_next_call_fetches_normally(
        self, fake_duckdb, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv(MINUTE_CACHE_ENV, "1")
        api = _FakeApi(remaining_bytes=-100)
        fetch_minute_kbars_cached(
            api, object(), source="shioaji", symbol="2330.TW",
            start_date="2020-01-01", end_date="2020-01-05",
        )
        api.remaining_bytes = 10_000  # quota reset
        df = fetch_minute_kbars_cached(
            api, object(), source="shioaji", symbol="2330.TW",
            start_date="2020-01-01", end_date="2020-01-05",
        )
        assert api.calls == [("2020-01-01", "2020-01-05")]
        assert len(df) == 5


class TestFetchMinuteKbarsCachedTodayBoundary:
    def test_today_is_always_refetched_within_the_same_process(
        self, fake_duckdb, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Today's bar is still forming intraday -- must never be treated as
        a settled, cacheable fact, or a later same-day call would see stale
        data (mirrors backtest.loaders.base.loader_cache_range_is_final)."""
        monkeypatch.setenv(MINUTE_CACHE_ENV, "1")
        today = dt.date.today().isoformat()
        api = _FakeApi()

        fetch_minute_kbars_cached(
            api, object(), source="shioaji", symbol="2330.TW",
            start_date=today, end_date=today,
        )
        fetch_minute_kbars_cached(
            api, object(), source="shioaji", symbol="2330.TW",
            start_date=today, end_date=today,
        )
        # Both calls hit Shioaji -- today is never served from a "covered" cache.
        assert api.calls == [(today, today), (today, today)]

        cache_path = _minute_cache_path("shioaji", "2330.TW")
        assert _read_coverage(cache_path) == []  # today never persisted as covered

    def test_yesterday_still_gets_cached_normally(
        self, fake_duckdb, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv(MINUTE_CACHE_ENV, "1")
        yesterday = (dt.date.today() - dt.timedelta(days=1)).isoformat()
        api = _FakeApi()

        fetch_minute_kbars_cached(
            api, object(), source="shioaji", symbol="2330.TW",
            start_date=yesterday, end_date=yesterday,
        )
        api.calls.clear()
        fetch_minute_kbars_cached(
            api, object(), source="shioaji", symbol="2330.TW",
            start_date=yesterday, end_date=yesterday,
        )
        assert api.calls == []  # yesterday IS settled -- full cache hit


def test_real_duckdb_round_trip_gap_fetch(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Exercise the real duckdb -> parquet -> duckdb path (other tests here use
    the pickle-based fake_duckdb fixture for speed; this pins that a real
    parquet round-trip doesn't corrupt the DatetimeIndex the gap-detection
    slicing depends on -- mirrors test_loader_cache_real_duckdb_round_trip
    in test_loader_retry_helpers.py)."""
    pytest.importorskip("duckdb")
    monkeypatch.setenv(MINUTE_CACHE_ENV, "1")
    monkeypatch.setenv("HOME", str(tmp_path))
    api = _FakeApi()

    fetch_minute_kbars_cached(
        api, object(), source="shioaji", symbol="2330.TW",
        start_date="2020-01-01", end_date="2020-01-10",
    )
    assert _minute_cache_path("shioaji", "2330.TW").is_file()
    api.calls.clear()

    df = fetch_minute_kbars_cached(
        api, object(), source="shioaji", symbol="2330.TW",
        start_date="2020-01-01", end_date="2020-01-15",
    )
    assert api.calls == [("2020-01-11", "2020-01-15")]
    assert len(df) == 15
    assert isinstance(df.index, pd.DatetimeIndex)
