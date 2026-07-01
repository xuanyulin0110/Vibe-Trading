"""Shared Shioaji K-bar plumbing: login lock cleanup, request chunking, and
minute-bar resampling to any target interval.

Shioaji's ``api.kbars()`` only returns 1-minute bars and caps each request at
29 calendar days (a hard 30-day server limit). Both the equity loader
(``shioaji_loader.py``), the futures loader (``shioaji_futures_loader.py``),
and — conceptually — the read-only trading connector all need the same three
things: clear stale contract-cache locks before login, pull 1-minute bars in
<=29-day chunks, and roll those minute bars up to whatever interval the caller
asked for. This module is the single source of truth so the logic is not
copy-pasted per loader (see plan Phase 2b/2c).

Market-data timestamps from Shioaji are already Taiwan wall-clock time -- do
not add +8h (see MARKET_DATA.md's "Market Data Time Handling" in the bundled
Shioaji skill).

Also hosts ``suppress_native_stdout`` (see plan Phase 3): Shioaji's connection
layer is a compiled extension (``shioaji/_core.abi3.so``), not pure Python, so
its "Response Code / Event Code / Session up" connection logs write straight
to the process's stdout file descriptor -- they can't be silenced via Python's
``logging`` module. That's harmless for a CLI backtest run, but fatal for the
MCP server: MCP stdio requires stdout to carry *only* JSON-RPC frames, and
these lines corrupt that stream for every TW equity/futures MCP tool call
(confirmed empirically -- a real ``mcp`` Python client's stdio reader logged
"Failed to parse JSONRPC message from server" for each such line). Every
Shioaji SDK call site (login, kbars, quotes, account/position/trade queries)
must run inside this context manager.

Also hosts ``fetch_minute_kbars_cached`` (see plan follow-up, 2026-07-01): a
gap-aware persistent minute-bar cache. Shioaji enforces a hard daily byte
quota (confirmed via ``api.usage()`` -- ``UsageOut(bytes=530316799,
limit_bytes=524288000, remaining_bytes=-6028799)`` after a day of heavy
testing) plus a combined 50-queries/10s rate limit across
ticks/snapshots/kbars/credit_enquires/short_stock_sources (see
https://sinotrade.github.io/zh/tutor/limit/). Exceeding either does NOT
raise -- kbars()/ticks()/snapshots() silently return empty results, which is
indistinguishable at the single-request level from "genuinely no data for
this range" (e.g. a holiday). See ``fetch_minute_kbars_cached``'s docstring
for the full design.
"""

from __future__ import annotations

import collections
import contextlib
import datetime as dt
import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Iterator, List

import pandas as pd

from backtest.loaders.base import (
    _read_loader_cache_frame,
    _sanitize_cache_segment,
    _write_loader_cache_frame,
)


@contextlib.contextmanager
def suppress_native_stdout() -> Iterator[None]:
    """Redirect the stdout file descriptor away during a Shioaji SDK call.

    Shioaji's native extension writes connection/event logs directly to fd 1,
    bypassing ``sys.stdout`` reassignment -- only an OS-level fd redirect
    catches it. Restores the original fd 1 afterward even on exception.
    """
    saved_fd = os.dup(1)
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull_fd, 1)
        yield
    finally:
        os.dup2(saved_fd, 1)
        os.close(devnull_fd)
        os.close(saved_fd)


class _ShioajiCallGate:
    """Serializes and rate-limits concurrent Shioaji native calls.

    Two independent constraints collapse into one gate, both because of the
    same underlying fact: Shioaji's account-level state (fd 1, and the
    server-side query quota) is *process-wide*, not thread-local, so
    parallelizing loader fetches (see fetch_minute_kbars_cached's callers)
    needs both handled at the one place every native call actually goes
    through:

    1. ``suppress_native_stdout``'s ``os.dup2`` redirects fd 1 for the whole
       process. Two threads concurrently entering/exiting that context race
       on the same fd -- worse than a corrupted log line, a torn dup2/close
       sequence can leave fd 1 permanently broken (or close a fd another
       thread is still using), and this is exactly the fd the MCP stdio
       transport depends on. Only one thread may be inside a suppressed
       native call at a time.
    2. Shioaji's documented combined 50-queries/10s limit across
       ticks/snapshots/kbars/credit_enquires/short_stock_sources is
       account-wide (sinotrade.github.io/zh/tutor/limit/); concurrent
       threads must share one counter, not one each.

    Both are satisfied by acquiring the same lock before every native call:
    only one thread is ever mid-call, and while holding the lock is also the
    natural place to enforce the shared rate window. Everything *outside* the
    gated call -- chunk-boundary math, DataFrame construction, HTTP wait
    inside the call itself for whichever thread currently holds the gate --
    still overlaps across threads, so parallelizing a multi-code fetch still
    gains real wall-clock throughput even though native calls are serialized.
    """

    def __init__(self, max_calls: int, period_seconds: float) -> None:
        self._max_calls = max_calls
        self._period = period_seconds
        self._lock = threading.Lock()
        self._call_times: collections.deque[float] = collections.deque()

    @contextlib.contextmanager
    def call(self) -> Iterator[None]:
        with self._lock:
            self._wait_for_slot_locked()
            self._call_times.append(time.monotonic())
            with suppress_native_stdout():
                yield

    def _wait_for_slot_locked(self) -> None:
        """Block (with the gate lock held) until a call slot is free. Caller
        must hold ``self._lock``."""
        while True:
            now = time.monotonic()
            while self._call_times and now - self._call_times[0] > self._period:
                self._call_times.popleft()
            if len(self._call_times) < self._max_calls:
                return
            time.sleep(self._period - (now - self._call_times[0]))


#: Conservative margin under Shioaji's documented 50-queries/10s combined
#: limit (ticks/snapshots/kbars/credit_enquires/short_stock_sources) -- 40
#: leaves headroom for other concurrent usage of the same account (e.g. a
#: live quote check running alongside a backtest) that this process can't see.
_shioaji_call_gate = _ShioajiCallGate(max_calls=40, period_seconds=10.0)

#: Concurrent per-code fetch workers, shared by both Shioaji loaders' fetch().
#: Native Shioaji calls stay serialized+rate-limited by _shioaji_call_gate
#: regardless of this count -- more workers only buys overlap on the
#: Python-level work (chunk math, DataFrame construction) and HTTP wait, not
#: more actual throughput past the account-wide rate limit. Matches the
#: precedent count alpha_bench_tool.py already uses for concurrent Tushare
#: fetches.
FETCH_WORKERS = 5

#: Lock files older than this are assumed abandoned by a dead/killed process
#: rather than held by a genuinely in-progress download (which finishes in
#: well under a minute) -- see ``clear_stale_shioaji_locks``.
_STALE_LOCK_SECONDS = 120.0

#: Stays under the upstream 30-calendar-day per-request window.
_CHUNK_DAYS = 29

#: Canonical backtest interval token -> pandas resample rule. ``1m`` maps to
#: None because the raw Shioaji bars are already 1-minute (no resample needed).
#: Accepts both the backtest vocabulary (``1H``/``1D``, per metrics.py) and the
#: connector vocabulary (``1h``/``1d``) by normalizing case on lookup.
_INTERVAL_RULES: dict[str, str | None] = {
    "1m": None,
    "5m": "5min",
    "15m": "15min",
    "30m": "30min",
    "1h": "1h",
    "4h": "4h",
    "1d": "1D",
}

_OHLCV_AGG = {
    "open": "first",
    "high": "max",
    "low": "min",
    "close": "last",
    "volume": "sum",
}


def normalize_interval(interval: str) -> str:
    """Normalize an interval token to the canonical lower-case key.

    ``1D`` -> ``1d``, ``1H`` -> ``1h``, ``5m`` -> ``5m``. Minute tokens keep
    their lower-case ``m``; hour/day tokens are lower-cased so both the
    backtest (``1H``/``1D``) and connector (``1h``/``1d``) vocabularies resolve
    to the same key.
    """
    token = interval.strip()
    return token[:-1] + token[-1].lower() if token else token


def is_supported_interval(interval: str) -> bool:
    """Return whether the interval is one this module can produce."""
    return normalize_interval(interval) in _INTERVAL_RULES


def clear_stale_shioaji_locks(max_age_seconds: float = _STALE_LOCK_SECONDS) -> None:
    """Remove stale Shioaji contract-cache lock files before login.

    Confirmed empirically: ``shioaji.Shioaji().login()`` writes a
    ``contracts-*.parquet.lock`` file per contract type into ``SJ_HOME_PATH``
    (default ``~/.shioaji``) during the contract download and never removes it
    afterward, even on a clean process exit. A later login then hangs
    indefinitely waiting on a lock no live process holds -- reproduced
    repeatedly in testing. Only locks older than ``max_age_seconds`` are
    removed, so a genuinely concurrent in-progress download is not disturbed.
    """
    home = Path(os.environ.get("SJ_HOME_PATH") or (Path.home() / ".shioaji"))
    if not home.is_dir():
        return
    now = time.time()
    for lock_file in home.glob("*.lock"):
        try:
            if now - lock_file.stat().st_mtime > max_age_seconds:
                lock_file.unlink()
        except OSError:
            pass


def date_chunks(start_date: str, end_date: str, days: int = _CHUNK_DAYS):
    """Yield (chunk_start, chunk_end) ISO date pairs covering [start_date, end_date]."""
    cur = dt.date.fromisoformat(start_date)
    last = dt.date.fromisoformat(end_date)
    step = dt.timedelta(days=days - 1)
    while cur <= last:
        chunk_end = min(cur + step, last)
        yield cur.isoformat(), chunk_end.isoformat()
        cur = chunk_end + dt.timedelta(days=1)


def fetch_minute_kbars(api: Any, contract: Any, start_date: str, end_date: str) -> pd.DataFrame:
    """Pull chunked 1-minute K-bars for one contract into a datetime-indexed frame.

    Returns an empty frame (with the OHLCV columns) when no data is available,
    so callers can uniformly branch on ``.empty``.
    """
    chunks: List[pd.DataFrame] = []
    for chunk_start, chunk_end in date_chunks(start_date, end_date):
        try:
            with _shioaji_call_gate.call():
                kbars = api.kbars(contract, start=chunk_start, end=chunk_end)
        except Exception as exc:  # noqa: BLE001 - one bad chunk should not kill the fetch
            print(f"[WARN] shioaji kbars failed for {chunk_start}..{chunk_end}: {exc}")
            continue
        if kbars is None or not getattr(kbars, "ts", None):
            continue
        chunks.append(pd.DataFrame({
            "open": kbars.Open,
            "high": kbars.High,
            "low": kbars.Low,
            "close": kbars.Close,
            "volume": kbars.Volume,
        }, index=pd.to_datetime(kbars.ts)))

    if not chunks:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    minute_df = pd.concat(chunks).sort_index()
    for col in ("open", "high", "low", "close", "volume"):
        minute_df[col] = pd.to_numeric(minute_df[col], errors="coerce")
    return minute_df.dropna(subset=["open", "high", "low", "close"])


# ---------------------------------------------------------------------------
# Gap-aware persistent minute-bar cache (see module docstring for why).
# ---------------------------------------------------------------------------

MINUTE_CACHE_ENV = "VIBE_TRADING_SHIOAJI_MINUTE_CACHE"

#: Unlike backtest.loaders.base's general opt-in loader cache (default off),
#: this one defaults ON: it's safe-by-construction (always gap-filled to the
#: full requested range -- via _quota_has_headroom's guard -- rather than a
#: possibly-stale partial read), and Shioaji's hard byte quota makes
#: cache-by-default the safer choice specifically for this source. Opt out
#: with VIBE_TRADING_SHIOAJI_MINUTE_CACHE=0.
_MINUTE_CACHE_FALSE_VALUES = {"0", "false", "no", "off"}


def _minute_cache_enabled() -> bool:
    return os.getenv(MINUTE_CACHE_ENV, "").strip().lower() not in _MINUTE_CACHE_FALSE_VALUES


def _minute_cache_path(source: str, symbol: str) -> Path:
    """Path to the persistent per-(source, symbol) 1-minute-bar parquet store."""
    source_dir = _sanitize_cache_segment(source)
    symbol_file = _sanitize_cache_segment(symbol)
    return (
        Path.home() / ".vibe-trading" / "cache" / "loaders"
        / source_dir / "_minute_bars" / f"{symbol_file}.parquet"
    )


def _coverage_path(cache_path: Path) -> Path:
    # Distinct suffix from _loader_cache_metadata_path's "<name>.json" (index
    # dtype/column metadata) -- no collision.
    return cache_path.with_suffix(cache_path.suffix + ".coverage.json")


def _read_coverage(cache_path: Path) -> List[tuple[str, str]]:
    """Load the list of (start, end) ISO date pairs already fetched from Shioaji."""
    path = _coverage_path(cache_path)
    if not path.is_file():
        return []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return [(str(s), str(e)) for s, e in raw]
    except Exception:  # noqa: BLE001 - corrupt sidecar degrades to "nothing covered yet"
        return []


def _write_coverage(cache_path: Path, intervals: List[tuple[str, str]]) -> None:
    path = _coverage_path(cache_path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(intervals), encoding="utf-8")
    except OSError:
        pass  # non-fatal, mirrors _write_loader_cache_frame's write-failure tolerance


def _merge_date_intervals(intervals: List[tuple[str, str]]) -> List[tuple[str, str]]:
    """Merge overlapping/adjacent (start, end) ISO date pairs into a minimal sorted set."""
    if not intervals:
        return []
    parsed = sorted(
        (dt.date.fromisoformat(s), dt.date.fromisoformat(e)) for s, e in intervals
    )
    merged: List[List[dt.date]] = [list(parsed[0])]
    for start, end in parsed[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end + dt.timedelta(days=1):
            merged[-1][1] = max(last_end, end)
        else:
            merged.append([start, end])
    return [(s.isoformat(), e.isoformat()) for s, e in merged]


def _subtract_date_intervals(
    requested: tuple[str, str], covered: List[tuple[str, str]],
) -> List[tuple[str, str]]:
    """Return the gap sub-ranges of ``requested`` not covered by any interval in ``covered``."""
    req_start = dt.date.fromisoformat(requested[0])
    req_end = dt.date.fromisoformat(requested[1])

    clipped: List[tuple[dt.date, dt.date]] = []
    for s, e in covered:
        cs = max(dt.date.fromisoformat(s), req_start)
        ce = min(dt.date.fromisoformat(e), req_end)
        if cs <= ce:
            clipped.append((cs, ce))
    clipped.sort()

    gaps: List[tuple[str, str]] = []
    cursor = req_start
    for cs, ce in clipped:
        if cs > cursor:
            gaps.append((cursor.isoformat(), (cs - dt.timedelta(days=1)).isoformat()))
        cursor = max(cursor, ce + dt.timedelta(days=1))
    if cursor <= req_end:
        gaps.append((cursor.isoformat(), req_end.isoformat()))
    return gaps


def _quota_has_headroom(api: Any) -> bool:
    """Best-effort pre-flight quota check via Shioaji's documented ``api.usage()``.

    An over-quota kbars() query returns an empty result, not an error (see
    module docstring) -- indistinguishable at the single-request level from
    "genuinely no data for this range". Without this check, a quota-exhausted
    fetch would get marked "covered" by ``fetch_minute_kbars_cached`` below,
    and the resulting hole would never be retried. If ``usage()`` itself
    fails, default to assuming headroom -- this guard exists for the known,
    confirmed failure mode (exhausted quota), not to block on an unrelated
    problem.
    """
    try:
        with _shioaji_call_gate.call():
            usage = api.usage()
        return usage.remaining_bytes > 0
    except Exception:  # noqa: BLE001
        return True


def fetch_minute_kbars_cached(
    api: Any, contract: Any, *, source: str, symbol: str, start_date: str, end_date: str,
) -> pd.DataFrame:
    """Gap-aware, persistent alternative to :func:`fetch_minute_kbars`.

    Maintains one 1-minute-bar parquet store per ``(source, symbol)`` plus a
    coverage sidecar of already-fetched date ranges. Each call fetches only
    the sub-ranges of ``[start_date, end_date]`` not yet covered, merges them
    into the local store, and serves the full requested range from disk.
    Every resampled interval shares the same underlying store, so e.g. a 5m
    request over dates a prior 1D request already covered is also a full
    cache hit. Falls back to a plain, uncached :func:`fetch_minute_kbars`
    call when ``VIBE_TRADING_SHIOAJI_MINUTE_CACHE=0``.

    Today is never marked "covered", however long ago it was first fetched
    this process: its bar is still forming intraday, so treating it as a
    settled fact would serve stale data to a later call the same day (same
    rationale as ``backtest.loaders.base.loader_cache_range_is_final``). It's
    still fetched and merged into the local store for *this* call's return
    value -- just not persisted into the coverage sidecar.

    A residual gap: if quota runs out *mid-way* through fetching a single
    multi-chunk gap (rather than being already exhausted before this call
    starts, which ``_quota_has_headroom`` does catch), the partial result for
    that gap still gets marked "covered". Accepted trade-off -- catching the
    common case (already exhausted) without tracking quota per 29-day chunk.
    """
    if not _minute_cache_enabled():
        return fetch_minute_kbars(api, contract, start_date, end_date)

    yesterday = (dt.date.today() - dt.timedelta(days=1)).isoformat()
    settled_end = min(end_date, yesterday)

    cache_path = _minute_cache_path(source, symbol)
    local = _read_loader_cache_frame(cache_path)
    covered = _read_coverage(cache_path)

    gaps: List[tuple[str, str]] = []
    if start_date <= settled_end:
        gaps = _subtract_date_intervals((start_date, settled_end), covered)

    today = dt.date.today().isoformat()
    live_start = max(start_date, today)
    fetch_live = live_start <= end_date

    if gaps or fetch_live:
        new_pieces: List[pd.DataFrame] = []
        if gaps:
            if _quota_has_headroom(api):
                for gap_start, gap_end in gaps:
                    chunk = fetch_minute_kbars(api, contract, gap_start, gap_end)
                    if not chunk.empty:
                        new_pieces.append(chunk)
                covered = _merge_date_intervals(covered + gaps)
            else:
                print(
                    f"[WARN] shioaji minute cache: quota exhausted, serving cached-only "
                    f"data for {symbol} ({len(gaps)} uncached gap(s) skipped this call)"
                )
                gaps = []  # nothing actually fetched -- don't persist a bogus "covered"
        if fetch_live:
            live_chunk = fetch_minute_kbars(api, contract, live_start, end_date)
            if not live_chunk.empty:
                new_pieces.append(live_chunk)

        if new_pieces:
            pieces = [local] if local is not None and not local.empty else []
            pieces.extend(new_pieces)
            local = pd.concat(pieces).sort_index()
            local = local[~local.index.duplicated(keep="last")]
            if local is not None and not local.empty:
                _write_loader_cache_frame(cache_path, local)
        if gaps:  # only persist coverage for the settled portion that was actually fetched
            _write_coverage(cache_path, covered)

    if local is None or local.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date) + pd.Timedelta(days=1)
    return local.loc[(local.index >= start_ts) & (local.index < end_ts)]


def resample_kbars(minute_df: pd.DataFrame, interval: str) -> pd.DataFrame:
    """Roll 1-minute OHLCV bars up to ``interval``.

    ``1m`` is a passthrough (the source is already 1-minute). Any other
    supported interval is aggregated with standard OHLCV rules.

    Raises:
        ValueError: ``interval`` is not one of the supported tokens.
    """
    key = normalize_interval(interval)
    if key not in _INTERVAL_RULES:
        raise ValueError(f"unsupported interval: {interval!r}")
    rule = _INTERVAL_RULES[key]
    if rule is None:  # 1m passthrough
        return minute_df
    agg = minute_df.resample(rule).agg(_OHLCV_AGG)
    return agg.dropna(subset=["open", "high", "low", "close"])
