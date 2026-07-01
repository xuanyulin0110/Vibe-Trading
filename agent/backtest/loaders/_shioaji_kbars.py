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
"""

from __future__ import annotations

import contextlib
import datetime as dt
import os
import time
from pathlib import Path
from typing import Any, Iterator, List

import pandas as pd


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
            with suppress_native_stdout():
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
