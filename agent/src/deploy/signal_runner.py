"""Compute today's target weight from a run's own signal_engine.py.

Fidelity contract (the invariant tests pin this):

* Same code: the run's ``code/signal_engine.py`` is loaded through the SAME
  validated loader the backtest runner uses (``backtest.runner``'s AST-checked
  module import), never re-implemented or re-generated.
* Same data: fetched through the same loader registry entry the backtest
  used, from the run config's ORIGINAL ``start_date`` -- full-history
  recompute so ewm/EMA indicator state is bit-identical to the backtest (a
  truncated warmup window drifts in the far decimals and can flip a signal
  sitting on a threshold). The gap-aware persistent kbar cache makes each
  tick's incremental cost just the newest bars.
* Same enrichment: ``backtest.engines.base._maybe_enrich_fundamentals`` --
  which constructs a fresh fundamentals provider per call, so a long-lived
  process never serves stale statement/chip tables.
* Same weight semantics: the backtest's ``_align`` shifts signals by one bar
  (next-bar-open execution) and clips to [-1, 1]; the live equivalent of
  "the weight to hold during the bar that just opened" is therefore the
  signal value at the LAST COMPLETE bar, clipped the same way.

Bar completeness -- the subtle part. A bar whose session hasn't closed yet
must be dropped before reading the last signal:

* Intraday bars: minute source stamps are right-edged (a 1m bar stamped
  08:46 covers 08:45-08:46), while pandas-resampled buckets are left-labeled
  (a 5m bar labeled 08:45 covers stamps 08:46..08:50). Both reduce to
  "complete iff bar END <= now" with END = ts (1m) or ts + interval.
* Daily bars: labeled by trading date at midnight. TAIFEX session-aware
  daily bars attribute the OVERNIGHT session to the next trading day, so at
  08:46 "today's" bar already exists but only holds night data -- half a
  bar. END = trading date's day-session close (13:45 futures / 13:30
  equities); today's bar is incomplete until then.
"""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from backtest.engines.base import _maybe_enrich_fundamentals
from backtest.loaders._shioaji_kbars import normalize_interval
from backtest.loaders.registry import LOADER_REGISTRY
from backtest.runner import _load_module_from_file, _validate_signal_engine_class

from src.deploy.market_calendar import TAIPEI, TW_EQUITY, day_session_close

_INTERVAL_MINUTES = {"1m": 1, "5m": 5, "15m": 15, "30m": 30, "1h": 60, "4h": 240}


class SignalComputationError(RuntimeError):
    """Raised when the tick cannot produce a trustworthy signal."""


@dataclass(frozen=True)
class SignalResult:
    symbol: str
    bar_ts: pd.Timestamp  # last complete bar (naive Taipei, loader convention)
    weight: float  # clipped to [-1, 1]
    close: float  # last complete bar's close (real price on newest segment)
    bars_evaluated: int
    elapsed_seconds: float


def load_run_config(run_dir: Path) -> dict[str, Any]:
    config_path = run_dir / "config.json"
    if not config_path.exists():
        raise SignalComputationError(f"{config_path} not found")
    return json.loads(config_path.read_text(encoding="utf-8"))


def load_signal_engine(run_dir: Path):
    """Load the run's SignalEngine through the backtest runner's validated path."""
    signal_path = run_dir / "code" / "signal_engine.py"
    if not signal_path.exists():
        raise SignalComputationError(f"{signal_path} not found")
    module = _load_module_from_file(signal_path, f"deploy_signal_{run_dir.name}")
    engine_cls = getattr(module, "SignalEngine", None)
    if engine_cls is None:
        raise SignalComputationError("SignalEngine class not found in signal_engine.py")
    _validate_signal_engine_class(engine_cls)
    return engine_cls()


def bar_end(ts: pd.Timestamp, interval: str, market: str) -> dt.datetime:
    """Wall-clock (Taipei) moment after which the bar labeled ``ts`` is final."""
    key = normalize_interval(interval)
    naive = pd.Timestamp(ts).to_pydatetime()
    if key == "1d":
        end = dt.datetime.combine(naive.date(), day_session_close(market))
    elif key == "1m":
        end = naive  # right-edge stamped: final at its own stamp
    else:
        end = naive + dt.timedelta(minutes=_INTERVAL_MINUTES[key])
    return end.replace(tzinfo=TAIPEI)


def drop_incomplete_bars(
    bars: pd.DataFrame, interval: str, market: str, now: dt.datetime,
) -> pd.DataFrame:
    """Trim trailing bars whose session/bucket hasn't closed yet."""
    if bars.empty:
        return bars
    keep = len(bars)
    # Only trailing bars can be incomplete; walk back from the end.
    for i in range(len(bars) - 1, -1, -1):
        if bar_end(bars.index[i], interval, market) <= now:
            break
        keep = i
    return bars.iloc[:keep]


def fetch_run_data(
    run_dir: Path,
    config: dict[str, Any],
    *,
    end_date: str,
    injected_api: Any = None,
) -> dict[str, pd.DataFrame]:
    """Fetch via the run's own loader + enrichment, optionally on a shared session.

    ``injected_api`` (the deploy runtime's persistent paper session -- market
    data is the same real feed in both Shioaji environments) is assigned to
    the loader's ``api`` slot before ``fetch()``; the loaders' idempotent
    ``_ensure_logged_in`` then reuses it instead of logging in per fetch,
    which matters at intraday cadence (Shioaji caps daily logins).
    """
    source = config.get("source", "auto")
    codes = list(config.get("codes") or [])
    interval = config.get("interval", "1D")
    if source == "auto" and codes:
        from backtest.runner import _detect_source

        source = _detect_source(codes[0])
    loader_cls = LOADER_REGISTRY.get(source)
    if loader_cls is None:
        raise SignalComputationError(f"unknown loader source {source!r}")
    loader = loader_cls()
    if injected_api is not None and hasattr(loader, "api"):
        loader.api = injected_api
    data_map = loader.fetch(
        codes,
        config.get("start_date", ""),
        end_date,
        interval=interval,
    )
    if not data_map:
        raise SignalComputationError(f"loader {source!r} returned no data")
    return _maybe_enrich_fundamentals(data_map, config)


def compute_signal(
    run_dir: Path,
    symbol: str,
    market: str,
    *,
    now: dt.datetime | None = None,
    injected_api: Any = None,
    data_map: dict[str, pd.DataFrame] | None = None,
) -> SignalResult:
    """Full pipeline: data -> completeness filter -> SignalEngine -> last weight.

    ``data_map`` can be supplied by tests / dry-runs to bypass fetching.
    """
    started = dt.datetime.now(dt.timezone.utc)
    now = now or dt.datetime.now(TAIPEI)
    config = load_run_config(run_dir)
    interval = config.get("interval", "1D")

    if data_map is None:
        end_date = now.astimezone(TAIPEI).date().isoformat()
        data_map = fetch_run_data(run_dir, config, end_date=end_date, injected_api=injected_api)
    if symbol not in data_map:
        raise SignalComputationError(f"no data for {symbol} (loader returned {sorted(data_map)})")

    filtered = {
        code: drop_incomplete_bars(frame, interval, market, now)
        for code, frame in data_map.items()
    }
    bars = filtered[symbol]
    if bars.empty:
        raise SignalComputationError("no complete bars after completeness filtering")

    engine = load_signal_engine(run_dir)
    signals = engine.generate(filtered)
    series = signals.get(symbol)
    if series is None or len(series) == 0:
        raise SignalComputationError(f"SignalEngine produced no series for {symbol}")
    series = series.reindex(bars.index).fillna(0.0).clip(-1.0, 1.0)  # same clip as _align

    bar_ts = bars.index[-1]
    elapsed = (dt.datetime.now(dt.timezone.utc) - started).total_seconds()
    return SignalResult(
        symbol=symbol,
        bar_ts=pd.Timestamp(bar_ts),
        weight=float(series.iloc[-1]),
        close=float(bars["close"].iloc[-1]),
        bars_evaluated=len(bars),
        elapsed_seconds=elapsed,
    )
