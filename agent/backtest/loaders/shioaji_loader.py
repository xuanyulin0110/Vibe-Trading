"""Shioaji loader for Taiwan equities (TWSE/TPEx) daily OHLCV.

Shioaji is SinoPac's official broker API. It only exposes 1-minute K-bars
(``api.kbars()``), no native daily resolution, so this loader resamples
1-minute bars up to daily OHLCV itself. Each request window is capped at 29
calendar days by the upstream API (a hard 30-day server limit), so a
multi-year fetch is chunked and concatenated -- see
``references/MARKET_DATA.md`` in the bundled Shioaji skill for the exact
chunking rule this follows.

Historical coverage starts 2020-03-02 for stocks; requests before that date
return empty/partial chunks (the caller's runtime fallback to ``finlab``
picks up the rest of the loader chain when this loader yields nothing).

Login uses ``SJ_API_KEY``/``SJ_SEC_KEY`` only -- market data (kbars/ticks/
snapshots) needs no CA certificate, only order placement does. Defaults to
``simulation=True`` (no CA, works on a simulation-only account); set
``SJ_PRODUCTION=true`` to use the production login instead, matching
Shioaji's own env var convention.

Market-data timestamps from Shioaji are already Taiwan wall-clock time --
do not add +8h (see MARKET_DATA.md's "Market Data Time Handling" section).
"""

from __future__ import annotations

import datetime as dt
import os
from typing import Dict, List, Optional

import pandas as pd

from backtest.loaders._symbol_utils import _strip_tw_suffix
from backtest.loaders.base import cached_loader_fetch, validate_date_range, validate_ohlc
from backtest.loaders.registry import register

SJ_KEY_PLACEHOLDERS = {"", "your_api_key", "your_sj_api_key"}
SJ_SECRET_PLACEHOLDERS = {"", "your_secret_key", "your_sj_secret_key"}

_CHUNK_DAYS = 29  # stays under the upstream 30-calendar-day request limit


def _date_chunks(start_date: str, end_date: str, days: int = _CHUNK_DAYS):
    """Yield (chunk_start, chunk_end) ISO date pairs covering [start_date, end_date]."""
    cur = dt.date.fromisoformat(start_date)
    last = dt.date.fromisoformat(end_date)
    step = dt.timedelta(days=days - 1)
    while cur <= last:
        chunk_end = min(cur + step, last)
        yield cur.isoformat(), chunk_end.isoformat()
        cur = chunk_end + dt.timedelta(days=1)


def _resample_minute_kbars_to_daily(frame: pd.DataFrame) -> pd.DataFrame:
    """Aggregate 1-minute OHLCV bars up to daily OHLCV."""
    daily = frame.resample("1D").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    })
    return daily.dropna(subset=["open", "high", "low", "close"])


@register
class DataLoader:
    """Shioaji-backed OHLCV loader for Taiwan equities."""

    name = "shioaji"
    markets = {"tw_equity"}
    requires_auth = True

    def is_available(self) -> bool:
        """Available when SJ_API_KEY and SJ_SEC_KEY are both set."""
        api_key = os.getenv("SJ_API_KEY", "").strip()
        sec_key = os.getenv("SJ_SEC_KEY", "").strip()
        return api_key not in SJ_KEY_PLACEHOLDERS and sec_key not in SJ_SECRET_PLACEHOLDERS

    def __init__(self) -> None:
        """Defer login until first use (see ``_ensure_logged_in``).

        ``backtest.loaders.registry.get_loader_cls_with_fallback`` constructs
        a throwaway instance just to call ``is_available()``, and
        ``runner.py`` then constructs a second instance for the actual fetch.
        Logging in eagerly here means two near-simultaneous Shioaji logins
        per backtest run, which raced on the SDK's on-disk contract-cache
        lock files (``~/.shioaji/contracts-*.parquet.lock``) and deadlocked
        in testing. Lazy login means the throwaway probe instance never
        touches the network at all.
        """
        self.api = None

    def _ensure_logged_in(self) -> None:
        """Log in to Shioaji on first use, idempotent on repeat calls."""
        if self.api is not None:
            return
        import shioaji as sj

        production = os.getenv("SJ_PRODUCTION", "false").strip().lower() in ("1", "true", "yes")
        api = sj.Shioaji(simulation=not production)
        api.login(
            api_key=os.environ["SJ_API_KEY"],
            secret_key=os.environ["SJ_SEC_KEY"],
            contracts_timeout=10000,
        )
        self.api = api

    def fetch(
        self,
        codes: List[str],
        start_date: str,
        end_date: str,
        fields: Optional[List[str]] = None,
        interval: str = "1D",
    ) -> Dict[str, pd.DataFrame]:
        """Fetch TW equity daily bars via Shioaji (resampled from 1-minute K-bars).

        Args:
            codes: Stock codes (e.g. ``2330.TW``).
            start_date: Start date (YYYY-MM-DD).
            end_date: End date (YYYY-MM-DD).
            fields: Unused today -- reserved for future enrichment (Phase 2).
            interval: Only ``1D`` is supported.

        Returns:
            Mapping code -> OHLCV DataFrame.
        """
        validate_date_range(start_date, end_date)

        if interval != "1D":
            print(f"[WARN] shioaji loader only supports 1D bars; got interval={interval!r}")
            return {}

        self._ensure_logged_in()

        result: Dict[str, pd.DataFrame] = {}
        for code in codes:
            def _fetch_one(code: str = code) -> Optional[pd.DataFrame]:
                return self._fetch_one_code(code, start_date, end_date)

            df = cached_loader_fetch(
                source=self.name,
                symbol=code,
                timeframe="1D",
                start_date=start_date,
                end_date=end_date,
                fields=None,
                fetch=_fetch_one,
            )
            if df is not None and not df.empty:
                result[code] = df

        return result

    def _fetch_one_code(
        self, code: str, start_date: str, end_date: str,
    ) -> Optional[pd.DataFrame]:
        """Pull chunked 1-minute K-bars for one symbol and resample to daily."""
        stock_id = _strip_tw_suffix(code)
        contract = self.api.Contracts.Stocks[stock_id]
        if contract is None:
            print(f"[WARN] shioaji has no contract for {code} (stock_id={stock_id})")
            return None

        chunks: List[pd.DataFrame] = []
        for chunk_start, chunk_end in _date_chunks(start_date, end_date):
            try:
                kbars = self.api.kbars(contract, start=chunk_start, end=chunk_end)
            except Exception as exc:
                print(f"[WARN] shioaji kbars failed for {code} {chunk_start}..{chunk_end}: {exc}")
                continue
            if not kbars.ts:
                continue
            chunk_df = pd.DataFrame({
                "open": kbars.Open,
                "high": kbars.High,
                "low": kbars.Low,
                "close": kbars.Close,
                "volume": kbars.Volume,
            }, index=pd.to_datetime(kbars.ts))
            chunks.append(chunk_df)

        if not chunks:
            return None

        minute_df = pd.concat(chunks).sort_index()
        for col in ["open", "high", "low", "close", "volume"]:
            minute_df[col] = pd.to_numeric(minute_df[col], errors="coerce")
        minute_df = minute_df.dropna(subset=["open", "high", "low", "close"])

        daily_df = _resample_minute_kbars_to_daily(minute_df)
        daily_df = validate_ohlc(daily_df)
        return daily_df if not daily_df.empty else None
