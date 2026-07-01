"""Shioaji loader for Taiwan equities (TWSE/TPEx) OHLCV.

Shioaji is SinoPac's official broker API. It only exposes 1-minute K-bars
(``api.kbars()``), no coarser native resolution, so this loader pulls 1-minute
bars (chunked at <=29 days per the upstream 30-day request limit) and resamples
up to the requested ``interval`` (1m/5m/15m/30m/1H/4H/1D). The chunking,
resampling, and stale-lock cleanup live in the shared ``_shioaji_kbars`` helper
so the equity loader, futures loader, and trading connector do not each keep a
private copy.

Historical coverage starts 2020-03-02 for stocks; requests before that date
return empty/partial chunks (the caller's runtime fallback to ``finlab`` picks
up the rest of the loader chain when this loader yields nothing).

Login uses ``SJ_API_KEY``/``SJ_SEC_KEY`` only -- market data (kbars/ticks/
snapshots) needs no CA certificate, only order placement does. Defaults to
``simulation=True`` (no CA, works on a simulation-only account); set
``SJ_PRODUCTION=true`` to use the production login instead, matching Shioaji's
own env var convention.

Market-data timestamps from Shioaji are already Taiwan wall-clock time -- do
not add +8h (see MARKET_DATA.md's "Market Data Time Handling" section).
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional

import pandas as pd

from backtest.loaders._shioaji_kbars import (
    FETCH_WORKERS,
    clear_stale_shioaji_locks,
    fetch_minute_kbars_cached,
    is_supported_interval,
    resample_kbars,
    suppress_native_stdout,
)
from backtest.loaders._symbol_utils import _strip_tw_suffix
from backtest.loaders.base import validate_date_range, validate_ohlc
from backtest.loaders.registry import register

SJ_KEY_PLACEHOLDERS = {"", "your_api_key", "your_sj_api_key"}
SJ_SECRET_PLACEHOLDERS = {"", "your_secret_key", "your_sj_secret_key"}


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

        ``backtest.loaders.registry.get_loader_cls_with_fallback`` constructs a
        throwaway instance just to call ``is_available()``, and ``runner.py``
        then constructs a second instance for the actual fetch. Logging in
        eagerly here means two near-simultaneous Shioaji logins per backtest
        run, which raced on the SDK's on-disk contract-cache lock files
        (``~/.shioaji/contracts-*.parquet.lock``) and deadlocked in testing.
        Lazy login means the throwaway probe instance never touches the network.
        """
        self.api = None

    def _ensure_logged_in(self) -> None:
        """Log in to Shioaji on first use, idempotent on repeat calls."""
        if self.api is not None:
            return
        import shioaji as sj

        clear_stale_shioaji_locks()
        production = os.getenv("SJ_PRODUCTION", "false").strip().lower() in ("1", "true", "yes")
        with suppress_native_stdout():
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
        """Fetch TW equity bars via Shioaji (resampled from 1-minute K-bars).

        Args:
            codes: Stock codes (e.g. ``2330.TW``).
            start_date: Start date (YYYY-MM-DD).
            end_date: End date (YYYY-MM-DD).
            fields: Unused today -- reserved for future enrichment.
            interval: 1m/5m/15m/30m/1H/4H/1D (default ``1D``).

        Returns:
            Mapping code -> OHLCV DataFrame.
        """
        validate_date_range(start_date, end_date)

        if not is_supported_interval(interval):
            print(f"[WARN] shioaji loader: unsupported interval {interval!r}")
            return {}

        # Suppress for the whole login+fetch lifetime, not just around the
        # login() call: Shioaji's native contract-subscription handshake can
        # continue on a background thread for a bit after login() itself
        # returns (confirmed empirically on the futures loader -- a "Subscribe
        # or Unsubscribe ok" event leaked with only the login() call wrapped).
        # Safe to nest under the per-call _shioaji_call_gate used inside the
        # parallel fetch below -- the gate's lock means only one thread is
        # ever mid-native-call at a time, so this single outer entry (held by
        # the calling thread only) never races with it.
        with suppress_native_stdout():
            self._ensure_logged_in()

            result: Dict[str, pd.DataFrame] = {}
            with ThreadPoolExecutor(max_workers=FETCH_WORKERS) as pool:
                futures = {
                    pool.submit(self._fetch_one_code, code, start_date, end_date, interval): code
                    for code in codes
                }
                for future in as_completed(futures):
                    code = futures[future]
                    try:
                        df = future.result()
                    except Exception as exc:  # noqa: BLE001 - one bad code must not sink the batch
                        print(f"[WARN] shioaji fetch failed for {code}: {exc}")
                        continue
                    if df is not None and not df.empty:
                        result[code] = df

        return result

    def _fetch_one_code(
        self, code: str, start_date: str, end_date: str, interval: str,
    ) -> Optional[pd.DataFrame]:
        """Pull 1-minute K-bars for one symbol (gap-aware cached) and resample to ``interval``."""
        stock_id = _strip_tw_suffix(code)
        contract = self.api.Contracts.Stocks[stock_id]
        if contract is None:
            print(f"[WARN] shioaji has no contract for {code} (stock_id={stock_id})")
            return None

        minute_df = fetch_minute_kbars_cached(
            self.api, contract, source=self.name, symbol=code,
            start_date=start_date, end_date=end_date,
        )
        if minute_df.empty:
            return None

        bars = resample_kbars(minute_df, interval)
        bars = validate_ohlc(bars)
        return bars if not bars.empty else None
