"""Shioaji loader for Taiwan index futures (TAIFEX: TXF/MXF/TMF) OHLCV.

Same 1-minute-kbar-chunk-and-resample mechanism as ``shioaji_loader.py`` (the
equity loader) via the shared ``_shioaji_kbars`` helper; the only difference is
contract resolution. Equities live under ``api.Contracts.Stocks[code]``;
futures live under the nested ``api.Contracts.Futures.<PRODUCT>.<contract>``
(e.g. ``api.Contracts.Futures.TXF.TXFR1``). Symbols use the project's ``.TWF``
suffix convention: ``TXFR1.TWF`` -> product ``TXF``, contract ``TXFR1``.

finlab has no TAIFEX price data (only 期貨三大法人 chip data), so there is no
finlab fallback for futures prices -- ``FALLBACK_CHAINS["tw_futures"]`` is
``["shioaji", "local"]``.

Historical futures coverage on Shioaji starts 2020-03-22. Market-data
timestamps are already Taiwan wall-clock time -- do not add +8h.
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
from backtest.loaders.base import validate_date_range, validate_ohlc
from backtest.loaders.registry import register
from backtest.loaders.shioaji_loader import SJ_KEY_PLACEHOLDERS, SJ_SECRET_PLACEHOLDERS

#: Known TAIFEX index-futures product categories under ``Contracts.Futures``.
_KNOWN_PRODUCTS = ("TXF", "MXF", "TMF")


def _split_tw_futures_symbol(code: str) -> tuple[str, str]:
    """Return (product_category, contract_code) for a ``.TWF`` futures symbol.

    ``TXFR1.TWF`` -> ``("TXF", "TXFR1")``; ``MXFR2.TWF`` -> ``("MXF", "MXFR2")``.
    Falls back to the leading 3 letters as the category when the prefix is not
    a known product.
    """
    contract = code.split(".")[0].upper()
    for product in _KNOWN_PRODUCTS:
        if contract.startswith(product):
            return product, contract
    return contract[:3], contract


@register
class DataLoader:
    """Shioaji-backed OHLCV loader for Taiwan index futures."""

    name = "shioaji_futures"
    markets = {"tw_futures"}
    requires_auth = True

    def is_available(self) -> bool:
        """Available when SJ_API_KEY and SJ_SEC_KEY are both set."""
        api_key = os.getenv("SJ_API_KEY", "").strip()
        sec_key = os.getenv("SJ_SEC_KEY", "").strip()
        return api_key not in SJ_KEY_PLACEHOLDERS and sec_key not in SJ_SECRET_PLACEHOLDERS

    def __init__(self) -> None:
        """Defer login until first use (see ``shioaji_loader.DataLoader`` for the
        deadlock rationale — the registry constructs a throwaway probe instance
        just to call ``is_available()``)."""
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
        """Fetch TAIFEX futures bars via Shioaji (resampled from 1-minute K-bars).

        Args:
            codes: Futures codes with the ``.TWF`` suffix (e.g. ``TXFR1.TWF``).
            start_date: Start date (YYYY-MM-DD).
            end_date: End date (YYYY-MM-DD).
            fields: Unused today.
            interval: 1m/5m/15m/30m/1H/4H/1D (default ``1D``).

        Returns:
            Mapping code -> OHLCV DataFrame.
        """
        validate_date_range(start_date, end_date)

        if not is_supported_interval(interval):
            print(f"[WARN] shioaji futures loader: unsupported interval {interval!r}")
            return {}

        # Suppress for the whole login+fetch lifetime, not just around the
        # login() call: Shioaji's native contract-subscription handshake
        # continues on a background thread for a bit after login() itself
        # returns (confirmed empirically -- a "Subscribe or Unsubscribe ok"
        # event still leaked to stdout with only the login() call wrapped).
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
                        print(f"[WARN] shioaji futures fetch failed for {code}: {exc}")
                        continue
                    if df is not None and not df.empty:
                        result[code] = df

        return result

    def _resolve_contract(self, code: str):
        """Resolve a ``.TWF`` symbol to a Shioaji futures contract object, or None."""
        product, contract_code = _split_tw_futures_symbol(code)
        category = getattr(self.api.Contracts.Futures, product, None)
        if category is None:
            print(f"[WARN] shioaji has no futures category {product} for {code}")
            return None
        contract = getattr(category, contract_code, None)
        if contract is None:
            # Some SDK builds expose contracts via mapping access rather than attribute.
            try:
                contract = category[contract_code]
            except Exception:
                contract = None
        if contract is None:
            print(f"[WARN] shioaji has no futures contract {contract_code} for {code}")
        return contract

    def _fetch_one_code(
        self, code: str, start_date: str, end_date: str, interval: str,
    ) -> Optional[pd.DataFrame]:
        """Pull 1-minute K-bars for one futures contract (gap-aware cached) and resample."""
        contract = self._resolve_contract(code)
        if contract is None:
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
