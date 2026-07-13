"""finlab loader for Taiwan equities (TWSE/TPEx) daily OHLCV.

finlab's SDK returns whole-market wide tables per field (one DataFrame per
field, indexed by date, columned by bare stock id) rather than answering a
per-symbol REST call, so this loader fetches each OHLCV field table once per
``fetch()`` call (cached on the instance) and slices per requested code --
the opposite shape from a per-code loader like ``tushare.py``, but
``fetch()``'s external contract (``{code: DataFrame}``) is unchanged.

Only the free tier is required for historical backtesting; near-real-time
data needs a paid finlab VIP subscription (unrelated to this loader -- the
SDK call surface is the same either way).

NOTE: the ``price:*`` field-name strings below match finlab's documented
dataset naming convention at the time this loader was written. Confirm them
against the installed ``finlab`` package's current catalogue (or
``finlab.data.search("price")``) the first time this loader is exercised
against a real token, since finlab does not pin a stable API contract.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

import pandas as pd

from backtest.loaders._symbol_utils import _strip_tw_suffix
from backtest.loaders.base import cached_loader_fetch, validate_date_range, validate_ohlc
from backtest.loaders.registry import register

FINLAB_TOKEN_PLACEHOLDERS = {"", "your-finlab-token"}

_FIELD_MAP = {
    "open": "price:開盤價",
    "high": "price:最高價",
    "low": "price:最低價",
    "close": "price:收盤價",
    "volume": "price:成交股數",
}


@register
class DataLoader:
    """finlab-backed OHLCV loader for Taiwan equities."""

    name = "finlab"
    markets = {"tw_equity"}
    requires_auth = True

    def is_available(self) -> bool:
        """Available when FINLAB_API_TOKEN is set."""
        from src.config.accessor import get_env_config

        return get_env_config().data.finlab_api_token.strip() not in FINLAB_TOKEN_PLACEHOLDERS

    def __init__(self) -> None:
        """Defer login until first use (see ``_ensure_logged_in``).

        ``backtest.loaders.registry.get_loader_cls_with_fallback`` constructs a
        throwaway instance just to call ``is_available()``. Logging in eagerly
        here meant that probe instance -- and every other construction, even
        with no token configured -- called ``finlab.login()`` unconditionally.
        With an empty/placeholder token, finlab's SDK falls back to an
        interactive browser-auth flow that prints "Opening browser for
        login..." to stdout and blocks waiting for it: fatal for a headless
        MCP subprocess, since MCP stdio requires stdout to carry only
        JSON-RPC frames (mirrors the Shioaji loaders' lazy-login rationale).
        """
        self._logged_in = False
        self._field_cache: Dict[str, pd.DataFrame] = {}

    def _ensure_logged_in(self) -> None:
        """Log in to finlab on first use, idempotent on repeat calls."""
        if self._logged_in:
            return
        from src.config.accessor import get_env_config

        token = get_env_config().data.finlab_api_token
        if token.strip() in FINLAB_TOKEN_PLACEHOLDERS:
            raise RuntimeError("FINLAB_API_TOKEN is not configured")
        import finlab

        finlab.login(token)
        self._logged_in = True

    def fetch(
        self,
        codes: List[str],
        start_date: str,
        end_date: str,
        fields: Optional[List[str]] = None,
        interval: str = "1D",
    ) -> Dict[str, pd.DataFrame]:
        """Fetch TW equity daily bars via finlab.

        Args:
            codes: Stock codes (e.g. ``2330.TW``).
            start_date: Start date (YYYY-MM-DD).
            end_date: End date (YYYY-MM-DD).
            fields: Unused -- finlab has no per-code "extra columns" fetch of its
                own (unlike tushare's extra_fields). 三大法人/融資融券/月營收/財報 etc.
                enrichment is a separate code path: config.json's fundamental_fields,
                handled by backtest.engines.base._maybe_enrich_fundamentals via
                FinlabFundamentalProvider, applied after fetch() returns.
            interval: Only ``1D`` is supported; finlab's free tier is
                daily-resolution only.

        Returns:
            Mapping code -> OHLCV DataFrame.
        """
        validate_date_range(start_date, end_date)

        if interval != "1D":
            print(f"[WARN] finlab only supports 1D bars; got interval={interval!r}")
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

    def _field_table(self, field_key: str) -> pd.DataFrame:
        """Return the whole-market wide table for one field, fetched once per instance."""
        if field_key not in self._field_cache:
            from finlab import data

            self._field_cache[field_key] = data.get(field_key)
        return self._field_cache[field_key]

    def _fetch_one_code(
        self, code: str, start_date: str, end_date: str,
    ) -> Optional[pd.DataFrame]:
        """Build one symbol's OHLCV frame by slicing the whole-market field tables."""
        stock_id = _strip_tw_suffix(code)
        columns: Dict[str, pd.Series] = {}
        for ohlcv_col, field_key in _FIELD_MAP.items():
            try:
                table = self._field_table(field_key)
            except Exception as exc:
                print(f"[WARN] finlab field fetch failed for {field_key}: {exc}")
                return None
            if stock_id not in table.columns:
                print(f"[WARN] finlab has no column for {code} (stock_id={stock_id}) in {field_key}")
                return None
            columns[ohlcv_col] = table[stock_id]

        df = pd.DataFrame(columns)
        df.index = pd.to_datetime(df.index)
        df = df.loc[(df.index >= pd.Timestamp(start_date)) & (df.index <= pd.Timestamp(end_date))]
        df = df.sort_index()
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["open", "high", "low", "close"])
        df = validate_ohlc(df)
        return df if not df.empty else None
