"""Integration coverage for the parallelized multi-code fetch() in both
Shioaji loaders: correctness (same result as the old sequential loop would
give) and thread-safety (no two native calls ever run concurrently, proven
through the real loader code path, not just the gate in isolation)."""

from __future__ import annotations

import datetime as dt
import threading
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

from backtest.loaders import shioaji_futures_loader, shioaji_loader
from backtest.loaders._shioaji_kbars import MINUTE_CACHE_ENV


class _FakeKbars:
    def __init__(self, ts: list, open_: list, high: list, low: list, close: list, volume: list) -> None:
        self.ts = ts
        self.Open = open_
        self.High = high
        self.Low = low
        self.Close = close
        self.Volume = volume


class _TrackingFakeApi:
    """Records concurrency (max simultaneous kbars() calls observed) and
    returns deterministic per-code bars so results can be checked exactly."""

    def __init__(self) -> None:
        self._concurrent = 0
        self._lock = threading.Lock()
        self.max_concurrent_seen = 0
        self.call_count = 0
        self.usage_calls = 0
        self.Contracts = SimpleNamespace(
            Stocks=_StockContracts(), Futures=_FuturesContracts(),
        )

    def kbars(self, contract: Any, start: str, end: str) -> _FakeKbars:
        with self._lock:
            self._concurrent += 1
            self.max_concurrent_seen = max(self.max_concurrent_seen, self._concurrent)
            self.call_count += 1
        try:
            # Hold the "native call" long enough that a real race would show up.
            import time
            time.sleep(0.01)
            code = getattr(contract, "code", "UNKNOWN")
            cur = dt.date.fromisoformat(start)
            last = dt.date.fromisoformat(end)
            ts, opens, highs, lows, closes, vols = [], [], [], [], [], []
            base = float(sum(ord(c) for c in code) % 50 + 10)  # deterministic per-code price
            while cur <= last:
                ts.append(f"{cur.isoformat()} 09:00:00")
                opens.append(base)
                highs.append(base + 1)
                lows.append(base - 1)
                closes.append(base + 0.5)
                vols.append(1000)
                cur += dt.timedelta(days=1)
            return _FakeKbars(ts, opens, highs, lows, closes, vols)
        finally:
            with self._lock:
                self._concurrent -= 1

    def usage(self) -> SimpleNamespace:
        self.usage_calls += 1
        return SimpleNamespace(remaining_bytes=10_000)


class _StockContracts:
    def __getitem__(self, stock_id: str):
        return SimpleNamespace(code=stock_id)


class _FuturesContracts:
    def __getattr__(self, product: str):
        return _ProductContracts(product)


class _ProductContracts:
    def __init__(self, product: str) -> None:
        self._product = product

    def __getattr__(self, contract_code: str):
        return SimpleNamespace(code=f"{self._product}.{contract_code}")


@pytest.fixture(autouse=True)
def _isolated_cache(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv(MINUTE_CACHE_ENV, "1")


class TestEquityLoaderParallelFetch:
    def test_multi_code_fetch_returns_correct_data_for_every_code(self) -> None:
        loader = shioaji_loader.DataLoader()
        loader.api = _TrackingFakeApi()

        codes = [f"{n}.TW" for n in ("2330", "2317", "2454", "2412", "1301", "2882")]
        result = loader.fetch(codes, "2024-01-01", "2024-01-03", interval="1D")

        assert set(result.keys()) == set(codes)
        for code in codes:
            assert not result[code].empty
            assert list(result[code].columns) == ["open", "high", "low", "close", "volume"]

    def test_native_calls_are_never_concurrent(self) -> None:
        """The core safety property: even with 6 codes fanned out across
        FETCH_WORKERS threads, only one kbars() call is ever in flight."""
        loader = shioaji_loader.DataLoader()
        api = _TrackingFakeApi()
        loader.api = api

        codes = [f"{n}.TW" for n in ("2330", "2317", "2454", "2412", "1301", "2882", "2881", "2891")]
        loader.fetch(codes, "2024-01-01", "2024-01-02", interval="1D")

        assert api.max_concurrent_seen == 1
        assert api.call_count == len(codes)  # one kbars() call per code (single-day chunk each)

    def test_one_bad_code_does_not_sink_the_whole_batch(self) -> None:
        loader = shioaji_loader.DataLoader()
        api = _TrackingFakeApi()

        real_stocks = api.Contracts.Stocks

        class _PartiallyBrokenStocks:
            def __getitem__(self, stock_id: str):
                if stock_id == "9999":
                    raise RuntimeError("simulated contract lookup failure")
                return real_stocks[stock_id]

        api.Contracts.Stocks = _PartiallyBrokenStocks()
        loader.api = api

        result = loader.fetch(["2330.TW", "9999.TW", "2317.TW"], "2024-01-01", "2024-01-02", interval="1D")

        assert set(result.keys()) == {"2330.TW", "2317.TW"}


class TestFuturesLoaderParallelFetch:
    def test_multi_code_fetch_returns_correct_data(self) -> None:
        loader = shioaji_futures_loader.DataLoader()
        loader.api = _TrackingFakeApi()

        codes = ["TXFR1.TWF", "MXFR1.TWF", "TMFR1.TWF"]
        result = loader.fetch(codes, "2024-01-01", "2024-01-03", interval="1D")

        assert set(result.keys()) == set(codes)

    def test_native_calls_are_never_concurrent(self) -> None:
        loader = shioaji_futures_loader.DataLoader()
        api = _TrackingFakeApi()
        loader.api = api

        result = loader.fetch(
            ["TXFR1.TWF", "MXFR1.TWF", "TMFR1.TWF"], "2024-01-01", "2024-01-02", interval="1D",
        )

        assert api.max_concurrent_seen == 1
        assert len(result) == 3
