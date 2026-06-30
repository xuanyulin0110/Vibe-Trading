"""End-to-end smoke test: TW equity symbol routes through the full backtest
pipeline (market detection -> engine dispatch -> run_backtest -> metrics)
using synthetic data, with no network/finlab dependency.

This proves the Phase 1 wiring (``_market_hooks``, ``_create_market_engine``,
``TWEquityEngine``) is connected end-to-end. Validating against *real*
finlab data additionally requires a live FINLAB_API_TOKEN and is exercised
separately once a token is available.
"""

from __future__ import annotations

import pandas as pd

from backtest.engines.tw_equity import TWEquityEngine
from backtest.runner import _create_market_engine, _detect_market


class TestTWEquityDispatch:
    def test_detect_market_routes_tw_suffix(self) -> None:
        assert _detect_market("2330.TW") == "tw_equity"
        assert _detect_market("6488.TWO") == "tw_equity"

    def test_create_market_engine_returns_tw_equity_engine(self) -> None:
        engine = _create_market_engine("finlab", {"initial_cash": 1_000_000}, ["2330.TW"])
        assert isinstance(engine, TWEquityEngine)


class TestTWEquityFullBacktest:
    def test_full_backtest_runs_end_to_end(self, tmp_path) -> None:
        dates = pd.bdate_range("2024-01-02", periods=20)
        bars = pd.DataFrame(
            {
                "open": [600.0 + i for i in range(20)],
                "high": [602.0 + i for i in range(20)],
                "low": [598.0 + i for i in range(20)],
                "close": [601.0 + i for i in range(20)],
                "volume": [10_000_000] * 20,
            },
            index=dates,
        )

        class FakeLoader:
            def fetch(self, *args, **kwargs):
                return {"2330.TW": bars.copy()}

        class SignalEngine:
            def generate(self, data_map):
                frame = data_map["2330.TW"]
                # Simple always-long signal so trades actually execute.
                return {"2330.TW": pd.Series(1.0, index=frame.index)}

        config = {
            "codes": ["2330.TW"],
            "start_date": "2024-01-02",
            "end_date": "2024-01-31",
            "source": "finlab",
            "initial_cash": 1_000_000,
        }

        engine = _create_market_engine("finlab", config, config["codes"])
        assert isinstance(engine, TWEquityEngine)

        metrics = engine.run_backtest(config, FakeLoader(), SignalEngine(), tmp_path)

        # Non-degenerate output: backtest actually traded and produced metrics.
        assert "sharpe" in metrics
        assert "total_return" in metrics
        assert (tmp_path / "artifacts" / "equity.csv").exists()
        assert (tmp_path / "artifacts" / "trades.csv").exists()

        trades_csv = pd.read_csv(tmp_path / "artifacts" / "trades.csv")
        assert len(trades_csv) > 0, "always-long signal should have produced at least one trade"
