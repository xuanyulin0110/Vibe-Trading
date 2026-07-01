"""tw50 universe wiring in alpha_bench_tool.py: _wide_from_fetched extras, _load_tw_panel."""

from __future__ import annotations

import pandas as pd
import pytest

from src.tools.alpha_bench_tool import (
    _TW50_FALLBACK_CODES,
    _UNIVERSE_TAG,
    _load_tw_panel,
    _wide_from_fetched,
)


class TestWideFromFetchedExtraFields:
    def test_extra_fields_stacked_alongside_ohlcv(self) -> None:
        dates = pd.to_datetime(["2024-01-01", "2024-01-02"])
        fetched = {
            "A.TW": pd.DataFrame(
                {"open": [1.0, 2.0], "high": [1.0, 2.0], "low": [1.0, 2.0],
                 "close": [1.0, 2.0], "volume": [10.0, 20.0], "chip_x": [100.0, 200.0]},
                index=dates,
            ),
            "B.TW": pd.DataFrame(
                {"open": [3.0, 4.0], "high": [3.0, 4.0], "low": [3.0, 4.0],
                 "close": [3.0, 4.0], "volume": [30.0, 40.0], "chip_x": [300.0, 400.0]},
                index=dates,
            ),
        }
        panel = _wide_from_fetched(fetched, include_amount=False, extra_fields=("chip_x",))
        assert "chip_x" in panel
        assert list(panel["chip_x"].columns) == ["A.TW", "B.TW"]
        assert panel["chip_x"].loc[dates[1], "A.TW"] == 200.0
        assert panel["chip_x"].loc[dates[1], "B.TW"] == 400.0
        # close panel unaffected by the extra field being present.
        assert panel["close"].loc[dates[0], "B.TW"] == 3.0

    def test_extra_field_absent_from_every_code_is_omitted(self) -> None:
        dates = pd.to_datetime(["2024-01-01"])
        fetched = {
            "A.TW": pd.DataFrame(
                {"open": [1.0], "high": [1.0], "low": [1.0], "close": [1.0], "volume": [10.0]},
                index=dates,
            ),
        }
        panel = _wide_from_fetched(fetched, include_amount=False, extra_fields=("chip_x",))
        assert "chip_x" not in panel

    def test_default_extra_fields_is_backward_compatible(self) -> None:
        """No extra_fields argument -> identical shape to the pre-existing OHLCV-only behavior."""
        dates = pd.to_datetime(["2024-01-01"])
        fetched = {
            "A.TW": pd.DataFrame(
                {"open": [1.0], "high": [1.0], "low": [1.0], "close": [1.0], "volume": [10.0]},
                index=dates,
            ),
        }
        panel = _wide_from_fetched(fetched, include_amount=False)
        assert set(panel) == {"open", "high", "low", "close", "volume"}


class TestUniverseTag:
    def test_tw50_tagged_equity_tw(self) -> None:
        assert _UNIVERSE_TAG["tw50"] == "equity_tw"


class TestLoadTwPanel:
    def test_assembles_ohlcv_and_chip_extras(self, monkeypatch: pytest.MonkeyPatch) -> None:
        dates = pd.to_datetime(["2024-01-01", "2024-01-02"])
        codes = [f"{c}.TW" for c in _TW50_FALLBACK_CODES]

        fetched = {
            code: pd.DataFrame(
                {"open": [10.0, 11.0], "high": [10.0, 11.0], "low": [10.0, 11.0],
                 "close": [10.0, 11.0], "volume": [1.0, 1.0]},
                index=dates,
            )
            for code in codes
        }

        class _FakeLoader:
            def fetch(self, requested_codes, start_date, end_date):
                assert sorted(requested_codes) == sorted(codes)
                return fetched

        monkeypatch.setattr(
            "backtest.loaders.registry.resolve_loader", lambda market: _FakeLoader(),
        )

        def _fake_enrich(price_map, provider, fields_by_table):
            enriched = {code: frame.copy() for code, frame in price_map.items()}
            for code, frame in enriched.items():
                frame["institutional_foreign_net"] = [1.0, 2.0]
            return enriched

        monkeypatch.setattr(
            "backtest.loaders.finlab_fundamentals.enrich_price_frames_with_finlab_fundamentals",
            _fake_enrich,
        )
        monkeypatch.setattr(
            "backtest.loaders.finlab_fundamentals.FinlabFundamentalProvider",
            lambda: object(),
        )

        panel = _load_tw_panel("2024-01-01", "2024-01-02")
        assert panel["close"].shape == (2, len(codes))
        assert "institutional_foreign_net" in panel

    def test_enrichment_failure_degrades_to_ohlcv_only(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Chip data is best-effort -- OHLCV-only alphas must still get a usable panel."""
        dates = pd.to_datetime(["2024-01-01"])
        codes = [f"{c}.TW" for c in _TW50_FALLBACK_CODES]
        fetched = {
            code: pd.DataFrame(
                {"open": [10.0], "high": [10.0], "low": [10.0], "close": [10.0], "volume": [1.0]},
                index=dates,
            )
            for code in codes
        }

        class _FakeLoader:
            def fetch(self, requested_codes, start_date, end_date):
                return fetched

        monkeypatch.setattr(
            "backtest.loaders.registry.resolve_loader", lambda market: _FakeLoader(),
        )

        def _raise(*a, **kw):
            raise RuntimeError("finlab down")

        monkeypatch.setattr(
            "backtest.loaders.finlab_fundamentals.FinlabFundamentalProvider", _raise,
        )

        panel = _load_tw_panel("2024-01-01", "2024-01-01")
        assert panel["close"].shape == (1, len(codes))
        assert "institutional_foreign_net" not in panel

    def test_empty_fetch_returns_empty_panel(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class _FakeLoader:
            def fetch(self, requested_codes, start_date, end_date):
                return {}

        monkeypatch.setattr(
            "backtest.loaders.registry.resolve_loader", lambda market: _FakeLoader(),
        )
        panel = _load_tw_panel("2024-01-01", "2024-01-01")
        assert panel == {}
