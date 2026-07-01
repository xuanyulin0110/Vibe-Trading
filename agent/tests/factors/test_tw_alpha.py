"""tw_alpha zoo: registration, formula correctness, missing-extras contract, look-ahead safety."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.factors.registry import Registry, RegistryError, SkipAlpha, reset_default_registry

_TW_ALPHA_IDS = [
    "tw_alpha_institutional_flow",
    "tw_alpha_margin_leverage",
    "tw_alpha_revenue_momentum",
]


@pytest.fixture()
def registry() -> Registry:
    reset_default_registry()
    return Registry()


class TestRegistration:
    def test_all_three_registered_under_tw_alpha_zoo(self, registry: Registry) -> None:
        assert registry.list(zoo="tw_alpha") == sorted(_TW_ALPHA_IDS)

    @pytest.mark.parametrize("alpha_id", _TW_ALPHA_IDS)
    def test_universe_tagged_equity_tw(self, registry: Registry, alpha_id: str) -> None:
        assert registry.get(alpha_id).meta["universe"] == ["equity_tw"]

    @pytest.mark.parametrize("alpha_id", _TW_ALPHA_IDS)
    def test_declares_its_extras(self, registry: Registry, alpha_id: str) -> None:
        assert registry.get(alpha_id).meta["extras_required"]

    def test_health_reports_zero_failures(self, registry: Registry) -> None:
        health = registry.health()
        tw_alpha_errors = [e for e in health["errors"] if e["alpha_id"].startswith("tw_alpha")]
        assert tw_alpha_errors == []


class TestMissingExtrasContract:
    """A panel missing the declared extras must SkipAlpha, not crash or silently degrade."""

    @pytest.mark.parametrize("alpha_id", _TW_ALPHA_IDS)
    def test_ohlcv_only_panel_raises_skip_alpha(self, registry: Registry, alpha_id: str) -> None:
        dates = pd.date_range("2024-01-01", periods=15, freq="D")
        cols = ["2330.TW", "2317.TW", "2454.TW"]
        close = pd.DataFrame(100.0, index=dates, columns=cols)
        ohlcv_only_panel = {"open": close, "high": close, "low": close, "close": close, "volume": close}

        with pytest.raises(SkipAlpha):
            registry.compute(alpha_id, ohlcv_only_panel)


def _panel_with_extras(
    *,
    foreign_net: dict[str, float] | None = None,
    trust_net: dict[str, float] | None = None,
    margin_usage_rate: dict[str, list[float]] | None = None,
    revenue_yoy_pct: dict[str, float] | None = None,
    n_days: int,
    codes: list[str],
) -> dict[str, pd.DataFrame]:
    """Build a minimal panel: constant-per-day extras unless a per-code day-list is given."""
    dates = pd.date_range("2024-01-01", periods=n_days, freq="D")
    close = pd.DataFrame(100.0, index=dates, columns=codes)
    panel: dict[str, pd.DataFrame] = {
        "open": close, "high": close, "low": close, "close": close, "volume": close,
    }
    if foreign_net is not None:
        panel["institutional_foreign_net"] = pd.DataFrame(
            {c: [foreign_net[c]] * n_days for c in codes}, index=dates,
        )
    if trust_net is not None:
        panel["institutional_trust_net"] = pd.DataFrame(
            {c: [trust_net[c]] * n_days for c in codes}, index=dates,
        )
    if margin_usage_rate is not None:
        panel["margin_margin_usage_rate"] = pd.DataFrame(
            {c: margin_usage_rate[c] for c in codes}, index=dates,
        )
    if revenue_yoy_pct is not None:
        panel["monthly_revenue_revenue_yoy_pct"] = pd.DataFrame(
            {c: [revenue_yoy_pct[c]] * n_days for c in codes}, index=dates,
        )
    return panel


class TestInstitutionalFlowFormula:
    def test_ranks_by_10d_mean_of_foreign_plus_trust_net(self, registry: Registry) -> None:
        codes = ["A", "B", "C"]
        panel = _panel_with_extras(
            foreign_net={"A": 60.0, "B": 150.0, "C": 20.0},
            trust_net={"A": 40.0, "B": 50.0, "C": 30.0},
            n_days=10,
            codes=codes,
        )
        # Constant daily flow -> 10d mean == the daily value: A=100, B=200, C=50.
        result = registry.compute("tw_alpha_institutional_flow", panel)
        last = result.iloc[-1]
        # Ascending sum order: C(50) < A(100) < B(200) -> pct ranks 1/3, 2/3, 1.0
        assert last["C"] == pytest.approx(1 / 3)
        assert last["A"] == pytest.approx(2 / 3)
        assert last["B"] == pytest.approx(1.0)

    def test_warmup_is_nan_before_10_days(self) -> None:
        # Calls compute() directly rather than through registry.compute(): the
        # registry rejects >95% NaN output as a degenerate-factor guard, which
        # is exactly what an all-NaN warmup period looks like on a 2-code panel
        # -- correct registry behavior, but not what this test is checking.
        from src.factors.zoo.tw_alpha.institutional_flow import compute

        codes = ["A", "B"]
        panel = _panel_with_extras(
            foreign_net={"A": 1.0, "B": 2.0}, trust_net={"A": 1.0, "B": 2.0},
            n_days=9, codes=codes,
        )
        result = compute(panel)
        assert result.isna().all().all()


class TestMarginLeverageFormula:
    def test_rising_leverage_scores_lower_than_falling(self, registry: Registry) -> None:
        codes = ["A", "B", "C"]
        panel = _panel_with_extras(
            margin_usage_rate={
                "A": [50.0, 50, 50, 50, 50, 70.0],  # +20 over 5 days -- rising
                "B": [50.0, 50, 50, 50, 50, 50.0],  # flat
                "C": [50.0, 50, 50, 50, 50, 30.0],  # -20 over 5 days -- falling
            },
            n_days=6,
            codes=codes,
        )
        result = registry.compute("tw_alpha_margin_leverage", panel)
        last = result.iloc[-1]
        # delta ascending: C(-20) < B(0) < A(+20) -> rank pct 1/3, 2/3, 1.0; factor = -rank
        assert last["C"] == pytest.approx(-1 / 3)
        assert last["B"] == pytest.approx(-2 / 3)
        assert last["A"] == pytest.approx(-1.0)
        # Falling leverage (C) must score strictly higher than rising leverage (A).
        assert last["C"] > last["A"]


class TestRevenueMomentumFormula:
    def test_ranks_by_yoy_pct_level(self, registry: Registry) -> None:
        codes = ["A", "B", "C"]
        panel = _panel_with_extras(
            revenue_yoy_pct={"A": 5.0, "B": 20.0, "C": -3.0},
            n_days=1,
            codes=codes,
        )
        result = registry.compute("tw_alpha_revenue_momentum", panel)
        last = result.iloc[-1]
        # Ascending: C(-3) < A(5) < B(20) -> pct ranks 1/3, 2/3, 1.0
        assert last["C"] == pytest.approx(1 / 3)
        assert last["A"] == pytest.approx(2 / 3)
        assert last["B"] == pytest.approx(1.0)


# ---------------------------------------------------------------- look-ahead


N_ROWS = 60
PROBE_T = 40
PERTURB_FROM = 45


def _lookahead_panel(seed: int = 0) -> dict[str, pd.DataFrame]:
    """Synthetic OHLCV + tw_alpha extras panel, long enough for all three factors'
    warmup (max 10 rows) with room before/after the look-ahead probe point."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=N_ROWS, freq="D")
    cols = [f"SYM{i}" for i in range(5)]
    close = pd.DataFrame(
        100.0 + np.cumsum(rng.normal(0.0, 1.0, size=(N_ROWS, 5)), axis=0), index=idx, columns=cols,
    ).abs() + 1.0
    return {
        "open": close, "high": close, "low": close, "close": close,
        "volume": pd.DataFrame(rng.integers(1_000, 100_000, size=(N_ROWS, 5)).astype(float), index=idx, columns=cols),
        "institutional_foreign_net": pd.DataFrame(rng.normal(0, 1000, size=(N_ROWS, 5)), index=idx, columns=cols),
        "institutional_trust_net": pd.DataFrame(rng.normal(0, 1000, size=(N_ROWS, 5)), index=idx, columns=cols),
        "margin_margin_usage_rate": pd.DataFrame(rng.uniform(0, 100, size=(N_ROWS, 5)), index=idx, columns=cols),
        "monthly_revenue_revenue_yoy_pct": pd.DataFrame(rng.normal(0, 20, size=(N_ROWS, 5)), index=idx, columns=cols),
    }


def _corrupt_future(panel: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for key, df in panel.items():
        clone = df.copy()
        for j, col in enumerate(clone.columns):
            clone.iloc[PERTURB_FROM:, j] = np.nan if j % 2 == 0 else 1e10
        out[key] = clone
    return out


@pytest.mark.parametrize("alpha_id", _TW_ALPHA_IDS)
def test_tw_alpha_has_no_lookahead(registry: Registry, alpha_id: str) -> None:
    """Corrupting rows >= PERTURB_FROM must not change the factor value at PROBE_T."""
    baseline = _lookahead_panel()
    try:
        baseline_factor = registry.compute(alpha_id, baseline)
    except (SkipAlpha, RegistryError) as exc:
        pytest.fail(f"{alpha_id}: unexpected failure on well-formed synthetic panel: {exc}")

    corrupted = _corrupt_future(baseline)
    corrupted_factor = registry.compute(alpha_id, corrupted)

    baseline_row = baseline_factor.iloc[PROBE_T].to_numpy(dtype=np.float64)
    corrupted_row = corrupted_factor.iloc[PROBE_T].to_numpy(dtype=np.float64)

    nan_baseline = np.isnan(baseline_row)
    nan_corrupted = np.isnan(corrupted_row)
    assert np.array_equal(nan_baseline, nan_corrupted), (
        f"{alpha_id}: NaN pattern at t={PROBE_T} diverges after future perturbation"
    )
    finite = ~nan_baseline
    np.testing.assert_allclose(
        baseline_row[finite], corrupted_row[finite], rtol=1e-9, atol=1e-9,
        err_msg=f"{alpha_id}: value at t={PROBE_T} changed after corrupting rows >= {PERTURB_FROM}",
    )
