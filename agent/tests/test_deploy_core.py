"""Deterministic deploy runtime: calendar / store / contracts / sizing /
signal fidelity / accounting / executor.

The fidelity invariant test here is the load-bearing one: the live signal
runner's last-complete-bar weight must equal the weight the backtest engine
holds on the very next bar (``_align``'s shift(1) semantics), computed from
the same data by the same signal_engine.py.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from src.deploy import accounting, contracts, executor, market_calendar, signal_runner, sizing
from src.deploy import store as deploy_store

TAIPEI = ZoneInfo("Asia/Taipei")


# --------------------------------------------------------------------------- #
# market_calendar
# --------------------------------------------------------------------------- #


class TestMarketCalendar:
    def test_weekend_and_holiday(self) -> None:
        assert not market_calendar.is_trading_day(dt.date(2026, 7, 4))  # Saturday
        assert not market_calendar.is_trading_day(dt.date(2026, 1, 1))  # holiday
        assert market_calendar.is_trading_day(dt.date(2026, 7, 6))  # Monday

    def test_equity_session(self) -> None:
        mk = lambda h, m: dt.datetime(2026, 7, 6, h, m, tzinfo=TAIPEI)
        assert not market_calendar.session_open_now("tw_equity", mk(8, 59))
        assert market_calendar.session_open_now("tw_equity", mk(9, 0))
        assert market_calendar.session_open_now("tw_equity", mk(13, 29))
        assert not market_calendar.session_open_now("tw_equity", mk(13, 30))

    def test_futures_day_session(self) -> None:
        mk = lambda h, m: dt.datetime(2026, 7, 6, h, m, tzinfo=TAIPEI)
        assert market_calendar.session_open_now("tw_futures", mk(8, 45))
        assert not market_calendar.session_open_now("tw_futures", mk(13, 45))

    def test_futures_night_session_crosses_midnight(self) -> None:
        # Monday 23:00 -> in-session; Tuesday 02:00 -> still the Monday night session.
        mon_night = dt.datetime(2026, 7, 6, 23, 0, tzinfo=TAIPEI)
        tue_early = dt.datetime(2026, 7, 7, 2, 0, tzinfo=TAIPEI)
        assert market_calendar.session_open_now("tw_futures", mon_night, include_night=True)
        assert market_calendar.session_open_now("tw_futures", tue_early, include_night=True)
        assert not market_calendar.session_open_now("tw_futures", mon_night, include_night=False)
        # Saturday early morning IS the tail of Friday's night session.
        sat_early = dt.datetime(2026, 7, 11, 2, 0, tzinfo=TAIPEI)
        assert market_calendar.session_open_now("tw_futures", sat_early, include_night=True)
        # Sunday early morning is not (Saturday isn't a trading day).
        sun_early = dt.datetime(2026, 7, 12, 2, 0, tzinfo=TAIPEI)
        assert not market_calendar.session_open_now("tw_futures", sun_early, include_night=True)

    def test_settlement_third_wednesday(self) -> None:
        assert market_calendar.taifex_settlement_date(2026, 7) == dt.date(2026, 7, 15)
        assert market_calendar.taifex_settlement_date(2026, 7).weekday() == 2


# --------------------------------------------------------------------------- #
# store
# --------------------------------------------------------------------------- #


@pytest.fixture
def tmp_store(monkeypatch, tmp_path):
    monkeypatch.setattr(deploy_store, "get_runtime_root", lambda: tmp_path)
    return tmp_path


def _create(symbol="TXFR1.TWF", market="tw_futures", **kw):
    defaults = dict(
        run_id="run1", symbol=symbol, market=market, environment="paper",
        interval="1D", allocated_capital=1_000_000.0,
        max_order_qty=5, max_daily_orders=10, max_order_notional=10_000_000.0,
    )
    defaults.update(kw)
    return deploy_store.create_deployment(**defaults)


class TestStore:
    def test_create_and_roundtrip(self, tmp_store) -> None:
        dep = _create()
        loaded = deploy_store.get_deployment(dep.id)
        assert loaded is not None
        assert loaded.symbol == "TXFR1.TWF"
        assert loaded.enabled is False

    def test_duplicate_symbol_rejected(self, tmp_store) -> None:
        _create()
        with pytest.raises(deploy_store.DeploymentError, match="already exists"):
            _create()

    def test_missing_caps_rejected(self, tmp_store) -> None:
        with pytest.raises(deploy_store.DeploymentError, match="safety caps"):
            _create(symbol="MXFR1.TWF", max_order_qty=0)

    def test_environment_immutable(self, tmp_store) -> None:
        dep = _create()
        with pytest.raises(deploy_store.DeploymentError, match="immutable"):
            deploy_store.update_deployment(dep.id, environment="live")

    def test_equity_night_session_rejected(self, tmp_store) -> None:
        with pytest.raises(deploy_store.DeploymentError, match="night"):
            _create(symbol="2330.TW", market="tw_equity", sessions="day_night")

    def test_kill_switch_persists(self, tmp_store) -> None:
        assert not deploy_store.kill_switch_engaged()
        deploy_store.set_kill_switch(True)
        assert deploy_store.kill_switch_engaged()
        deploy_store.set_kill_switch(False)
        assert not deploy_store.kill_switch_engaged()


# --------------------------------------------------------------------------- #
# contracts
# --------------------------------------------------------------------------- #


class _Contract:
    def __init__(self, code: str, delivery_month: str):
        self.code = code
        self.delivery_month = delivery_month


class _Category:
    def __init__(self, items):
        self._items = items

    def __iter__(self):
        return iter(self._items)


class _FakeApi:
    def __init__(self, months=("202607", "202608", "202609")):
        items = [_Contract(f"TXF{m}", m) for m in months]
        self.Contracts = type("C", (), {"Futures": type("F", (), {"TXF": _Category(items)})()})()


class TestContracts:
    def test_product_and_continuous(self) -> None:
        assert contracts.product_of("TXFR1.TWF") == "TXF"
        assert contracts.is_continuous("TXFR1.TWF")
        assert not contracts.is_continuous("TXF202607.TWF")

    def test_resolution_before_settlement(self) -> None:
        api = _FakeApi()
        resolved = contracts.resolve_order_contract(api, "TXFR1.TWF", dt.date(2026, 7, 6))
        assert resolved.delivery_month == "202607"

    def test_settlement_day_routes_to_next_month(self) -> None:
        api = _FakeApi()
        settlement = market_calendar.taifex_settlement_date(2026, 7)
        resolved = contracts.resolve_order_contract(api, "TXFR1.TWF", settlement)
        assert resolved.delivery_month == "202608"

    def test_rollover_planned_for_expiring_position(self) -> None:
        api = _FakeApi()
        settlement = market_calendar.taifex_settlement_date(2026, 7)
        positions = [{"symbol": "TXF202607", "quantity": 2, "direction": 1}]
        plan = contracts.plan_rollover(api, "TXFR1.TWF", positions, settlement)
        assert plan is not None
        assert plan.expiring_code == "TXF202607"
        assert plan.quantity == 2
        assert plan.next.delivery_month == "202608"

    def test_no_rollover_before_settlement(self) -> None:
        api = _FakeApi()
        positions = [{"symbol": "TXF202607", "quantity": 2, "direction": 1}]
        assert contracts.plan_rollover(api, "TXFR1.TWF", positions, dt.date(2026, 7, 6)) is None


# --------------------------------------------------------------------------- #
# sizing
# --------------------------------------------------------------------------- #


class TestSizing:
    def test_equity_board_lots(self) -> None:
        r = sizing.target_quantity(
            market="tw_equity", symbol="2330.TW", weight=1.0,
            allocated_capital=3_000_000, price=1200,
        )
        assert r.target_qty == 2000  # 2.5 lots floor -> 2 lots

    def test_equity_zero_lot_has_reason(self) -> None:
        r = sizing.target_quantity(
            market="tw_equity", symbol="2330.TW", weight=1.0,
            allocated_capital=1_000_000, price=1200,
        )
        assert r.target_qty == 0
        assert "can't afford" in r.reason

    def test_equity_short_flattens(self) -> None:
        r = sizing.target_quantity(
            market="tw_equity", symbol="2330.TW", weight=-1.0,
            allocated_capital=3_000_000, price=1200,
        )
        assert r.target_qty == 0
        assert "short" in r.reason

    def test_futures_contracts_and_short(self) -> None:
        # TMF: 20/point -> 23000*20 = 460k notional/contract
        long = sizing.target_quantity(
            market="tw_futures", symbol="TMFR1.TWF", weight=1.0,
            allocated_capital=1_000_000, price=23000,
        )
        assert long.target_qty == 2
        short = sizing.target_quantity(
            market="tw_futures", symbol="TMFR1.TWF", weight=-1.0,
            allocated_capital=1_000_000, price=23000,
        )
        assert short.target_qty == -2

    def test_futures_margin_cap(self) -> None:
        # TXF margin 636k/contract: notional affords 1 contract at high
        # leverage-free sizing only if capital covers margin too.
        r = sizing.target_quantity(
            market="tw_futures", symbol="TXFR1.TWF", weight=1.0,
            allocated_capital=5_000_000, price=23000,
        )
        # notional sizing: 5M / 4.6M = 1 contract; margin 636k fits 5M.
        assert r.target_qty == 1
        assert r.margin_required == 636_000.0


# --------------------------------------------------------------------------- #
# signal_runner: completeness + fidelity invariant
# --------------------------------------------------------------------------- #


_ENGINE_SOURCE = '''
import pandas as pd
from typing import Dict


class SignalEngine:
    def generate(self, data_map: Dict[str, pd.DataFrame]) -> Dict[str, pd.Series]:
        signals = {}
        for code, df in data_map.items():
            close = df["close"]
            ma3 = close.rolling(3, min_periods=3).mean()
            sig = pd.Series(0.0, index=df.index)
            sig[close > ma3] = 1.0
            sig[close < ma3] = -1.0
            signals[code] = sig.fillna(0.0)
        return signals
'''


def _make_run_dir(tmp_path: Path, interval: str = "1D") -> Path:
    run_dir = tmp_path / "runX"
    (run_dir / "code").mkdir(parents=True)
    (run_dir / "code" / "signal_engine.py").write_text(_ENGINE_SOURCE, encoding="utf-8")
    (run_dir / "config.json").write_text(
        json.dumps({
            "source": "shioaji_futures", "codes": ["TXFR1.TWF"],
            "start_date": "2026-06-01", "end_date": "2026-07-06",
            "interval": interval, "initial_cash": 1_000_000,
        }),
        encoding="utf-8",
    )
    return run_dir


def _daily_frame(n: int = 12) -> pd.DataFrame:
    idx = pd.bdate_range("2026-06-15", periods=n)
    base = np.linspace(22000, 23100, n)
    return pd.DataFrame(
        {"open": base, "high": base + 50, "low": base - 50, "close": base + 10,
         "volume": np.full(n, 1000.0)},
        index=idx,
    )


class TestSignalFidelity:
    def test_completeness_drops_partial_daily_bar(self, tmp_path) -> None:
        bars = _daily_frame(5)  # last bar dated 2026-06-19 (Fri)
        # "Now" = the last bar's morning, before the day session closes:
        now = dt.datetime(2026, 6, 19, 8, 46, tzinfo=TAIPEI)
        out = signal_runner.drop_incomplete_bars(bars, "1D", "tw_futures", now)
        assert out.index[-1] == pd.Timestamp("2026-06-18")
        # After the close it is complete:
        later = dt.datetime(2026, 6, 19, 13, 46, tzinfo=TAIPEI)
        out2 = signal_runner.drop_incomplete_bars(bars, "1D", "tw_futures", later)
        assert out2.index[-1] == pd.Timestamp("2026-06-19")

    def test_completeness_intraday_bucket(self, tmp_path) -> None:
        idx = pd.date_range("2026-07-06 09:00", periods=4, freq="5min")
        bars = pd.DataFrame({"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0,
                             "volume": 1.0}, index=idx)
        # now = 09:16 -> the 09:15 bucket (covering to 09:20) is incomplete.
        now = dt.datetime(2026, 7, 6, 9, 16, tzinfo=TAIPEI)
        out = signal_runner.drop_incomplete_bars(bars, "5m", "tw_futures", now)
        assert out.index[-1] == pd.Timestamp("2026-07-06 09:10")

    def test_fidelity_invariant_matches_backtest_align(self, tmp_path) -> None:
        """Live weight at last complete bar == backtest _align weight on the NEXT bar."""
        from backtest.engines.base import _align

        run_dir = _make_run_dir(tmp_path)
        full = _daily_frame(12)
        symbol = "TXFR1.TWF"

        # Backtest view: all 12 bars.
        module = signal_runner.load_signal_engine(run_dir)
        signals = module.generate({symbol: full})
        _, _, pos, _ = _align({symbol: full}, signals, [symbol])

        # Live view: tick happens during bar 12 -> completeness leaves 11 bars.
        live_data = {symbol: full.iloc[:11]}
        result = signal_runner.compute_signal(
            run_dir, symbol, "tw_futures",
            now=dt.datetime(2026, 7, 6, 8, 46, tzinfo=TAIPEI),
            data_map=live_data,
        )
        # The weight the backtest holds during bar 12 is pos.iloc[11].
        assert result.weight == pytest.approx(float(pos[symbol].iloc[11]))
        assert result.bar_ts == full.index[10]


# --------------------------------------------------------------------------- #
# accounting
# --------------------------------------------------------------------------- #


class TestAccounting:
    def test_fill_dedupe(self, tmp_path) -> None:
        fill = {"exchange_seq": "X1", "side": "buy", "quantity": 1, "price": 100}
        assert accounting.append_fill(tmp_path, fill)
        assert not accounting.append_fill(tmp_path, fill)
        assert len(accounting.load_fills(tmp_path)) == 1

    def test_futures_fifo_realized_pnl_with_multiplier(self, tmp_path) -> None:
        fills = [
            {"side": "buy", "quantity": 2, "price": 22000},
            {"side": "sell", "quantity": 1, "price": 22100},
        ]
        snap = accounting.compute_equity(
            market="tw_futures", symbol="TMFR1.TWF", allocated_capital=1_000_000,
            fills=fills, mark_price=22200,
        )
        # realized: +100 points x 20/point = 2000; unrealized: +200 x 20 = 4000
        assert snap.realized_pnl == pytest.approx(2000.0)
        assert snap.unrealized_pnl == pytest.approx(4000.0)
        assert snap.position_qty == 1

    def test_equity_pnl_and_fees(self, tmp_path) -> None:
        fills = [
            {"side": "buy", "quantity": 1000, "price": 1000.0},
            {"side": "sell", "quantity": 1000, "price": 1010.0},
        ]
        snap = accounting.compute_equity(
            market="tw_equity", symbol="2330.TW", allocated_capital=2_000_000,
            fills=fills, mark_price=1010.0,
        )
        assert snap.realized_pnl == pytest.approx(10_000.0)
        assert snap.position_qty == 0
        assert snap.fees > 0  # commission both legs + sell-side tax

    def test_tick_record_two_phase(self, tmp_path) -> None:
        accounting.write_tick_record(tmp_path, "2026-07-06", {"phase": "intent"})
        assert accounting.read_tick_record(tmp_path, "2026-07-06")["phase"] == "intent"
        accounting.write_tick_record(tmp_path, "2026-07-06", {"phase": "final"})
        assert accounting.read_tick_record(tmp_path, "2026-07-06")["phase"] == "final"


# --------------------------------------------------------------------------- #
# executor
# --------------------------------------------------------------------------- #


def _deployment(**kw) -> deploy_store.Deployment:
    defaults = dict(
        id="d1", run_id="runX", symbol="TMFR1.TWF", market="tw_futures",
        environment="paper", interval="1D", sessions="day",
        allocated_capital=1_000_000.0, max_order_qty=5, max_daily_orders=10,
        max_order_notional=20_000_000.0, enabled=True, created_at="",
    )
    defaults.update(kw)
    return deploy_store.Deployment(**defaults)


def _signal(weight: float, close: float = 23000.0) -> signal_runner.SignalResult:
    return signal_runner.SignalResult(
        symbol="TMFR1.TWF", bar_ts=pd.Timestamp("2026-07-03"),
        weight=weight, close=close, bars_evaluated=100, elapsed_seconds=0.1,
    )


@pytest.fixture
def exec_env(monkeypatch, tmp_path):
    """Stubbed sdk + kill switch off; returns (run_dir, calls dict)."""
    monkeypatch.setattr(deploy_store, "get_runtime_root", lambda: tmp_path)
    calls: dict = {"orders": [], "cancels": []}
    monkeypatch.setattr(
        executor.sdk, "get_positions",
        lambda cfg=None, api=None: {"status": "ok", "positions": calls.get("positions", [])},
    )
    monkeypatch.setattr(
        executor.sdk, "get_open_orders",
        lambda cfg=None, api=None, **kw: {"status": "ok", "open_orders": calls.get("open", [])},
    )

    def _place(cfg=None, **kwargs):
        calls["orders"].append(kwargs)
        return {"status": "ok", "order_id": f"O{len(calls['orders'])}",
                "filled_qty": kwargs["quantity"]}

    monkeypatch.setattr(executor.sdk, "place_order", _place)
    monkeypatch.setattr(
        executor.sdk, "cancel_order",
        lambda cfg=None, order_id="", **kw: calls["cancels"].append(order_id) or {"status": "ok"},
    )
    run_dir = tmp_path / "runX"
    run_dir.mkdir()
    return run_dir, calls


class TestExecutor:
    def test_long_entry_places_order(self, exec_env) -> None:
        run_dir, calls = exec_env
        outcome = executor.run_tick(
            _deployment(), run_dir, session_api=_FakeApiTMF(),
            signal_result=_signal(1.0), quote_fn=lambda a, s: {},
            now=dt.datetime(2026, 7, 6, 8, 46, tzinfo=TAIPEI),
        )
        assert outcome.status == "ok"
        assert calls["orders"][0]["side"] == "buy"
        assert calls["orders"][0]["quantity"] == 2  # TMF 2 contracts per sizing test
        assert calls["orders"][0]["time_in_force"] == "ioc"
        record = accounting.read_tick_record(run_dir, str(_signal(1.0).bar_ts))
        assert record["phase"] == "final"
        assert "equity_snapshot" in record

    def test_converged_position_no_order(self, exec_env) -> None:
        run_dir, calls = exec_env
        calls["positions"] = [{"symbol": "TMF202607", "quantity": 2, "direction": 1}]
        # plan_rollover consults contracts; give the executor a fake api with categories
        api = _FakeApiTMF()
        outcome = executor.run_tick(
            _deployment(), run_dir, session_api=api,
            signal_result=_signal(1.0), quote_fn=lambda a, s: {},
            now=dt.datetime(2026, 7, 6, 8, 46, tzinfo=TAIPEI),
        )
        assert outcome.status == "ok"
        assert calls["orders"] == []

    def test_equity_position_lots_convert_to_shares(self, exec_env) -> None:
        """Shioaji equity positions are in LOTS (confirmed live 2026-07-06):
        holding quantity=2 means 2,000 shares -- a 2,000-share target must
        read as converged, not as a 1,998-share odd-lot diff."""
        run_dir, calls = exec_env
        calls["positions"] = [{"symbol": "2330", "quantity": 2, "direction": 1}]
        dep = _deployment(
            symbol="2330.TW", market="tw_equity", allocated_capital=5_000_000.0,
            max_order_qty=2000, max_order_notional=6_000_000.0,
        )
        sig = signal_runner.SignalResult(
            symbol="2330.TW", bar_ts=pd.Timestamp("2026-07-03"),
            weight=1.0, close=2450.0, bars_evaluated=100, elapsed_seconds=0.1,
        )
        outcome = executor.run_tick(
            dep, run_dir, session_api=object(), signal_result=sig,
            quote_fn=lambda a, s: {},
            now=dt.datetime(2026, 7, 6, 9, 5, tzinfo=TAIPEI),
        )
        assert outcome.status == "ok"
        assert outcome.current_qty == 2000
        assert calls["orders"] == []  # converged

    def test_bar_idempotency(self, exec_env) -> None:
        run_dir, calls = exec_env
        args = dict(session_api=_FakeApiTMF(), signal_result=_signal(1.0),
                    quote_fn=lambda a, s: {},
                    now=dt.datetime(2026, 7, 6, 8, 46, tzinfo=TAIPEI))
        executor.run_tick(_deployment(), run_dir, **args)
        outcome = executor.run_tick(_deployment(), run_dir, **args)
        assert outcome.status == "skipped"
        assert len(calls["orders"]) == 1

    def test_kill_switch_blocks(self, exec_env) -> None:
        run_dir, calls = exec_env
        deploy_store.set_kill_switch(True)
        outcome = executor.run_tick(
            _deployment(), run_dir, session_api=object(), signal_result=_signal(1.0),
        )
        assert outcome.status == "blocked"
        assert calls["orders"] == []

    def test_safety_cap_rejects_whole_order(self, exec_env) -> None:
        run_dir, calls = exec_env
        dep = _deployment(max_order_qty=1)  # target 2 contracts > cap 1
        outcome = executor.run_tick(
            dep, run_dir, session_api=object(), signal_result=_signal(1.0),
            quote_fn=lambda a, s: {},
            now=dt.datetime(2026, 7, 6, 8, 46, tzinfo=TAIPEI),
        )
        assert outcome.status == "blocked"
        assert "max_order_qty" in outcome.reason
        assert calls["orders"] == []  # rejected whole, never resized

    def test_limit_guard_blocks_buy_at_limit_up(self, exec_env) -> None:
        run_dir, calls = exec_env
        outcome = executor.run_tick(
            _deployment(), run_dir, session_api=object(), signal_result=_signal(1.0),
            quote_fn=lambda a, s: {"change_rate": 10.0},
            now=dt.datetime(2026, 7, 6, 8, 46, tzinfo=TAIPEI),
        )
        assert outcome.status == "blocked"
        assert "blocked_limit" in outcome.reason
        assert calls["orders"] == []

    def test_incomplete_intent_with_resting_orders_blocks(self, exec_env) -> None:
        run_dir, calls = exec_env
        sig = _signal(1.0)
        accounting.write_tick_record(run_dir, str(sig.bar_ts), {"phase": "intent"})
        calls["open"] = [{"symbol": "TMF202607", "order_id": "O9"}]
        outcome = executor.run_tick(
            _deployment(), run_dir, session_api=object(), signal_result=sig,
            quote_fn=lambda a, s: {},
            now=dt.datetime(2026, 7, 6, 8, 46, tzinfo=TAIPEI),
        )
        assert outcome.status == "blocked"
        assert "reconcile" in outcome.reason

    def test_partial_fill_refills_bounded(self, exec_env, monkeypatch) -> None:
        run_dir, calls = exec_env

        def _partial(cfg=None, **kwargs):
            calls["orders"].append(kwargs)
            return {"status": "ok", "order_id": f"O{len(calls['orders'])}", "filled_qty": 1}

        monkeypatch.setattr(executor.sdk, "place_order", _partial)
        dep = _deployment(allocated_capital=3_000_000.0, max_order_qty=10)  # 6 TMF contracts target
        outcome = executor.run_tick(
            dep, run_dir, session_api=_FakeApiTMF(), signal_result=_signal(1.0),
            quote_fn=lambda a, s: {},
            now=dt.datetime(2026, 7, 6, 8, 46, tzinfo=TAIPEI),
        )
        assert outcome.status == "ok"
        # initial + up to 2 refills, 1 fill each
        assert len(calls["orders"]) == 3
        assert any(o.get("kind") == "partial_gaveup" for o in outcome.orders)

    def test_rollover_on_settlement_day(self, exec_env) -> None:
        run_dir, calls = exec_env
        settlement = market_calendar.taifex_settlement_date(2026, 7)
        calls["positions"] = [{"symbol": "TMF202607", "quantity": 2, "direction": 1}]
        api = _FakeApiTMF()
        outcome = executor.run_tick(
            _deployment(), run_dir, session_api=api, signal_result=_signal(1.0),
            quote_fn=lambda a, s: {},
            now=dt.datetime(settlement.year, settlement.month, settlement.day, 13, 0, tzinfo=TAIPEI),
        )
        assert outcome.status == "ok"
        legs = [(o["symbol"], o["side"]) for o in calls["orders"]]
        assert ("TMF202607.TWF", "sell") in legs  # close expiring
        assert ("TMF202608.TWF", "buy") in legs  # reopen next month

    def test_flatten(self, exec_env) -> None:
        run_dir, calls = exec_env
        calls["positions"] = [{"symbol": "TMF202607", "quantity": 3, "direction": 1}]
        api = _FakeApiTMF()
        results = executor.flatten(
            _deployment(), run_dir, session_api=api,
            now=dt.datetime(2026, 7, 6, 10, 0, tzinfo=TAIPEI),
        )
        assert calls["orders"][0]["side"] == "sell"
        assert calls["orders"][0]["quantity"] == 3
        assert any(r.get("kind") in ("flatten", "signal") or "response" in r for r in results)


class _FakeApiTMF:
    def __init__(self, months=("202607", "202608", "202609")):
        items = [_Contract(f"TMF{m}", m) for m in months]
        self.Contracts = type("C", (), {"Futures": type("F", (), {"TMF": _Category(items)})()})()
