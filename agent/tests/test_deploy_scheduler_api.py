"""Deploy scheduler timing + /deployments API routes."""

from __future__ import annotations

import datetime as dt
import json
from zoneinfo import ZoneInfo

import pytest

from src.deploy import scheduler as sched
from src.deploy import store as deploy_store
from src.deploy.executor import TickOutcome

TAIPEI = ZoneInfo("Asia/Taipei")


def _dep(interval="1D", market="tw_futures", sessions="day", **kw):
    defaults = dict(
        id="d1", run_id="runX", symbol="TXFR1.TWF", market=market,
        environment="paper", interval=interval, sessions=sessions,
        allocated_capital=1_000_000.0, max_order_qty=5, max_daily_orders=10,
        max_order_notional=10_000_000.0, enabled=True, created_at="",
    )
    defaults.update(kw)
    return deploy_store.Deployment(**defaults)


class TestDueBarBoundary:
    def test_daily_futures_fires_after_open_delay(self) -> None:
        dep = _dep()
        before = dt.datetime(2026, 7, 6, 8, 45, 30, tzinfo=TAIPEI)
        after = dt.datetime(2026, 7, 6, 8, 46, 30, tzinfo=TAIPEI)
        assert sched.due_bar_boundary(dep, before) is None
        boundary = sched.due_bar_boundary(dep, after)
        assert boundary is not None and boundary.date() == dt.date(2026, 7, 6)

    def test_daily_not_on_weekend(self) -> None:
        dep = _dep()
        saturday = dt.datetime(2026, 7, 4, 9, 0, tzinfo=TAIPEI)
        assert sched.due_bar_boundary(dep, saturday) is None

    def test_daily_equity_uses_0900(self) -> None:
        dep = _dep(market="tw_equity", symbol="2330.TW")
        at_0901 = dt.datetime(2026, 7, 6, 9, 1, 30, tzinfo=TAIPEI)
        assert sched.due_bar_boundary(dep, at_0901) is not None

    def test_intraday_boundary_grid(self) -> None:
        dep = _dep(interval="5m")
        # 09:05:30 -> ref 09:05:10 -> boundary 09:05 (bar 09:00-09:05 closed)
        now = dt.datetime(2026, 7, 6, 9, 5, 30, tzinfo=TAIPEI)
        boundary = sched.due_bar_boundary(dep, now)
        assert boundary == dt.datetime(2026, 7, 6, 9, 5, tzinfo=TAIPEI)

    def test_intraday_day_session_only_skips_night(self) -> None:
        dep = _dep(interval="5m", sessions="day")
        night = dt.datetime(2026, 7, 6, 22, 5, 30, tzinfo=TAIPEI)
        assert sched.due_bar_boundary(dep, night) is None

    def test_intraday_day_night_fires_in_night_session(self) -> None:
        dep = _dep(interval="5m", sessions="day_night")
        night = dt.datetime(2026, 7, 6, 22, 5, 30, tzinfo=TAIPEI)
        assert sched.due_bar_boundary(dep, night) is not None
        # ...including the after-midnight tail of the prior day's session:
        after_midnight = dt.datetime(2026, 7, 7, 2, 5, 30, tzinfo=TAIPEI)
        assert sched.due_bar_boundary(dep, after_midnight) is not None

    def test_intraday_dead_zone_between_sessions(self) -> None:
        dep = _dep(interval="5m", sessions="day_night")
        dead = dt.datetime(2026, 7, 6, 14, 30, 30, tzinfo=TAIPEI)
        assert sched.due_bar_boundary(dep, dead) is None


# --------------------------------------------------------------------------- #
# API routes
# --------------------------------------------------------------------------- #


class _StubScheduler:
    def __init__(self):
        self.flattened: list[str] = []
        self.ran: list[tuple[str, bool]] = []
        self.sessions = type(
            "S", (), {"status": lambda self: {}, "reset": lambda self, e: None,
                      "invalidate": lambda self, e, r="": None},
        )()

    def run_once(self, deployment_id: str, *, dry_run: bool) -> TickOutcome:
        self.ran.append((deployment_id, dry_run))
        return TickOutcome("dry_run", "planned", bar_ts="2026-07-03")

    def flatten(self, deployment_id: str):
        self.flattened.append(deployment_id)
        return [{"kind": "flatten"}]

    def on_deployment_toggled(self) -> None:
        pass


@pytest.fixture
def client(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient

    import api_server
    from src.deploy import api as deploy_api

    monkeypatch.setattr(deploy_store, "get_runtime_root", lambda: tmp_path)
    stub = _StubScheduler()
    monkeypatch.setattr(deploy_api, "_scheduler", stub)

    # A resolvable run dir with a config for creation flow.
    runs_root = tmp_path / "runs"
    run_dir = runs_root / "runX"
    (run_dir / "code").mkdir(parents=True)
    (run_dir / "config.json").write_text(json.dumps({
        "source": "shioaji_futures", "codes": ["TXFR1.TWF"],
        "start_date": "2026-06-01", "end_date": "2026-07-06", "interval": "1D",
    }), encoding="utf-8")
    monkeypatch.setenv("VIBE_TRADING_ALLOWED_RUN_ROOTS", str(runs_root))

    return TestClient(api_server.app), stub


def _create_body(**kw):
    body = dict(
        run_id="runX", environment="paper", allocated_capital=1_000_000,
        max_order_qty=5, max_daily_orders=10, max_order_notional=10_000_000,
    )
    body.update(kw)
    return body


class TestDeploymentRoutes:
    def test_create_list_toggle_delete(self, client) -> None:
        http, _ = client
        created = http.post("/deployments", json=_create_body())
        assert created.status_code == 201, created.text
        dep_id = created.json()["id"]
        assert created.json()["symbol"] == "TXFR1.TWF"
        assert created.json()["market"] == "tw_futures"

        listing = http.get("/deployments").json()
        assert len(listing["deployments"]) == 1

        assert http.post(f"/deployments/{dep_id}/start").json()["enabled"] is True
        # enabled deployments cannot be deleted
        assert http.delete(f"/deployments/{dep_id}").status_code == 400
        http.post(f"/deployments/{dep_id}/stop")
        assert http.delete(f"/deployments/{dep_id}").status_code == 200

    def test_live_requires_typed_confirmation_and_ca(self, client) -> None:
        http, _ = client
        r = http.post("/deployments", json=_create_body(environment="live"))
        assert r.status_code == 400
        assert "confirm_symbol" in r.json()["detail"]
        r2 = http.post(
            "/deployments", json=_create_body(environment="live", confirm_symbol="TXFR1.TWF"),
        )
        assert r2.status_code == 400
        assert "ca_path" in r2.json()["detail"] or "CA" in r2.json()["detail"]

    def test_duplicate_symbol_rejected(self, client) -> None:
        http, _ = client
        assert http.post("/deployments", json=_create_body()).status_code == 201
        r = http.post("/deployments", json=_create_body())
        assert r.status_code == 400
        assert "already exists" in r.json()["detail"]

    def test_kill_switch_roundtrip(self, client) -> None:
        http, _ = client
        assert http.post("/deployments/kill-switch", json={"engaged": True}).json()["engaged"] is True
        assert http.get("/deployments").json()["kill_switch"] is True
        http.post("/deployments/kill-switch", json={"engaged": False})
        assert deploy_store.kill_switch_engaged() is False

    def test_flatten_requires_exact_symbol(self, client) -> None:
        http, stub = client
        dep_id = http.post("/deployments", json=_create_body()).json()["id"]
        r = http.post(f"/deployments/{dep_id}/flatten", json={"confirm_symbol": "WRONG"})
        assert r.status_code == 400
        assert stub.flattened == []
        ok = http.post(f"/deployments/{dep_id}/flatten", json={"confirm_symbol": "TXFR1.TWF"})
        assert ok.status_code == 200
        assert stub.flattened == [dep_id]

    def test_run_once_defaults_to_dry_run(self, client) -> None:
        http, stub = client
        dep_id = http.post("/deployments", json=_create_body()).json()["id"]
        r = http.post(f"/deployments/{dep_id}/run-once")
        assert r.status_code == 200
        assert stub.ran == [(dep_id, True)]

    def test_events_route_not_shadowed_by_id_route(self) -> None:
        """Regression (2026-07-06): /deployments/events 404'd because the
        {deployment_id} catch-all was registered first and swallowed
        "events" as an id -- the events router must be included first.

        Asserted on route ORDER (matching is first-registered-wins) rather
        than by streaming the endpoint: the SSE generator is infinite and
        TestClient blocks on it. FastAPI's lazy include keeps included
        routers as _IncludedRouter entries wrapping original_router, so
        flatten those to recover path order.
        """
        import api_server

        paths: list[str] = []
        for route in api_server.app.router.routes:
            inner = getattr(route, "original_router", None)
            if inner is not None:
                paths.extend(getattr(r, "path", "") for r in inner.routes)
            else:
                paths.append(getattr(route, "path", ""))
        events_idx = paths.index("/deployments/events")
        id_idx = paths.index("/deployments/{deployment_id}")
        assert events_idx < id_idx

    def test_history_and_equity_empty_ok(self, client) -> None:
        http, _ = client
        dep_id = http.post("/deployments", json=_create_body()).json()["id"]
        assert http.get(f"/deployments/{dep_id}/history").json() == {"ticks": [], "fills": []}
        eq = http.get(f"/deployments/{dep_id}/equity").json()
        assert eq["allocated_capital"] == 1_000_000
        assert eq["points"] == []
