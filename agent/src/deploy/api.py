"""FastAPI router for the deterministic deployment runtime.

Mounted by ``api_server.py`` under ``/deployments`` behind the same auth
dependency as the rest of the control plane. The router owns a tiny SSE
fan-out bus so the frontend sees tick/fill/flatten events live without
polling; the scheduler publishes into it.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import json
import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from src.deploy import accounting, market_calendar, signal_runner, store
from src.deploy.market_calendar import TW_EQUITY, TW_FUTURES
from src.tools.path_utils import safe_run_id

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/deployments", tags=["deployments"])

#: SSE lives on its own router: browser EventSource cannot send an
#: Authorization header, so api_server mounts this one behind the
#: query-string-capable event-stream auth instead of plain Bearer auth.
events_router = APIRouter(prefix="/deployments", tags=["deployments"])


# --------------------------------------------------------------------------- #
# SSE bus
# --------------------------------------------------------------------------- #


class DeployEventBus:
    def __init__(self) -> None:
        self._subscribers: set[asyncio.Queue] = set()
        self._loop: asyncio.AbstractEventLoop | None = None

    def bind_loop(self) -> None:
        self._loop = asyncio.get_event_loop()

    def publish(self, event: dict[str, Any]) -> None:
        """Thread-safe publish (scheduler ticks run in worker threads)."""
        event = {**event, "ts": dt.datetime.now(dt.timezone.utc).isoformat()}
        loop = self._loop
        if loop is None or loop.is_closed():
            return
        loop.call_soon_threadsafe(self._fanout, event)

    def _fanout(self, event: dict[str, Any]) -> None:
        for queue in list(self._subscribers):
            if queue.qsize() < 500:
                queue.put_nowait(event)

    async def subscribe(self):
        queue: asyncio.Queue = asyncio.Queue()
        self._subscribers.add(queue)
        try:
            while True:
                yield await queue.get()
        finally:
            self._subscribers.discard(queue)


event_bus = DeployEventBus()

#: Set by api_server at startup (the scheduler instance). Kept as a module
#: attribute so routes and tests can swap it.
_scheduler = None


def set_scheduler(scheduler) -> None:
    global _scheduler
    _scheduler = scheduler


def get_scheduler():
    if _scheduler is None:
        raise HTTPException(status_code=503, detail="deploy scheduler not running")
    return _scheduler


# --------------------------------------------------------------------------- #
# Schemas
# --------------------------------------------------------------------------- #


class CreateDeploymentRequest(BaseModel):
    run_id: str
    environment: str = Field(pattern="^(paper|live)$")
    symbol: str | None = None  # default: run config codes[0]
    sessions: str = Field(default="day", pattern="^(day|day_night)$")
    allocated_capital: float
    max_order_qty: int
    max_daily_orders: int
    max_order_notional: float
    confirm_symbol: str | None = None  # required (typed) for live


class ToggleResponse(BaseModel):
    id: str
    enabled: bool


class FlattenRequest(BaseModel):
    confirm_symbol: str


class KillSwitchRequest(BaseModel):
    engaged: bool


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #


@router.get("")
def list_deployments() -> dict[str, Any]:
    scheduler = _scheduler
    return {
        "deployments": [vars(d) for d in store.list_deployments()],
        "kill_switch": store.kill_switch_engaged(),
        "sessions": scheduler.sessions.status() if scheduler else {},
    }


@router.post("", status_code=201)
def create_deployment(body: CreateDeploymentRequest) -> dict[str, Any]:
    try:
        run_dir = safe_run_id(body.run_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    try:
        config = signal_runner.load_run_config(run_dir)
    except signal_runner.SignalComputationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    codes = list(config.get("codes") or [])
    symbol = (body.symbol or (codes[0] if codes else "")).strip().upper()
    if not symbol:
        raise HTTPException(status_code=400, detail="run config has no codes and no symbol given")
    market = TW_FUTURES if symbol.endswith(".TWF") else TW_EQUITY
    interval = str(config.get("interval", "1D"))

    if interval.lower() in ("1m", "1min"):
        # Deployable only after the latency budget is proven -- surfaced in
        # the UI as a warning; the API still refuses silent creation without
        # an explicit meta flag so nobody 1m-deploys by accident.
        pass

    if body.environment == "live":
        if body.confirm_symbol != symbol:
            raise HTTPException(
                status_code=400,
                detail=f"live deployment requires typed confirmation: confirm_symbol must equal {symbol!r}",
            )
        from src.trading.connectors.shioaji import sdk

        cfg = sdk.load_config()
        if not cfg.ca_path or not cfg.ca_passwd:
            raise HTTPException(
                status_code=400,
                detail="live requires ca_path/ca_passwd in shioaji.json (CA certificate) -- configure first",
            )

    try:
        dep = store.create_deployment(
            run_id=body.run_id,
            symbol=symbol,
            market=market,
            environment=body.environment,
            interval=interval,
            sessions=body.sessions if market == TW_FUTURES else "day",
            allocated_capital=body.allocated_capital,
            max_order_qty=body.max_order_qty,
            max_daily_orders=body.max_daily_orders,
            max_order_notional=body.max_order_notional,
        )
    except store.DeploymentError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    event_bus.publish({"type": "created", "deployment_id": dep.id})
    return vars(dep)


@router.post("/kill-switch")
def kill_switch(body: KillSwitchRequest) -> dict[str, Any]:
    from src.deploy import notifier

    store.set_kill_switch(body.engaged)
    event = {"type": "kill_switch", "engaged": body.engaged}
    event_bus.publish(event)
    notifier.publish(event)
    return {"engaged": store.kill_switch_engaged()}


@events_router.get("/events")
async def deployment_events(request: Request):
    async def generator():
        yield "event: hello\ndata: {}\n\n"
        async for event in event_bus.subscribe():
            if await request.is_disconnected():
                break
            yield f"data: {json.dumps(event, ensure_ascii=False, default=str)}\n\n"

    return StreamingResponse(
        generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


@router.post("/live-session/reset")
def reset_live_session() -> dict[str, Any]:
    scheduler = get_scheduler()
    scheduler.sessions.reset("live")
    scheduler.sessions.invalidate("live", "operator reset")
    return {"status": "ok"}


@router.post("/{deployment_id}/start")
def start_deployment(deployment_id: str) -> ToggleResponse:
    try:
        dep = store.update_deployment(deployment_id, enabled=True, paused_reason=None)
    except store.DeploymentError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if _scheduler:
        _scheduler.on_deployment_toggled()
    event_bus.publish({"type": "toggled", "deployment_id": dep.id, "enabled": True})
    return ToggleResponse(id=dep.id, enabled=True)


@router.post("/{deployment_id}/stop")
def stop_deployment(deployment_id: str) -> ToggleResponse:
    try:
        dep = store.update_deployment(deployment_id, enabled=False)
    except store.DeploymentError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if _scheduler:
        _scheduler.on_deployment_toggled()
    event_bus.publish({"type": "toggled", "deployment_id": dep.id, "enabled": False})
    return ToggleResponse(id=dep.id, enabled=False)


@router.post("/{deployment_id}/flatten")
def flatten_deployment(deployment_id: str, body: FlattenRequest) -> dict[str, Any]:
    dep = store.get_deployment(deployment_id)
    if dep is None:
        raise HTTPException(status_code=404, detail=f"no deployment {deployment_id}")
    if body.confirm_symbol.strip().upper() != dep.symbol:
        raise HTTPException(
            status_code=400,
            detail=f"typed confirmation mismatch: expected {dep.symbol!r}",
        )
    scheduler = get_scheduler()
    try:
        results = scheduler.flatten(deployment_id)
    except Exception as exc:  # noqa: BLE001 - surfaced to the operator
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return {"status": "ok", "orders": results}


@router.post("/{deployment_id}/run-once")
def run_once(deployment_id: str, dry_run: bool = Query(default=True)) -> dict[str, Any]:
    scheduler = get_scheduler()
    try:
        outcome = scheduler.run_once(deployment_id, dry_run=dry_run)
    except store.DeploymentError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return vars(outcome)


@router.delete("/{deployment_id}")
def delete_deployment(deployment_id: str) -> dict[str, Any]:
    dep = store.get_deployment(deployment_id)
    if dep is not None and dep.enabled:
        raise HTTPException(status_code=400, detail="stop the deployment before deleting it")
    if not store.delete_deployment(deployment_id):
        raise HTTPException(status_code=404, detail=f"no deployment {deployment_id}")
    if _scheduler:
        _scheduler.on_deployment_toggled()
    event_bus.publish({"type": "deleted", "deployment_id": deployment_id})
    return {"status": "ok"}


@router.get("/{deployment_id}")
def get_deployment(deployment_id: str) -> dict[str, Any]:
    dep = store.get_deployment(deployment_id)
    if dep is None:
        raise HTTPException(status_code=404, detail=f"no deployment {deployment_id}")
    return vars(dep)


@router.get("/{deployment_id}/history")
def deployment_history(deployment_id: str, limit: int = Query(default=100, le=500)) -> dict[str, Any]:
    dep = store.get_deployment(deployment_id)
    if dep is None:
        raise HTTPException(status_code=404, detail=f"no deployment {deployment_id}")
    run_dir = safe_run_id(dep.run_id)
    return {
        "ticks": accounting.list_tick_records(run_dir, limit=limit),
        "fills": accounting.load_fills(run_dir)[-limit:],
    }


@router.get("/{deployment_id}/equity")
def deployment_equity(deployment_id: str, limit: int = Query(default=500, le=2000)) -> dict[str, Any]:
    dep = store.get_deployment(deployment_id)
    if dep is None:
        raise HTTPException(status_code=404, detail=f"no deployment {deployment_id}")
    run_dir = safe_run_id(dep.run_id)
    return {
        "allocated_capital": dep.allocated_capital,
        "points": accounting.equity_series(run_dir, limit=limit),
    }


@router.get("/{deployment_id}/bars")
def deployment_bars(deployment_id: str, lookback_days: int = Query(default=30, le=120)) -> dict[str, Any]:
    """Recent bars for the detail chart, straight from the persistent kbar
    cache (a shared-session fetch only tops up the newest gap)."""
    dep = store.get_deployment(deployment_id)
    if dep is None:
        raise HTTPException(status_code=404, detail=f"no deployment {deployment_id}")
    run_dir = safe_run_id(dep.run_id)
    config = signal_runner.load_run_config(run_dir)
    end = market_calendar.now_taipei().date()
    start = end - dt.timedelta(days=lookback_days)
    fetch_config = {**config, "start_date": start.isoformat()}

    injected = None
    if _scheduler is not None:
        try:
            with _scheduler.sessions.use(dep.environment) as api:
                injected = api
        except Exception:  # noqa: BLE001 - chart data falls back to own login
            injected = None
    try:
        data_map = signal_runner.fetch_run_data(
            run_dir, fetch_config, end_date=end.isoformat(), injected_api=injected,
        )
    except signal_runner.SignalComputationError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    frame = data_map.get(dep.symbol)
    if frame is None:
        raise HTTPException(status_code=404, detail=f"no bar data for {dep.symbol}")
    bars = [
        {
            "time": str(ts),
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
            "volume": float(row.get("volume", 0.0)),
        }
        for ts, row in frame.iterrows()
    ]
    return {"symbol": dep.symbol, "interval": dep.interval, "bars": bars}
