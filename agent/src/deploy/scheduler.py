"""Bar-boundary scheduler for deterministic deployments.

A single asyncio loop (one per process -- guarded by an in-process singleton
AND an exclusive file lock, so an accidental multi-worker uvicorn config can
never double-fire orders) wakes every few seconds, asks each enabled
deployment "has a new bar boundary passed?", and runs at most one tick per
deployment per bar. Durable idempotency lives in the executor's tick records
(existence of the bar's file); the scheduler's in-memory watermark is just an
optimization.

Timing model (Asia/Taipei throughout):

* ``1D``: one tick per trading day at day-session open + delay (futures
  08:46, equities 09:01). The signal bar is YESTERDAY's completed bar --
  next-bar-open semantics, same as the backtest.
* Intraday: ticks at each interval boundary + delay while the deployment's
  sessions are open (day, or day+night for futures). The boundary grid is
  midnight-aligned floor arithmetic, which matches pandas' default resample
  alignment -- the executor's completeness filter is the authority on which
  bar actually gets used, so grid drift can never mis-trade; it can only
  waste a wake-up.
* Catch-up on restart: the normal check IS the catch-up -- if we're
  in-session and the current boundary has no record, it runs. Bars missed
  while down stay missed (position converges on the next tick; that is the
  point of stateless diff execution).
* Data lag: if a tick came back "skipped" because the freshest complete bar
  was already executed (feed hasn't published the new bar yet), retry once
  after a short pause, then record the lag and move on.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import fcntl
import logging
from pathlib import Path
from typing import Any, Callable

from src.deploy import executor, market_calendar, store
from src.deploy.connection import SessionManager, SessionUnavailableError
from src.deploy.market_calendar import TAIPEI, TW_FUTURES
from src.tools.path_utils import safe_run_id

logger = logging.getLogger(__name__)

_WAKE_SECONDS = 10
_TICK_DELAY_SECONDS = 60  # 1D: after open; intraday: after bar close
_INTRADAY_DELAY_SECONDS = 20
_DATA_LAG_RETRY_SECONDS = 30

_INTERVAL_MINUTES = {"1m": 1, "5m": 5, "15m": 15, "30m": 30, "1h": 60, "4h": 240}


def _norm_interval(interval: str) -> str:
    return str(interval).strip().lower().replace("1day", "1d").replace("1min", "1m")


def due_bar_boundary(deployment: store.Deployment, now: dt.datetime) -> dt.datetime | None:
    """The bar boundary a tick right now would serve, or None if not due.

    Returns a tz-aware Taipei datetime identifying the boundary (used as the
    scheduler's in-memory watermark key).
    """
    local = now.astimezone(TAIPEI)
    interval = _norm_interval(deployment.interval)

    if interval == "1d":
        if not market_calendar.is_trading_day(local.date()):
            return None
        open_t = market_calendar.day_session_open(deployment.market)
        tick_at = dt.datetime.combine(local.date(), open_t, tzinfo=TAIPEI) + dt.timedelta(
            seconds=_TICK_DELAY_SECONDS
        )
        close_t = market_calendar.day_session_close(deployment.market)
        session_close = dt.datetime.combine(local.date(), close_t, tzinfo=TAIPEI)
        if tick_at <= local < session_close:
            return dt.datetime.combine(local.date(), open_t, tzinfo=TAIPEI)
        return None

    minutes = _INTERVAL_MINUTES.get(interval)
    if minutes is None:
        return None
    include_night = deployment.market == TW_FUTURES and deployment.sessions == "day_night"
    # The boundary that most recently passed (with the landing delay):
    ref = local - dt.timedelta(seconds=_INTRADAY_DELAY_SECONDS)
    floored_minutes = (ref.hour * 60 + ref.minute) // minutes * minutes
    boundary = dt.datetime.combine(
        ref.date(), dt.time(floored_minutes // 60, floored_minutes % 60), tzinfo=TAIPEI
    )
    # Only fire when the bar period that ENDED at this boundary overlapped an
    # open session (otherwise there is no new data to act on).
    probe = boundary - dt.timedelta(minutes=min(minutes, 1))
    if not market_calendar.session_open_now(deployment.market, probe, include_night=include_night):
        return None
    return boundary


class DeployScheduler:
    """Owns the loop, the sessions, and tick fan-out."""

    def __init__(self, publish: Callable[[dict[str, Any]], None] | None = None):
        self._publish = publish or (lambda event: None)
        self.sessions = SessionManager(fill_sink=self._on_fill)
        self._task: asyncio.Task | None = None
        self._watermarks: dict[str, dt.datetime] = {}
        self._lag_retry: dict[str, dt.datetime] = {}
        self._dep_busy: set[str] = set()
        self._lock_handle = None

    # -- fills -> journal ----------------------------------------------------

    def _on_fill(self, environment: str, stat: Any, msg: Any) -> None:
        """order_deal_event sink: route deal events to the owning deployment."""
        try:
            payload = dict(msg) if isinstance(msg, dict) else {"raw": str(msg)}
            code = str(payload.get("code") or "")
            if not code:
                return
            from src.deploy import accounting, contracts

            for dep in store.list_deployments():
                if dep.environment != environment:
                    continue
                if dep.market == TW_FUTURES:
                    owns = contracts.product_of(code) == contracts.product_of(dep.symbol)
                else:
                    owns = code.split(".")[0] == dep.symbol.split(".")[0]
                if owns:
                    run_dir = safe_run_id(dep.run_id)
                    payload["state"] = str(stat)
                    if dep.market != TW_FUTURES and payload.get("quantity") is not None:
                        # Common-lot stock deal events report quantity in
                        # LOTS (張); deploy-internal equity units are shares.
                        try:
                            payload["quantity_lots"] = float(payload["quantity"])
                            payload["quantity"] = float(payload["quantity"]) * 1000
                        except (TypeError, ValueError):
                            pass
                    if accounting.append_fill(run_dir, payload):
                        self._publish({"type": "fill", "deployment_id": dep.id, "fill": payload})
                    break
        except Exception:  # noqa: BLE001 - SDK callback thread must never die
            logger.exception("fill sink failed")

    # -- loop -----------------------------------------------------------------

    def _acquire_singleton_lock(self) -> bool:
        lock_path = store.store_path().parent / "deploy_scheduler.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        handle = open(lock_path, "w")  # noqa: SIM115 - held for process lifetime
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            handle.close()
            logger.error("deploy scheduler lock held elsewhere -- NOT starting (double-fire guard)")
            return False
        self._lock_handle = handle
        return True

    def start(self) -> None:
        if self._task is not None:
            return
        if not self._acquire_singleton_lock():
            return
        self._task = asyncio.get_event_loop().create_task(self._run())
        logger.info("deploy scheduler started")

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        await asyncio.to_thread(self.sessions.shutdown)
        if self._lock_handle is not None:
            self._lock_handle.close()
            self._lock_handle = None

    async def _run(self) -> None:
        while True:
            try:
                await self._sweep()
            except Exception:  # noqa: BLE001 - the loop must survive anything
                logger.exception("deploy scheduler sweep failed")
            await asyncio.sleep(_WAKE_SECONDS)

    async def _sweep(self) -> None:
        if store.kill_switch_engaged():
            return
        now = market_calendar.now_taipei()
        for dep in store.list_deployments():
            if not dep.enabled or dep.id in self._dep_busy:
                continue
            retry_at = self._lag_retry.get(dep.id)
            boundary = due_bar_boundary(dep, now)
            if boundary is None:
                continue
            if self._watermarks.get(dep.id) == boundary and (retry_at is None or now < retry_at):
                continue
            is_retry = self._watermarks.get(dep.id) == boundary
            self._dep_busy.add(dep.id)
            asyncio.get_event_loop().create_task(
                self._tick(dep, boundary, is_retry=is_retry)
            )

    async def _tick(self, dep: store.Deployment, boundary: dt.datetime, *, is_retry: bool) -> None:
        try:
            outcome = await asyncio.to_thread(self._tick_sync, dep)
            self._watermarks[dep.id] = boundary
            lagging = (
                outcome.status == "skipped"
                and outcome.bar_ts is not None
                and not is_retry
            )
            if lagging:
                self._lag_retry[dep.id] = market_calendar.now_taipei() + dt.timedelta(
                    seconds=_DATA_LAG_RETRY_SECONDS
                )
            else:
                self._lag_retry.pop(dep.id, None)
            status = outcome.status if not lagging else "data_lag_retry_scheduled"
            store.update_deployment(
                dep.id,
                last_tick_at=market_calendar.now_taipei().isoformat(),
                last_tick_status=status,
                last_error=None if outcome.status in ("ok", "skipped", "dry_run") else outcome.reason,
            )
            self._publish({
                "type": "tick",
                "deployment_id": dep.id,
                "status": outcome.status,
                "reason": outcome.reason,
                "bar_ts": outcome.bar_ts,
                "elapsed_seconds": outcome.elapsed_seconds,
            })
        except SessionUnavailableError as exc:
            store.update_deployment(
                dep.id, last_tick_status="failed", last_error=str(exc),
                last_tick_at=market_calendar.now_taipei().isoformat(),
            )
            self._publish({"type": "tick", "deployment_id": dep.id, "status": "failed", "reason": str(exc)})
        except Exception as exc:  # noqa: BLE001 - one deployment's bug must not stop others
            logger.exception("tick crashed for %s", dep.id)
            try:
                store.update_deployment(
                    dep.id, last_tick_status="failed", last_error=str(exc),
                    last_tick_at=market_calendar.now_taipei().isoformat(),
                )
            except Exception:  # noqa: BLE001
                pass
            self._publish({"type": "tick", "deployment_id": dep.id, "status": "failed", "reason": str(exc)})
        finally:
            self._dep_busy.discard(dep.id)

    def _tick_sync(self, dep: store.Deployment) -> executor.TickOutcome:
        run_dir = safe_run_id(dep.run_id)
        with self.sessions.use(dep.environment) as api:
            data_api = self._data_session(api, dep)
            return executor.run_tick(
                dep, run_dir, session_api=api, data_api=data_api,
            )

    def _data_session(self, trading_api: Any, dep: store.Deployment) -> Any:
        """Market data rides the paper session (same feed both environments)."""
        if dep.environment == "paper":
            return trading_api
        try:
            with self.sessions.use("paper") as paper_api:
                return paper_api
        except SessionUnavailableError:
            return trading_api  # live session also carries market data

    # -- manual operations (API layer calls these) ----------------------------

    def run_once(self, deployment_id: str, *, dry_run: bool) -> executor.TickOutcome:
        dep = store.get_deployment(deployment_id)
        if dep is None:
            raise store.DeploymentError(f"no deployment {deployment_id}")
        run_dir = safe_run_id(dep.run_id)
        with self.sessions.use(dep.environment) as api:
            return executor.run_tick(
                dep, run_dir, session_api=api, dry_run=dry_run,
            )

    def flatten(self, deployment_id: str) -> list[dict[str, Any]]:
        dep = store.get_deployment(deployment_id)
        if dep is None:
            raise store.DeploymentError(f"no deployment {deployment_id}")
        run_dir = safe_run_id(dep.run_id)
        with self.sessions.use(dep.environment) as api:
            results = executor.flatten(dep, run_dir, session_api=api)
        store.update_deployment(dep.id, enabled=False, last_tick_status="flattened")
        self._release_unused_sessions()
        self._publish({"type": "flatten", "deployment_id": dep.id, "orders": results})
        return results

    def _release_unused_sessions(self) -> None:
        active = {d.environment for d in store.list_deployments() if d.enabled}
        for env in ("paper", "live"):
            self.sessions.release_if_unused(env, still_needed=env in active)

    def on_deployment_toggled(self) -> None:
        self._release_unused_sessions()
