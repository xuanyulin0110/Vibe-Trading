"""Deterministic deployment runtime wiring (fork feature, ``src/deploy``).

Mounted by ``agent/api_server.py``: the scheduler lifecycle hooks run from the
app's startup/shutdown events, and ``register_deploy_routes`` mounts the
control + SSE routers. Isolated from the LLM live framework: this schedules
signal_engine-driven ticks only.
"""

from __future__ import annotations

import logging
from typing import Any, Awaitable, Callable

from fastapi import Depends, FastAPI

logger = logging.getLogger(__name__)

_deploy_scheduler = None


def start_deploy_scheduler() -> None:
    """Boot the deterministic deployment scheduler (src/deploy).

    Failures here must never block API startup.
    """
    global _deploy_scheduler
    try:
        from src.deploy import notifier as deploy_notifier
        from src.deploy.api import event_bus, set_scheduler
        from src.deploy.scheduler import DeployScheduler

        event_bus.bind_loop()
        deploy_notifier.bind_loop()

        def _fanout(event: dict) -> None:
            event_bus.publish(event)  # SSE for the web UI
            deploy_notifier.publish(event)  # IM push (Telegram etc.)

        _deploy_scheduler = DeployScheduler(publish=_fanout)
        _deploy_scheduler.start()
        set_scheduler(_deploy_scheduler)
    except Exception:  # noqa: BLE001 - deploy runtime is optional at boot
        logger.exception("deploy scheduler failed to start")


async def stop_deploy_scheduler() -> None:
    global _deploy_scheduler
    if _deploy_scheduler is not None:
        try:
            await _deploy_scheduler.stop()
        except Exception:  # noqa: BLE001
            logger.exception("deploy scheduler shutdown failed")
        _deploy_scheduler = None


AuthDep = Callable[..., Awaitable[Any] | Any]


def register_deploy_routes(
    app: FastAPI,
    require_auth: AuthDep,
    require_event_stream_auth: AuthDep,
) -> None:
    """Mount the deployment control + SSE routers onto ``app``.

    JSON control routes take normal Bearer auth; the SSE stream takes the
    EventSource-compatible query auth, same split as /sessions/{id}/events.
    """
    from src.deploy.api import events_router, router

    # events_router FIRST: /deployments/events must win over the main router's
    # /deployments/{deployment_id} catch-all (routes match in registration order).
    app.include_router(events_router, dependencies=[Depends(require_event_stream_auth)])
    app.include_router(router, dependencies=[Depends(require_auth)])
