"""Persistent Shioaji session management for the deploy runtime.

Follows the Shioaji skill's CONCEPTS.md model: one login == one connection,
long-running trading needs a long-lived process, and simulation/production
are fixed at ``Shioaji()`` construction -- so this manager keeps AT MOST one
session per environment (``paper``, ``live``), created lazily, reused by
every deployment and every data fetch, and logged out only on shutdown or
when the environment's last deployment is disabled. There is no per-tick
login/logout: at intraday cadence that would both burn the daily login quota
and add seconds of connect latency to every bar.

Connection budget: paper (1) + live (1) persistent, and the paper session is
also lent to the data loaders (market data is the same real feed in both
Shioaji environments), so the whole runtime stays well under the 5-per-person
cap without login-count growth.

Order/deal events (``order_deal_event``) are received via
``set_order_callback`` per the skill's ORDERS.md best practice (prefer push
reports over ``update_status`` polling) and handed to a sink callable the
runtime wires to the fills journal.
"""

from __future__ import annotations

import logging
import threading
import time
from contextlib import contextmanager
from typing import Any, Callable

from src.trading.connectors.shioaji import sdk

logger = logging.getLogger(__name__)

_MAX_RECONNECT_ATTEMPTS = 5
_RECONNECT_BASE_DELAY = 2.0  # seconds; doubles per attempt


class SessionUnavailableError(RuntimeError):
    """Raised when a usable session for the environment cannot be provided."""


class SessionManager:
    """Lazy, per-environment, lock-serialized Shioaji sessions."""

    def __init__(self, fill_sink: Callable[[str, Any, Any], None] | None = None):
        self._sessions: dict[str, Any] = {}
        self._locks = {"paper": threading.RLock(), "live": threading.RLock()}
        self._failed: dict[str, str] = {}
        self._reconnects: dict[str, int] = {"paper": 0, "live": 0}
        self._fill_sink = fill_sink

    # -- config ------------------------------------------------------------

    @staticmethod
    def _config_for(environment: str) -> sdk.ShioajiConfig:
        """Saved shioaji.json for this environment's profile.

        ``sdk.load_config()`` already falls back to SJ_API_KEY/SJ_SEC_KEY/
        SJ_CA_PATH/SJ_CA_PASSWD when the file leaves a field empty, so a
        deployment works with credentials living only in agent/.env, same as
        the backtest loaders and every other Shioaji entry point.
        """
        cfg = sdk.load_config()
        profile = "paper" if environment == "paper" else "live"
        return cfg.with_overrides(profile=profile)

    # -- lifecycle ----------------------------------------------------------

    def _build(self, environment: str) -> Any:
        cfg = self._config_for(environment)
        api = sdk._login(cfg)  # raises ShioajiConfigError on CA failure (live)
        try:
            api.set_order_callback(self._make_order_callback(environment))
        except Exception:  # noqa: BLE001 - callback wiring is best-effort on fakes
            logger.warning("could not register order callback for %s session", environment)
        return api

    def _make_order_callback(self, environment: str):
        def _on_order_event(stat: Any, msg: Any) -> None:
            sink = self._fill_sink
            if sink is not None:
                try:
                    sink(environment, stat, msg)
                except Exception:  # noqa: BLE001 - a sink bug must not kill the SDK thread
                    logger.exception("fill sink failed for %s event", environment)

        return _on_order_event

    @contextmanager
    def use(self, environment: str):
        """Yield the environment's session under its lock (serialized use).

        Creates the session lazily; on any prior hard failure the environment
        stays failed (red light) until ``reset()`` -- no silent retry storms.
        """
        if environment not in self._locks:
            raise SessionUnavailableError(f"unknown environment {environment!r}")
        with self._locks[environment]:
            if environment in self._failed:
                raise SessionUnavailableError(
                    f"{environment} session marked failed: {self._failed[environment]}"
                )
            api = self._sessions.get(environment)
            if api is None:
                api = self._connect_with_backoff(environment)
            yield api

    def _connect_with_backoff(self, environment: str) -> Any:
        attempts = 0
        while True:
            try:
                api = self._build(environment)
                self._sessions[environment] = api
                self._reconnects[environment] = 0
                logger.info("shioaji %s session established", environment)
                return api
            except sdk.ShioajiConfigError as exc:
                # Config/CA problems don't heal by retrying -- fail the env.
                self._failed[environment] = str(exc)
                raise SessionUnavailableError(str(exc)) from exc
            except Exception as exc:  # noqa: BLE001 - network-ish: bounded backoff
                attempts += 1
                if attempts >= _MAX_RECONNECT_ATTEMPTS:
                    self._failed[environment] = f"login failed after {attempts} attempts: {exc}"
                    raise SessionUnavailableError(self._failed[environment]) from exc
                delay = _RECONNECT_BASE_DELAY * (2 ** (attempts - 1))
                logger.warning(
                    "shioaji %s login attempt %d failed (%s); retrying in %.0fs",
                    environment, attempts, exc, delay,
                )
                time.sleep(delay)

    def invalidate(self, environment: str, reason: str = "") -> None:
        """Drop a (presumably broken) session so the next use reconnects."""
        with self._locks[environment]:
            api = self._sessions.pop(environment, None)
            if api is not None:
                sdk._logout_best_effort(api)
            logger.warning("shioaji %s session invalidated%s", environment, f": {reason}" if reason else "")

    def reset(self, environment: str) -> None:
        """Clear a failed flag (operator-initiated, e.g. after fixing CA)."""
        with self._locks[environment]:
            self._failed.pop(environment, None)

    def release_if_unused(self, environment: str, still_needed: bool) -> None:
        """Log out when the environment's last deployment was disabled."""
        if still_needed:
            return
        with self._locks[environment]:
            api = self._sessions.pop(environment, None)
            if api is not None:
                sdk._logout_best_effort(api)
                logger.info("shioaji %s session released (no enabled deployments)", environment)

    def shutdown(self) -> None:
        for environment in list(self._sessions):
            self.invalidate(environment, "shutdown")

    # -- introspection -------------------------------------------------------

    def status(self) -> dict[str, Any]:
        return {
            env: {
                "connected": env in self._sessions,
                "failed": self._failed.get(env),
            }
            for env in self._locks
        }
