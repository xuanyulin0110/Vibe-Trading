"""Deployment configuration store (``~/.vibe-trading/deployments.json``).

A deployment binds one backtested run to one tradable symbol on one
environment. The file lives on the ``vibe-home`` volume so it survives
container rebuilds. Writes are atomic (temp file + rename) and serialized by
a process-wide lock -- the API server is single-process (see the scheduler's
singleton guard), so no cross-process locking is needed here.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import threading
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from src.config.paths import get_runtime_root

from src.deploy.market_calendar import TW_EQUITY, TW_FUTURES

STORE_FILENAME = "deployments.json"
KILL_SWITCH_FILENAME = "deploy_kill_switch"

_LOCK = threading.Lock()

VALID_ENVIRONMENTS = ("paper", "live")
VALID_SESSIONS = ("day", "day_night")


class DeploymentError(ValueError):
    """Raised on invalid deployment definitions or store operations."""


@dataclass
class Deployment:
    """One strategy deployment.

    ``environment`` is immutable after creation (paper positions cannot carry
    over to live; going live means creating a new deployment through the
    UI's live-friction flow). ``interval`` is copied from the run's
    config.json at creation and never user-editable -- the strategy's bar
    interval IS the execution cadence.
    """

    id: str
    run_id: str
    symbol: str
    market: str  # tw_equity | tw_futures
    environment: str  # paper | live
    interval: str  # inherited from run config, e.g. "1D", "5m"
    sessions: str = "day"  # futures only: day | day_night
    allocated_capital: float = 0.0
    # Deterministic safety caps -- all mandatory at creation, no defaults in
    # the UI. Exceeding any one rejects the whole order (never resizes).
    max_order_qty: int = 0  # contracts (futures) / shares (equity)
    max_daily_orders: int = 0
    max_order_notional: float = 0.0  # TWD
    enabled: bool = False
    created_at: str = ""
    # Runtime summary (updated by the executor; not user-editable).
    last_tick_at: str | None = None
    last_tick_status: str | None = None
    last_error: str | None = None
    paused_reason: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.run_id.strip():
            raise DeploymentError("run_id is required")
        symbol = self.symbol.strip().upper()
        if self.market == TW_FUTURES and not symbol.endswith(".TWF"):
            raise DeploymentError("tw_futures deployments need a .TWF symbol")
        if self.market == TW_EQUITY and not (symbol.endswith(".TW") and not symbol.endswith(".TWF")):
            raise DeploymentError("tw_equity deployments need a .TW symbol")
        if self.market not in (TW_EQUITY, TW_FUTURES):
            raise DeploymentError(f"unknown market {self.market!r}")
        if self.environment not in VALID_ENVIRONMENTS:
            raise DeploymentError("environment must be 'paper' or 'live'")
        if self.sessions not in VALID_SESSIONS:
            raise DeploymentError("sessions must be 'day' or 'day_night'")
        if self.market == TW_EQUITY and self.sessions != "day":
            raise DeploymentError("equities have no night session")
        if self.allocated_capital <= 0:
            raise DeploymentError("allocated_capital must be positive")
        if self.max_order_qty <= 0 or self.max_daily_orders <= 0 or self.max_order_notional <= 0:
            raise DeploymentError(
                "safety caps (max_order_qty, max_daily_orders, max_order_notional) are all required and positive"
            )


def store_path() -> Path:
    return get_runtime_root() / STORE_FILENAME


def kill_switch_path() -> Path:
    return get_runtime_root() / KILL_SWITCH_FILENAME


def kill_switch_engaged() -> bool:
    """Global stop flag -- a plain file so it survives restarts."""
    return kill_switch_path().exists()


def set_kill_switch(engaged: bool) -> None:
    path = kill_switch_path()
    if engaged:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(dt.datetime.now(dt.timezone.utc).isoformat(), encoding="utf-8")
    else:
        try:
            path.unlink()
        except FileNotFoundError:
            pass


def _read_all() -> list[dict[str, Any]]:
    path = store_path()
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DeploymentError(f"corrupt deployments store at {path}: {exc}") from exc
    return list(data.get("deployments", []))


def _write_all(rows: list[dict[str, Any]]) -> None:
    path = store_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(
        json.dumps({"deployments": rows}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, path)


def _from_row(row: dict[str, Any]) -> Deployment:
    known = {f for f in Deployment.__dataclass_fields__}
    return Deployment(**{k: v for k, v in row.items() if k in known})


def list_deployments() -> list[Deployment]:
    with _LOCK:
        return [_from_row(r) for r in _read_all()]


def get_deployment(deployment_id: str) -> Deployment | None:
    for d in list_deployments():
        if d.id == deployment_id:
            return d
    return None


def create_deployment(
    *,
    run_id: str,
    symbol: str,
    market: str,
    environment: str,
    interval: str,
    sessions: str = "day",
    allocated_capital: float = 0.0,
    max_order_qty: int = 0,
    max_daily_orders: int = 0,
    max_order_notional: float = 0.0,
    meta: dict[str, Any] | None = None,
) -> Deployment:
    """Create and persist a deployment; rejects a second deployment on the
    same symbol (clean position attribution requires 1:1 symbol ownership)."""
    dep = Deployment(
        id=uuid.uuid4().hex[:12],
        run_id=run_id.strip(),
        symbol=symbol.strip().upper(),
        market=market,
        environment=environment,
        interval=str(interval),
        sessions=sessions,
        allocated_capital=float(allocated_capital),
        max_order_qty=int(max_order_qty),
        max_daily_orders=int(max_daily_orders),
        max_order_notional=float(max_order_notional),
        enabled=False,
        created_at=dt.datetime.now(dt.timezone.utc).isoformat(),
        meta=dict(meta or {}),
    )
    dep.validate()
    with _LOCK:
        rows = _read_all()
        if any(r.get("symbol", "").upper() == dep.symbol for r in rows):
            raise DeploymentError(
                f"a deployment for {dep.symbol} already exists -- one deployment per symbol"
            )
        rows.append(asdict(dep))
        _write_all(rows)
    return dep


def update_deployment(deployment_id: str, **changes: Any) -> Deployment:
    """Update mutable fields. ``environment``/``run_id``/``symbol``/``market``/
    ``interval`` are immutable by design."""
    immutable = {"id", "environment", "run_id", "symbol", "market", "interval", "created_at"}
    bad = immutable.intersection(changes)
    if bad:
        raise DeploymentError(f"immutable fields: {sorted(bad)}")
    with _LOCK:
        rows = _read_all()
        for row in rows:
            if row.get("id") == deployment_id:
                row.update(changes)
                dep = _from_row(row)
                dep.validate()
                _write_all(rows)
                return dep
    raise DeploymentError(f"no deployment {deployment_id}")


def delete_deployment(deployment_id: str) -> bool:
    with _LOCK:
        rows = _read_all()
        kept = [r for r in rows if r.get("id") != deployment_id]
        if len(kept) == len(rows):
            return False
        _write_all(kept)
        return True
