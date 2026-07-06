"""Per-deployment records: tick journal, fills journal, equity accounting.

File layout (all under ``agent/runs/<run_id>/live/`` on the shared volume the
web UI reads):

* ``ticks/<bar-ts>.json`` -- two-phase tick record (``intent`` written before
  any order, updated to ``final`` after). The file's existence is the bar's
  idempotency key; an ``intent``-only file after a crash routes the next
  attempt through reconcile-repair instead of blind re-execution.
* ``fills.jsonl`` -- append-only order/deal events from ``order_deal_event``,
  deduped by exchange sequence/order id (reconnects can re-deliver).
* ``daily/<date>.json`` -- end-of-day summary feeding the daily-PnL chart.

Equity attribution: a deployment's synthetic equity = allocated_capital
+ realized PnL from its own fills (FIFO-paired, fees included via the SAME
``calc_commission`` the backtest engines use) + unrealized PnL on its current
position. Whole-account equity is NOT used -- multiple deployments share one
Shioaji account and must not bleed into each other's curves.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from backtest.engines.tw_equity import TWEquityEngine
from backtest.engines.tw_futures import TWFuturesEngine

from src.deploy.market_calendar import TW_EQUITY

_LOCK = threading.Lock()


def live_dir(run_dir: Path) -> Path:
    return run_dir / "live"


def ticks_dir(run_dir: Path) -> Path:
    return live_dir(run_dir) / "ticks"


def fills_path(run_dir: Path) -> Path:
    return live_dir(run_dir) / "fills.jsonl"


def daily_dir(run_dir: Path) -> Path:
    return live_dir(run_dir) / "daily"


def tick_record_path(run_dir: Path, bar_ts: str) -> Path:
    safe = bar_ts.replace(":", "").replace(" ", "T")
    return ticks_dir(run_dir) / f"{safe}.json"


def write_tick_record(run_dir: Path, bar_ts: str, payload: dict[str, Any]) -> Path:
    path = tick_record_path(run_dir, bar_ts)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    return path


def read_tick_record(run_dir: Path, bar_ts: str) -> dict[str, Any] | None:
    path = tick_record_path(run_dir, bar_ts)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def list_tick_records(run_dir: Path, limit: int = 200) -> list[dict[str, Any]]:
    folder = ticks_dir(run_dir)
    if not folder.exists():
        return []
    records = []
    for path in sorted(folder.glob("*.json"))[-limit:]:
        try:
            records.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            continue
    return records


# --- fills journal ----------------------------------------------------------


def _fill_key(fill: dict[str, Any]) -> str:
    return str(
        fill.get("exchange_seq")
        or fill.get("seqno")
        or f"{fill.get('ordno')}|{fill.get('ts')}"
    )


def load_fill_keys(run_dir: Path) -> set[str]:
    path = fills_path(run_dir)
    keys: set[str] = set()
    if not path.exists():
        return keys
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                keys.add(_fill_key(json.loads(line)))
            except json.JSONDecodeError:
                continue
    return keys


def append_fill(run_dir: Path, fill: dict[str, Any], seen_keys: set[str] | None = None) -> bool:
    """Append one fill, deduped. Returns True if written (i.e. new)."""
    with _LOCK:
        keys = seen_keys if seen_keys is not None else load_fill_keys(run_dir)
        key = _fill_key(fill)
        if key in keys:
            return False
        path = fills_path(run_dir)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(fill, ensure_ascii=False, default=str) + "\n")
        keys.add(key)
        return True


def load_fills(run_dir: Path) -> list[dict[str, Any]]:
    path = fills_path(run_dir)
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


# --- equity accounting -------------------------------------------------------


@dataclass(frozen=True)
class EquitySnapshot:
    realized_pnl: float
    unrealized_pnl: float
    fees: float
    equity: float  # allocated + realized - fees + unrealized
    position_qty: int  # signed
    avg_entry: float | None


def _fee_for(market: str, symbol: str, qty: float, price: float, is_open: bool) -> float:
    """Fees via the backtest engines' own calc_commission (same rates/rules)."""
    if market == TW_EQUITY:
        engine = TWEquityEngine({"initial_cash": 1.0})
        engine._active_symbol = symbol
        return float(engine.calc_commission(qty, price, 1, is_open))
    engine = TWFuturesEngine({"initial_cash": 1.0})
    engine._active_symbol = symbol
    return float(engine.calc_commission(qty, price, 1, is_open))


def compute_equity(
    *,
    market: str,
    symbol: str,
    allocated_capital: float,
    fills: list[dict[str, Any]],
    mark_price: float | None,
) -> EquitySnapshot:
    """FIFO-pair the deployment's own fills into realized/unrealized PnL.

    Fill rows need: ``side`` (buy/sell), ``quantity`` (shares or contracts,
    positive), ``price``. Futures PnL scales by the contract multiplier via
    the engine (single-sourced); equity PnL is share-based. ``rollover``
    fills participate normally (they carry real PnL; only trade STATISTICS
    exclude them, handled at the stats layer).
    """
    multiplier = 1.0
    if market != TW_EQUITY:
        engine = TWFuturesEngine({"initial_cash": 1.0})
        multiplier = float(engine.get_contract_multiplier(symbol))

    lots: list[list[float]] = []  # [signed_qty_remaining, entry_price]
    realized = 0.0
    fees = 0.0
    for fill in fills:
        qty = float(fill.get("quantity") or 0)
        price = float(fill.get("price") or 0)
        if qty <= 0 or price <= 0:
            continue
        direction = 1 if str(fill.get("side", "")).lower() in ("buy", "long") else -1
        # Equity fee rule cares about open vs close (transaction tax is
        # sell-only); with no short selling, buy==open and sell==close.
        # Futures fees are side-symmetric in the engine, so this is exact
        # there too regardless of short opens.
        fees += _fee_for(market, symbol, qty, price, is_open=direction > 0)
        remaining = qty
        while remaining > 1e-9 and lots and (lots[0][0] > 0) != (direction > 0):
            lot = lots[0]
            matched = min(remaining, abs(lot[0]))
            lot_dir = 1 if lot[0] > 0 else -1
            realized += lot_dir * matched * (price - lot[1]) * multiplier
            lot[0] -= lot_dir * matched
            remaining -= matched
            if abs(lot[0]) < 1e-9:
                lots.pop(0)
        if remaining > 1e-9:
            lots.append([direction * remaining, price])

    position_qty = int(round(sum(lot[0] for lot in lots)))
    avg_entry = None
    unrealized = 0.0
    if lots and mark_price:
        total = sum(abs(lot[0]) for lot in lots)
        avg_entry = sum(abs(lot[0]) * lot[1] for lot in lots) / total
        direction = 1 if position_qty > 0 else -1
        unrealized = direction * total * (mark_price - avg_entry) * multiplier

    return EquitySnapshot(
        realized_pnl=realized,
        unrealized_pnl=unrealized,
        fees=fees,
        equity=allocated_capital + realized - fees + unrealized,
        position_qty=position_qty,
        avg_entry=avg_entry,
    )


def write_daily_summary(run_dir: Path, day: dt.date, payload: dict[str, Any]) -> Path:
    path = daily_dir(run_dir) / f"{day.isoformat()}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    return path


def equity_series(run_dir: Path, limit: int = 500) -> list[dict[str, Any]]:
    """Per-tick equity snapshots for the chart, oldest first."""
    points = []
    for record in list_tick_records(run_dir, limit=limit):
        snap = record.get("equity_snapshot") or {}
        if "equity" in snap:
            points.append(
                {
                    "ts": record.get("bar_ts"),
                    "equity": snap.get("equity"),
                    "realized_pnl": snap.get("realized_pnl"),
                    "unrealized_pnl": snap.get("unrealized_pnl"),
                }
            )
    return points
