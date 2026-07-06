"""One deterministic tick: signal -> target -> diff -> orders -> records.

No LLM anywhere in this path. Every decision is the run's own
``signal_engine.py`` output pushed through the backtest engines' sizing, a
position diff against the broker's own truth, and hard safety caps. The tick
writes a two-phase record (``intent`` before any order, ``final`` after) so a
crash mid-tick is repairable by reconciliation instead of blind re-execution.
"""

from __future__ import annotations

import datetime as dt
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

from src.deploy import accounting, contracts, market_calendar, signal_runner, sizing
from src.deploy.market_calendar import TW_EQUITY, TW_FUTURES
from src.deploy.store import Deployment, kill_switch_engaged
from src.trading.connectors.shioaji import sdk

#: Bounded IOC-remainder re-places within one tick (futures partial fills).
_MAX_REFILLS = 2
#: Price-limit guard mirrors the backtest engines' ±10% can_execute rule.
_PRICE_LIMIT = 0.10
#: ROD fill patience: poll until filled (or this budget), then cancel the rest.
_ROD_FILL_WAIT_SECONDS = 30
_ROD_POLL_SECONDS = 3


@dataclass
class TickOutcome:
    status: str  # ok | dry_run | skipped | blocked | failed
    reason: str
    bar_ts: str | None = None
    weight: float | None = None
    target_qty: int | None = None
    current_qty: int | None = None
    orders: list[dict[str, Any]] | None = None
    elapsed_seconds: float | None = None


def current_position_qty(positions: list[dict[str, Any]], deployment: Deployment, api: Any) -> int:
    """Signed broker position for this deployment's symbol.

    Equities match by bare stock id. Futures match by PRODUCT across dated
    codes (the deployment symbol is the continuous alias; broker positions
    come back as dated contracts).
    """
    total = 0
    if deployment.market == TW_EQUITY:
        stock_id = deployment.symbol.split(".")[0]
        for pos in positions:
            code = str(pos.get("symbol") or pos.get("code") or "")
            if code.split(".")[0] == stock_id:
                # Shioaji common-lot equity positions are denominated in
                # LOTS (張) -- confirmed live 2026-07-06: a 2,000-share fill
                # reports quantity=2. Deploy-internal equity quantities are
                # SHARES throughout, so convert at this boundary.
                total += _signed_qty(pos) * 1000
        return total
    product = contracts.product_of(deployment.symbol)
    for pos in positions:
        code = str(pos.get("symbol") or pos.get("code") or "")
        if contracts.product_of(code) == product:
            total += _signed_qty(pos)
    return total


def _signed_qty(pos: dict[str, Any]) -> int:
    qty = int(abs(float(pos.get("quantity") or 0)))
    direction = pos.get("direction")
    if direction in (1, -1):
        return direction * qty
    side = str(pos.get("side", pos.get("action", ""))).lower()
    return -qty if side in ("sell", "short") else qty


def check_safety_caps(
    deployment: Deployment, order_qty: int, price: float, orders_today: int, market: str,
) -> str | None:
    """Return a rejection reason, or None when the order passes every cap."""
    if order_qty > deployment.max_order_qty:
        return f"order qty {order_qty} exceeds max_order_qty {deployment.max_order_qty}"
    if orders_today + 1 > deployment.max_daily_orders:
        return f"daily order cap {deployment.max_daily_orders} reached"
    if market == TW_FUTURES:
        from backtest.engines.tw_futures import TWFuturesEngine

        cm = TWFuturesEngine({"initial_cash": 1.0}).get_contract_multiplier(deployment.symbol)
        notional = order_qty * price * cm
    else:
        notional = order_qty * price
    if notional > deployment.max_order_notional:
        return f"order notional {notional:,.0f} exceeds max_order_notional {deployment.max_order_notional:,.0f}"
    return None


def limit_guard_blocks(side: str, quote: dict[str, Any]) -> bool:
    """Mirror the backtest can_execute ±10% rule with a live quote.

    Blocks buys at limit-up and sells at limit-down; missing reference data
    means no block (same as the backtest engines, which skip the check when
    the bar carries no prior-price reference). Accepts the Shioaji snapshot
    shape (``change_rate`` in percent, or ``last``+``change_price``).
    """
    pct: float | None = None
    if quote.get("change_rate") is not None:
        try:
            pct = float(quote["change_rate"]) / 100.0
        except (TypeError, ValueError):
            pct = None
    if pct is None:
        last = quote.get("last") or quote.get("close")
        change = quote.get("change_price")
        try:
            last_f, change_f = float(last), float(change)
            ref = last_f - change_f
            pct = (last_f - ref) / ref if ref > 0 else None
        except (TypeError, ValueError):
            pct = None
    if pct is None:
        return False
    if side == "buy" and pct >= _PRICE_LIMIT - 0.001:
        return True
    if side == "sell" and pct <= -_PRICE_LIMIT + 0.001:
        return True
    return False


def orders_placed_today(run_dir: Path, today: dt.date) -> int:
    count = 0
    for record in accounting.list_tick_records(run_dir, limit=500):
        ts = str(record.get("executed_at") or "")
        if ts[:10] == today.isoformat():
            count += len(record.get("orders") or [])
    return count


def run_tick(
    deployment: Deployment,
    run_dir: Path,
    *,
    session_api: Any,
    now: dt.datetime | None = None,
    dry_run: bool = False,
    data_api: Any = None,
    quote_fn: Callable[[Any, str], dict[str, Any]] | None = None,
    signal_result: signal_runner.SignalResult | None = None,
) -> TickOutcome:
    """Execute one tick for ``deployment``.

    Args:
        session_api: The environment's logged-in trading session.
        data_api: Session lent to loaders for data fetch (defaults to
            ``session_api``; tests inject fakes).
        quote_fn: Live-quote getter for the price-limit guard; defaults to
            ``sdk.get_quote``.
        signal_result: Precomputed signal (tests / dry-run previews).
    """
    started = time.monotonic()
    now = now or market_calendar.now_taipei()
    today = now.astimezone(market_calendar.TAIPEI).date()

    if kill_switch_engaged():
        return TickOutcome("blocked", "global kill switch engaged")

    # 1. Signal (fidelity path -- run's own engine, full-history recompute).
    try:
        sig = signal_result or signal_runner.compute_signal(
            run_dir,
            deployment.symbol,
            deployment.market,
            now=now,
            injected_api=data_api if data_api is not None else session_api,
        )
    except signal_runner.SignalComputationError as exc:
        return TickOutcome("failed", f"signal: {exc}")

    bar_key = str(sig.bar_ts)
    existing = accounting.read_tick_record(run_dir, bar_key)
    if existing is not None and existing.get("phase") == "final":
        return TickOutcome("skipped", "bar already executed", bar_ts=bar_key)

    # Crash-recovery: an intent without a final means orders may be in
    # flight. Refuse to re-run blindly unless the broker shows no resting
    # orders for this symbol (diff-based convergence is idempotent AFTER
    # in-flight orders settle).
    if existing is not None and existing.get("phase") == "intent" and not dry_run:
        open_orders = (sdk.get_open_orders(None, api=session_api) or {}).get("open_orders", [])
        if any(_order_matches(deployment, o) for o in open_orders):
            return TickOutcome(
                "blocked", "incomplete prior tick with resting orders -- reconcile first", bar_ts=bar_key,
            )

    # 2. Broker truth: positions + (futures) rollover before the diff.
    positions_payload = sdk.get_positions(None, api=session_api)
    if positions_payload.get("status") == "error":
        return TickOutcome("failed", f"get_positions: {positions_payload.get('error')}", bar_ts=bar_key)
    positions = list(positions_payload.get("positions") or [])

    orders: list[dict[str, Any]] = []
    if deployment.market == TW_FUTURES and not dry_run:
        rollover = contracts.plan_rollover(session_api, deployment.symbol, positions, today)
        if rollover is not None:
            roll_orders = _execute_rollover(deployment, rollover, session_api)
            orders.extend(roll_orders)
            positions_payload = sdk.get_positions(None, api=session_api)
            positions = list(positions_payload.get("positions") or [])

    current = current_position_qty(positions, deployment, session_api)

    # 3. Sizing (same engines as the backtest).
    sized = sizing.target_quantity(
        market=deployment.market,
        symbol=deployment.symbol,
        weight=sig.weight,
        allocated_capital=deployment.allocated_capital,
        price=sig.close,
    )
    delta = sized.target_qty - current

    intent: dict[str, Any] = {
        "phase": "intent",
        "deployment_id": deployment.id,
        "symbol": deployment.symbol,
        "environment": deployment.environment,
        "bar_ts": bar_key,
        "signal_weight": sig.weight,
        "signal_close": sig.close,
        "bars_evaluated": sig.bars_evaluated,
        "target_qty": sized.target_qty,
        "sizing_reason": sized.reason,
        "current_qty": current,
        "delta": delta,
        "dry_run": dry_run,
        "executed_at": now.isoformat(),
    }

    if delta == 0:
        intent.update(phase="final", status="ok", orders=[], note="position already converged")
        intent["equity_snapshot"] = _equity_snapshot(deployment, run_dir, sig.close)
        intent["elapsed_seconds"] = time.monotonic() - started
        if not dry_run:
            accounting.write_tick_record(run_dir, bar_key, intent)
        return TickOutcome(
            "dry_run" if dry_run else "ok", sized.reason or "converged",
            bar_ts=bar_key, weight=sig.weight, target_qty=sized.target_qty,
            current_qty=current, orders=[], elapsed_seconds=intent["elapsed_seconds"],
        )

    side = "buy" if delta > 0 else "sell"
    order_qty = abs(delta)

    # 4. Safety caps + price-limit guard.
    cap_reason = check_safety_caps(
        deployment, order_qty, sig.close, orders_placed_today(run_dir, today), deployment.market,
    )
    if cap_reason:
        intent.update(phase="final", status="blocked", orders=[], note=cap_reason)
        intent["elapsed_seconds"] = time.monotonic() - started
        if not dry_run:
            accounting.write_tick_record(run_dir, bar_key, intent)
        return TickOutcome("blocked", cap_reason, bar_ts=bar_key, weight=sig.weight,
                           target_qty=sized.target_qty, current_qty=current)

    get_quote = quote_fn or (lambda api, symbol: sdk.get_quote(symbol, api=api))
    try:
        quote = get_quote(session_api, deployment.symbol) or {}
    except Exception:  # noqa: BLE001 - quote failure degrades to no-block, like backtest
        quote = {}
    if limit_guard_blocks(side, quote):
        reason = f"blocked_limit: {side} at price limit (matches backtest can_execute)"
        intent.update(phase="final", status="blocked", orders=[], note=reason)
        intent["elapsed_seconds"] = time.monotonic() - started
        if not dry_run:
            accounting.write_tick_record(run_dir, bar_key, intent)
        return TickOutcome("blocked", reason, bar_ts=bar_key, weight=sig.weight,
                           target_qty=sized.target_qty, current_qty=current)

    planned = _plan_orders(deployment, side, order_qty, session_api, today)
    intent["planned_orders"] = planned

    if dry_run:
        intent["elapsed_seconds"] = time.monotonic() - started
        return TickOutcome(
            "dry_run", "orders planned (not sent)", bar_ts=bar_key, weight=sig.weight,
            target_qty=sized.target_qty, current_qty=current, orders=planned,
            elapsed_seconds=intent["elapsed_seconds"],
        )

    accounting.write_tick_record(run_dir, bar_key, intent)  # phase=intent, pre-order

    # 5. Place, with bounded partial-fill refills, never leaving resting orders.
    orders.extend(
        _place_with_refills(deployment, planned, session_api, run_dir)
    )

    intent.update(phase="final", status="ok", orders=orders)
    intent["equity_snapshot"] = _equity_snapshot(deployment, run_dir, sig.close)
    intent["elapsed_seconds"] = time.monotonic() - started
    accounting.write_tick_record(run_dir, bar_key, intent)
    return TickOutcome(
        "ok", "executed", bar_ts=bar_key, weight=sig.weight, target_qty=sized.target_qty,
        current_qty=current, orders=orders, elapsed_seconds=intent["elapsed_seconds"],
    )


def _plan_orders(
    deployment: Deployment, side: str, qty: int, api: Any, today: dt.date,
) -> list[dict[str, Any]]:
    if deployment.market == TW_FUTURES:
        resolved = contracts.resolve_order_contract(api, deployment.symbol, today)
        return [{
            "symbol": f"{resolved.code}.TWF",
            "side": side,
            "quantity": qty,
            "order_type": "market",
            "time_in_force": "ioc",  # TAIFEX rejects market+rod
            "delivery_month": resolved.delivery_month,
            "kind": "signal",
        }]
    return [{
        "symbol": deployment.symbol,
        "side": side,
        "quantity": qty,
        "order_type": "market",
        "time_in_force": "rod",  # legal on TWSE; remainder cancelled at tick end
        "kind": "signal",
    }]


def _place_with_refills(
    deployment: Deployment, planned: list[dict[str, Any]], api: Any, run_dir: Path,
) -> list[dict[str, Any]]:
    allow_live = deployment.environment == "live"
    results: list[dict[str, Any]] = []
    for order in planned:
        remaining = int(order["quantity"])
        attempts = 0
        while remaining > 0 and attempts <= _MAX_REFILLS:
            attempts += 1
            response = sdk.place_order(
                None,
                symbol=order["symbol"],
                side=order["side"],
                quantity=remaining,
                order_type=order["order_type"],
                time_in_force=order["time_in_force"],
                allow_live=allow_live,
                api=api,
            )
            entry = {**order, "attempt": attempts, "requested": remaining, "response": response}
            results.append(entry)
            if response.get("status") != "ok":
                break
            filled = int(response.get("filled_qty") or 0)
            if order["time_in_force"] == "ioc":
                # IOC: the unfilled remainder is exchange-cancelled; re-place
                # it (bounded). A zero fill means no liquidity at market right
                # now -- stop immediately, the next bar's diff converges.
                remaining -= filled
                if filled == 0:
                    break
            else:
                # ROD: wait for the fill with bounded patience, THEN cancel
                # any remainder -- no resting orders may survive a tick, but
                # a market order needs real seconds to ack+fill (confirmed
                # live 2026-07-06 on the simulation feed: a 2s wait cancelled
                # a perfectly good PendingSubmit order before it could fill).
                deadline = time.monotonic() + _ROD_FILL_WAIT_SECONDS
                resting: list[dict[str, Any]] = []
                while time.monotonic() < deadline:
                    time.sleep(_ROD_POLL_SECONDS)
                    open_orders = (sdk.get_open_orders(None, api=api) or {}).get("open_orders", [])
                    resting = [o for o in open_orders if _order_matches(deployment, o)]
                    if not resting:
                        break
                for o in resting:
                    cancel = sdk.cancel_order(
                        None, str(o.get("order_id") or ""), allow_live=allow_live, api=api,
                    )
                    results.append({"kind": "cleanup_cancel", "order_id": o.get("order_id"), "response": cancel})
                remaining = 0
        if remaining > 0:
            results.append({"kind": "partial_gaveup", "symbol": order["symbol"], "unfilled": remaining})
    return results


def _execute_rollover(
    deployment: Deployment, plan: contracts.RolloverPlan, api: Any,
) -> list[dict[str, Any]]:
    """Close expiring month, reopen next month; recorded as ``rollover`` kind."""
    allow_live = deployment.environment == "live"
    close_side = "sell" if plan.direction > 0 else "buy"
    open_side = "buy" if plan.direction > 0 else "sell"
    results = []
    close_resp = sdk.place_order(
        None, symbol=f"{plan.expiring_code}.TWF", side=close_side, quantity=plan.quantity,
        order_type="market", time_in_force="ioc", octype="cover",
        allow_live=allow_live, api=api,
    )
    results.append({"kind": "rollover", "leg": "close", "symbol": plan.expiring_code,
                    "side": close_side, "quantity": plan.quantity, "response": close_resp})
    if close_resp.get("status") != "ok":
        results.append({"kind": "rollover_failed", "reason": close_resp.get("error")})
        return results
    open_resp = sdk.place_order(
        None, symbol=f"{plan.next.code}.TWF", side=open_side, quantity=plan.quantity,
        order_type="market", time_in_force="ioc", octype="new",
        allow_live=allow_live, api=api,
    )
    results.append({"kind": "rollover", "leg": "open", "symbol": plan.next.code,
                    "side": open_side, "quantity": plan.quantity, "response": open_resp})
    if open_resp.get("status") != "ok":
        results.append({"kind": "rollover_failed", "reason": open_resp.get("error")})
    return results


def _order_matches(deployment: Deployment, order: dict[str, Any]) -> bool:
    code = str(order.get("symbol") or order.get("code") or "")
    if deployment.market == TW_EQUITY:
        return code.split(".")[0] == deployment.symbol.split(".")[0]
    return contracts.product_of(code) == contracts.product_of(deployment.symbol)


def _equity_snapshot(deployment: Deployment, run_dir: Path, mark_price: float) -> dict[str, Any]:
    snap = accounting.compute_equity(
        market=deployment.market,
        symbol=deployment.symbol,
        allocated_capital=deployment.allocated_capital,
        fills=accounting.load_fills(run_dir),
        mark_price=mark_price,
    )
    return asdict(snap)


def flatten(
    deployment: Deployment, run_dir: Path, *, session_api: Any, now: dt.datetime | None = None,
) -> list[dict[str, Any]]:
    """Stop-and-flatten: market-close whatever the broker holds on this symbol."""
    now = now or market_calendar.now_taipei()
    positions_payload = sdk.get_positions(None, api=session_api)
    positions = list(positions_payload.get("positions") or [])
    current = current_position_qty(positions, deployment, session_api)
    if current == 0:
        return [{"kind": "flatten", "note": "no position"}]
    side = "sell" if current > 0 else "buy"
    planned = _plan_orders(deployment, side, abs(current), session_api,
                           now.astimezone(market_calendar.TAIPEI).date())
    for order in planned:
        order["kind"] = "flatten"
    results = _place_with_refills(deployment, planned, session_api, run_dir)
    accounting.write_tick_record(
        run_dir,
        f"flatten-{now.strftime('%Y%m%dT%H%M%S')}",
        {
            "phase": "final", "status": "ok", "kind": "flatten",
            "deployment_id": deployment.id, "orders": results,
            "executed_at": now.isoformat(),
        },
    )
    return results
