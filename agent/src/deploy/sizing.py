"""Target-position sizing -- the SAME arithmetic the backtest engines use.

Fidelity rule: live sizing must not re-implement the weight->quantity math.
This module instantiates the real backtest engines (``TWEquityEngine``,
``TWFuturesEngine``) and calls their ``_calc_raw_size``/``round_size``/
``_calc_margin``, so board-lot rounding, contract multipliers, and the
fixed-per-contract TAIFEX margin table stay single-sourced. A quantity that
rounds to zero is reported with an explicit reason -- the silent-skip failure
mode found in this project's own backtest engine (2026-07-05) must never
recur here.
"""

from __future__ import annotations

from dataclasses import dataclass

from backtest.engines.tw_equity import TWEquityEngine
from backtest.engines.tw_futures import TWFuturesEngine

from src.deploy.market_calendar import TW_EQUITY, TW_FUTURES


@dataclass(frozen=True)
class SizingResult:
    """Signed target quantity (shares or contracts) plus the how/why."""

    target_qty: int  # signed: >0 long, <0 short, 0 flat
    reason: str
    notional: float  # TWD value of the target exposure at `price`
    margin_required: float  # TWD (futures: fixed per-contract; equity: == notional)


def target_quantity(
    *,
    market: str,
    symbol: str,
    weight: float,
    allocated_capital: float,
    price: float,
) -> SizingResult:
    """Convert a signal weight into a signed target quantity.

    Args:
        market: ``tw_equity`` or ``tw_futures``.
        symbol: Deployment symbol (product extraction for futures tables).
        weight: Last complete bar's signal value, already in [-1, 1].
        allocated_capital: This deployment's capital slice (TWD) -- NOT the
            whole account, so multiple deployments sharing one account stay
            cleanly attributed.
        price: Latest real price (for continuous futures the newest segment
            is unadjusted, so this is directly tradable).

    Returns:
        SizingResult. ``target_qty == 0`` always carries a human-readable
        ``reason`` (flat signal vs. can't afford one lot/contract are very
        different situations for the operator).
    """
    if price <= 0:
        return SizingResult(0, "no usable price", 0.0, 0.0)
    weight = max(-1.0, min(1.0, float(weight)))

    if market == TW_EQUITY:
        engine = TWEquityEngine({"initial_cash": allocated_capital})
        if weight < 0:
            # Backtest tw_equity blocks shorts (can_execute direction == -1);
            # live mirrors that: a short signal flattens, never shorts.
            return SizingResult(0, "short signal on equity -> flat (no short selling, matches backtest)", 0.0, 0.0)
        target_notional = weight * allocated_capital
        raw = engine._calc_raw_size(symbol, target_notional, price)
        shares = int(engine.round_size(raw, price))
        if weight > 0 and shares == 0:
            return SizingResult(
                0,
                f"weight {weight:.4f} x capital {allocated_capital:,.0f} can't afford one 1,000-share lot at {price:,.2f}",
                0.0,
                0.0,
            )
        notional = shares * price
        return SizingResult(shares, "ok" if shares else "flat signal", notional, notional)

    if market == TW_FUTURES:
        engine = TWFuturesEngine({"initial_cash": allocated_capital})
        target_notional = abs(weight) * allocated_capital
        raw = engine._calc_raw_size(symbol, target_notional, price)
        contracts = int(engine.round_size(raw, price))
        margin_per = engine.get_margin_per_contract(symbol)
        # Mirror the engine's capital check: margin must fit allocated capital.
        if contracts > 0 and contracts * margin_per > allocated_capital:
            contracts = int(allocated_capital // margin_per)
        if abs(weight) > 1e-9 and contracts == 0:
            return SizingResult(
                0,
                f"weight {weight:.4f} x capital {allocated_capital:,.0f} affords 0 contracts "
                f"(notional/point x price or margin {margin_per:,.0f}/contract too large)",
                0.0,
                0.0,
            )
        signed = contracts if weight > 0 else -contracts
        cm = engine.get_contract_multiplier(symbol)
        return SizingResult(
            signed,
            "ok" if contracts else "flat signal",
            contracts * price * cm,
            contracts * margin_per,
        )

    raise ValueError(f"unknown market {market!r}")
