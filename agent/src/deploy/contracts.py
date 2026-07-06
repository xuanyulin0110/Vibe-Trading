"""Continuous-alias -> actual-contract resolution and rollover planning.

Backtests run on the continuous near-month series (``TXFR1.TWF``,
splice-back-adjusted -- the latest segment is real, unadjusted prices), but
orders fill on a dated contract and broker positions come back keyed by the
dated code. This module owns that translation plus the settlement-day
rollover:

* Resolution: pick the dated contract with the nearest delivery month whose
  settlement date is still >= today, EXCEPT on settlement day itself, where
  new exposure must go to the next month (never open new positions in a
  contract that stops trading at 13:30 today).
* Rollover: on settlement day, an existing position in the expiring month is
  closed and re-opened same-direction/same-size in the next month, recorded
  as ``rollover`` trades (they carry PnL but are excluded from signal-trade
  statistics). Timed near the backtest's splice point (settlement-day 13:30)
  so live behavior matches the continuous series' roll timing.
"""

from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass
from typing import Any

from src.deploy.market_calendar import taifex_settlement_date

_PRODUCTS = ("TXF", "MXF", "TMF")

#: Contract multiplier, NT$ per index point -- keep in sync with
#: ``backtest.engines.tw_futures._MULTIPLIER`` (imported there for sizing;
#: duplicated here only for docstring completeness of order records).
CONTINUOUS_SUFFIXES = ("R1", "R2")


def product_of(symbol_or_code: str) -> str:
    """``TXFR1.TWF``/``TXF202607``/``TXFG6`` -> ``TXF``."""
    code = symbol_or_code.split(".")[0].upper()
    for p in _PRODUCTS:
        if code.startswith(p):
            return p
    m = re.match(r"([A-Z]+)", code)
    return m.group(1) if m else code


def is_continuous(symbol: str) -> bool:
    code = symbol.split(".")[0].upper()
    return any(code == product_of(code) + s for s in CONTINUOUS_SUFFIXES)


def _delivery_month_of(contract: Any) -> str | None:
    value = getattr(contract, "delivery_month", None)
    if value is None and isinstance(contract, dict):
        value = contract.get("delivery_month")
    text = str(value or "").strip()
    return text if re.fullmatch(r"\d{6}", text) else None


def settlement_of_delivery(delivery_month: str) -> dt.date:
    """Final settlement date for a ``YYYYMM`` delivery month."""
    return taifex_settlement_date(int(delivery_month[:4]), int(delivery_month[4:6]))


def dated_contracts(api: Any, product: str) -> list[Any]:
    """All dated (non-alias) contracts of a product category, sorted by delivery."""
    category = getattr(api.Contracts.Futures, product, None)
    if category is None:
        return []
    seen: dict[str, Any] = {}
    for contract in _iter_category(category):
        month = _delivery_month_of(contract)
        if month:
            seen.setdefault(month, contract)
    return [seen[m] for m in sorted(seen)]


def _iter_category(category: Any):
    try:
        yield from category
        return
    except TypeError:
        pass
    for name in dir(category):
        if name.startswith("_"):
            continue
        value = getattr(category, name, None)
        if value is not None and _delivery_month_of(value):
            yield value


@dataclass(frozen=True)
class ResolvedContract:
    contract: Any
    code: str
    delivery_month: str
    settlement_date: dt.date


def resolve_order_contract(api: Any, symbol: str, today: dt.date) -> ResolvedContract:
    """Resolve the dated contract new orders for ``symbol`` must use today.

    Continuous aliases pick the nearest delivery whose settlement is strictly
    after today; ON settlement day the expiring month is already excluded
    (``settlement > today``), which is exactly the "no new positions in the
    expiring contract on settlement day" rule. A dated symbol resolves to
    itself (with a guard against ordering an already-expired contract).
    """
    product = product_of(symbol)
    candidates = dated_contracts(api, product)
    if not candidates:
        raise LookupError(f"no dated {product} contracts available from Shioaji")

    bare = symbol.split(".")[0].upper()
    if not is_continuous(symbol):
        for contract in candidates:
            code = str(getattr(contract, "code", "")).upper()
            month = _delivery_month_of(contract) or ""
            if code == bare or bare.endswith(month):
                settlement = settlement_of_delivery(month)
                if settlement < today:
                    raise LookupError(f"{bare} expired on {settlement}")
                return ResolvedContract(contract, code, month, settlement)
        raise LookupError(f"no Shioaji contract matches {symbol}")

    for contract in candidates:  # sorted by delivery month
        month = _delivery_month_of(contract) or ""
        settlement = settlement_of_delivery(month)
        if settlement > today:
            return ResolvedContract(
                contract, str(getattr(contract, "code", "")).upper(), month, settlement,
            )
    raise LookupError(f"no unexpired {product} contract found")


@dataclass(frozen=True)
class RolloverPlan:
    """Close ``expiring_code`` and reopen the same exposure in ``next``."""

    product: str
    expiring_code: str
    direction: int  # +1 long / -1 short
    quantity: int
    next: ResolvedContract


def plan_rollover(
    api: Any, symbol: str, positions: list[dict[str, Any]], today: dt.date,
) -> RolloverPlan | None:
    """Return the rollover needed today for ``symbol``'s product, if any.

    Scans broker positions for the product's DATED codes whose settlement is
    today (or earlier -- a missed roll still gets moved). Positions in other
    later months are left alone (they belong to reconciliation warnings, not
    auto-rollover).
    """
    product = product_of(symbol)
    for pos in positions:
        code = str(pos.get("symbol") or pos.get("code") or "").upper()
        if product_of(code) != product or is_continuous(code):
            continue
        month = _month_from_position_code(api, product, code)
        if month is None:
            continue
        if settlement_of_delivery(month) > today:
            continue
        qty = int(abs(float(pos.get("quantity") or 0)))
        if qty == 0:
            continue
        direction = int(pos.get("direction") or 0) or (
            1 if str(pos.get("side", "")).lower() in ("buy", "long") else -1
        )
        return RolloverPlan(
            product=product,
            expiring_code=code,
            direction=direction,
            quantity=qty,
            next=resolve_order_contract(api, f"{product}R1.TWF", today),
        )
    return None


def _month_from_position_code(api: Any, product: str, code: str) -> str | None:
    """Delivery month for a broker position code, via contract metadata.

    Broker position codes are dated contract codes (e.g. ``TXFG6``); match
    them against the category's contracts rather than parsing the month-letter
    encoding by hand.
    """
    for contract in dated_contracts(api, product):
        if str(getattr(contract, "code", "")).upper() == code:
            return _delivery_month_of(contract)
    m = re.search(r"(\d{6})", code)
    return m.group(1) if m else None
