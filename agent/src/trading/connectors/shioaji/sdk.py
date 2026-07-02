"""Shioaji (SinoPac, 永豐金證券) connector via the official ``shioaji`` SDK.

Wraps ``shioaji.Shioaji`` for the read operations the trading layer exposes
(account / positions / orders / quote / history), plus futures order
placement/cancellation gated to the ``paper`` (simulation) profile only --
see ``place_order``/``cancel_order`` below.

Unlike Tiger's paper-vs-live split (two different account-number formats
on the SAME connector), Shioaji has one account and a ``simulation``
boolean that selects which trading environment a login session talks to;
market data is the same real feed in both modes (see PREPARE.md/MARKET_DATA.md
in the bundled Shioaji skill). So ``profile`` here just toggles that
boolean -- there is no account-format mismatch to guard against.

Login uses ``api_key``/``secret_key`` only; no CA certificate is needed for
any read operation, nor for simulation order placement (Shioaji skips CA
signing automatically in simulation -- see ORDERS.md "Prerequisites" in the
bundled Shioaji skill). Real-money order placement (``live-readonly`` ->
some future ``live`` profile) needs CA activation, mandate-gate wiring, and
a futures-aware ``InstrumentType``/``AssetClass`` in ``src/live/mandate``
(none of that exists yet) -- ``place_order``/``cancel_order`` here
deliberately refuse any non-simulation profile until that's built.
"""

from __future__ import annotations

import datetime as dt
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from backtest.loaders._shioaji_kbars import (
    clear_stale_shioaji_locks,
    fetch_minute_kbars,
    is_supported_interval,
    resample_kbars,
    suppress_native_stdout,
)
from src.config.paths import get_runtime_root

CONFIG_FILENAME = "shioaji.json"

#: Profiles this connector understands and their ``simulation`` flag.
PROFILE_SIMULATION = {
    "paper": True,
    "live-readonly": False,
}


class ShioajiDependencyError(RuntimeError):
    """Raised when the optional ``shioaji`` package is not installed."""


class ShioajiConfigError(RuntimeError):
    """Raised when the connector configuration is missing or invalid."""


@dataclass(frozen=True)
class ShioajiConfig:
    """Shioaji connector connection settings.

    Args:
        api_key: API key from https://www.sinotrade.com.tw/newweb/PythonAPIKey/.
        secret_key: Secret key issued alongside the API key.
        profile: ``paper`` (simulation) or ``live-readonly`` (production, read-only).
        timeout: Contract-load timeout in milliseconds.
    """

    api_key: str = ""
    secret_key: str = ""
    profile: str = "paper"
    timeout: int = 10000

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None = None) -> "ShioajiConfig":
        """Build a config from a JSON-like mapping, normalizing the profile."""
        payload = dict(data or {})
        profile = str(payload.get("profile") or "paper").strip().lower()
        if profile not in PROFILE_SIMULATION:
            raise ShioajiConfigError("profile must be 'paper' or 'live-readonly'")
        return cls(
            api_key=str(payload.get("api_key") or "").strip(),
            secret_key=str(payload.get("secret_key") or "").strip(),
            profile=profile,
            timeout=int(payload.get("timeout") or 10000),
        )

    def with_overrides(
        self,
        *,
        api_key: str | None = None,
        secret_key: str | None = None,
        profile: str | None = None,
    ) -> "ShioajiConfig":
        """Return a copy with CLI/tool overrides applied."""
        payload = asdict(self)
        if api_key is not None:
            payload["api_key"] = api_key
        if secret_key is not None:
            payload["secret_key"] = secret_key
        if profile is not None:
            payload["profile"] = profile
        return ShioajiConfig.from_mapping(payload)

    @property
    def simulation(self) -> bool:
        """Return whether this profile logs in against the simulation environment."""
        return PROFILE_SIMULATION.get(self.profile, True)


_OVERRIDE_KEYS = ("api_key", "secret_key", "profile")


def build_config(
    profile_config: Mapping[str, Any] | None = None, overrides: Mapping[str, Any] | None = None,
) -> "ShioajiConfig":
    """Resolve the effective config: saved file <- profile defaults <- CLI overrides."""
    base = asdict(load_config())
    for key, value in dict(profile_config or {}).items():
        if value is not None:
            base[key] = value
    cfg = ShioajiConfig.from_mapping(base)
    clean = {k: v for k, v in dict(overrides or {}).items() if k in _OVERRIDE_KEYS and v not in (None, "")}
    return cfg.with_overrides(**clean) if clean else cfg


def config_path() -> Path:
    """Return the user-level Shioaji config path."""
    return get_runtime_root() / CONFIG_FILENAME


def load_config() -> ShioajiConfig:
    """Load Shioaji settings from ``~/.vibe-trading/shioaji.json``."""
    path = config_path()
    if not path.exists():
        return ShioajiConfig()
    try:
        return ShioajiConfig.from_mapping(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise ShioajiConfigError(f"invalid Shioaji config at {path}: {exc}") from exc


def save_config(config: ShioajiConfig) -> Path:
    """Persist Shioaji settings with owner-only permissions."""
    path = config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(config), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    try:
        path.chmod(0o600)
    except OSError:
        pass
    return path


def shioaji_available() -> bool:
    """Return whether the optional ``shioaji`` SDK can be imported."""
    try:
        _require_shioaji()
        return True
    except ShioajiDependencyError:
        return False


def check_status(config: ShioajiConfig | None = None) -> dict[str, Any]:
    """Check SDK readiness, config completeness, and login.

    Returns a JSON-serializable health report. Does not place or mutate any
    broker state.
    """
    cfg = config or load_config()
    report: dict[str, Any] = {
        "status": "ok",
        "config": _public_config(cfg),
        "sdk": {"package": "shioaji", "installed": shioaji_available()},
    }

    missing = _missing_fields(cfg)
    if missing:
        report["status"] = "error"
        report["error"] = f"Shioaji connector not configured: missing {', '.join(missing)}."
        return report

    if not report["sdk"]["installed"]:
        report["status"] = "error"
        report["error"] = "Optional dependency missing: install with `pip install shioaji`."
        return report

    try:
        api = _login(cfg)
    except Exception as exc:  # noqa: BLE001 - health endpoint reports cleanly
        report["status"] = "error"
        report["error"] = str(exc)
        return report

    report["account"] = {
        "profile": cfg.profile,
        "simulation": cfg.simulation,
        "accounts": [_account_to_dict(a) for a in _safe_call(api, "list_accounts") or []],
    }
    _logout_best_effort(api)
    return report


def get_account_snapshot(config: ShioajiConfig | None = None) -> dict[str, Any]:
    """Fetch stock account balance for the configured profile.

    Simulation mode returns a default/zero balance per the SDK -- this is
    not real buying power and must not be treated as such (see PREPARE.md's
    "Features with Simulation Guards" in the bundled Shioaji skill).
    """
    cfg = config or load_config()
    api = _login(cfg)
    balance = _safe_call(api, "account_balance")
    result = {
        "status": "ok",
        "profile": cfg.profile,
        "simulation": cfg.simulation,
        "balance": _balance_to_dict(balance),
    }
    _logout_best_effort(api)
    return result


def get_positions(config: ShioajiConfig | None = None) -> dict[str, Any]:
    """Fetch current stock positions for the configured profile."""
    cfg = config or load_config()
    api = _login(cfg)
    positions = _safe_call(api, "list_positions") or []
    result = {
        "status": "ok",
        "profile": cfg.profile,
        "simulation": cfg.simulation,
        "positions": [_position_to_dict(item) for item in positions],
    }
    _logout_best_effort(api)
    return result


def get_open_orders(
    config: ShioajiConfig | None = None, *, include_executions: bool = False,
) -> dict[str, Any]:
    """Fetch recent trades/orders for the configured profile."""
    cfg = config or load_config()
    api = _login(cfg)
    _sync_trade_status(api)
    trades = _safe_call(api, "list_trades") or []
    rows = [_trade_to_dict(item) for item in trades]
    result: dict[str, Any] = {
        "status": "ok",
        "profile": cfg.profile,
        "simulation": cfg.simulation,
        "open_orders": [r for r in rows if r["status"] not in ("Filled", "Cancelled")],
    }
    if include_executions:
        result["executions"] = [r for r in rows if r["status"] == "Filled"]
    _logout_best_effort(api)
    return result


def get_quote(symbol: str, *, config: ShioajiConfig | None = None, **_: Any) -> dict[str, Any]:
    """Fetch a real-time top-of-book snapshot for ``symbol``.

    Accepts TW equities (``2330.TW``) and TAIFEX index futures
    (``TXFR1.TWF``); routing is by suffix (see ``_resolve_contract``).
    """
    cfg = config or load_config()
    api = _login(cfg)
    contract = _resolve_contract(api, symbol)
    if contract is None:
        _logout_best_effort(api)
        return {"status": "error", "error": f"no Shioaji contract for {symbol}"}

    snapshots = _safe_call(api, "snapshots", [contract]) or []
    _logout_best_effort(api)
    if not snapshots:
        return {"status": "error", "error": f"no snapshot returned for {symbol}"}
    return {"status": "ok", "symbol": symbol, "quote": _snapshot_to_dict(snapshots[0])}


def get_historical_bars(
    symbol: str,
    *,
    config: ShioajiConfig | None = None,
    period: str = "1d",
    limit: int = 90,
    **_: Any,
) -> dict[str, Any]:
    """Fetch historical OHLCV bars for ``symbol`` via Shioaji K-bars, resampled to ``period``.

    Shioaji's ``kbars()`` only returns 1-minute bars and caps each request at
    29 calendar days; the shared ``backtest.loaders._shioaji_kbars`` helper
    handles the chunking and resampling (same code path the backtest loaders
    use). ``period`` accepts 1m/5m/15m/30m/1h/4h/1d.
    """
    cfg = config or load_config()
    api = _login(cfg)
    contract = _resolve_contract(api, symbol)
    if contract is None:
        _logout_best_effort(api)
        return {"status": "error", "error": f"no Shioaji contract for {symbol}"}

    if not is_supported_interval(period):
        _logout_best_effort(api)
        return {"status": "error", "error": f"unsupported period: {period!r}"}

    end = dt.date.today()
    start = end - dt.timedelta(days=max(int(limit), 1) + 5)  # pad for weekends/holidays
    minute_df = fetch_minute_kbars(api, contract, start.isoformat(), end.isoformat())
    _logout_best_effort(api)
    if minute_df.empty:
        return {"status": "ok", "symbol": symbol, "period": period, "bars": []}

    is_futures = symbol.upper().endswith(".TWF")
    bars_df = resample_kbars(minute_df, period, session_aware=is_futures).tail(int(limit))
    return {
        "status": "ok",
        "symbol": symbol,
        "period": period,
        "bars": [_bar_row_to_dict(ts, row) for ts, row in bars_df.iterrows()],
    }


#: TAIFEX rejects MKT+ROD (op_code 9938) -- market orders must be IOC/FOK.
_FUTURES_TIME_IN_FORCE = ("rod", "ioc", "fok")

#: Returned by order methods when a non-simulation config reaches them.
#: Shioaji exposes no runtime paper/live discriminator (same account, a
#: ``simulation`` boolean picks the environment), so -- following the
#: Longbridge/Dhan/Shoonya precedent -- the connector is structurally capped
#: at paper until CA activation + mandate-gate wiring exists for a real live
#: order path.
_PAPER_ONLY_ERROR = (
    "Shioaji connector is paper-only for order placement: it exposes no "
    "runtime paper/live discriminator, and real-money orders need CA "
    "activation + mandate-gate wiring that doesn't exist yet. Use the "
    "'paper' (simulation) profile."
)


def place_order(
    config: ShioajiConfig | None = None,
    *,
    symbol: str,
    side: str,
    quantity: float | None = None,
    notional: float | None = None,
    order_type: str = "market",
    limit_price: float | None = None,
    time_in_force: str = "rod",
    octype: str = "auto",
) -> dict[str, Any]:
    """Submit a TAIFEX futures order against the configured Shioaji account.

    Simulation-only for now: refuses any profile where ``cfg.simulation`` is
    ``False`` (i.e. ``live-readonly``), since real-money order placement
    needs CA activation and mandate-gate wiring that doesn't exist yet (see
    the module docstring). Only futures symbols (``*.TWF``) are supported --
    TW equity order placement is a separate future extension.

    ``order_status`` in the returned dict is whatever the exchange
    acknowledged synchronously when ``place_order`` returned -- commonly
    ``PendingSubmit``, sometimes a same-tick rejection (e.g. price outside
    the daily limit band). This function does NOT poll for a settled status
    (see the ``update_status`` note inline) -- call :func:`get_open_orders`
    afterward, as a separate call, to see whether the order was accepted,
    rejected, or filled.

    Args:
        config: Connector config; falls back to the saved config when ``None``.
        symbol: Futures symbol, e.g. ``TXFR1.TWF``.
        side: ``buy`` or ``sell``.
        quantity: Number of contracts (lots). Required -- Shioaji futures
            orders are sized by contract count, not notional.
        notional: Not supported for futures; passing this is an error.
        order_type: ``market`` or ``limit``.
        limit_price: Required when ``order_type`` is ``limit``.
        time_in_force: ``rod`` (rest-of-day), ``ioc``, or ``fok``. A
            ``market`` order must use ``ioc``/``fok`` -- TAIFEX rejects
            market+rod.
        octype: ``auto`` (default, recommended), ``new``, ``cover``, or
            ``daytrade`` -- open/close type.

    Returns:
        On success ``{"status": "ok", "order_id", "symbol", "side", "profile",
        "simulation", "order_type", "time_in_force", "octype", "quantity",
        "limit_price", "order_status", "filled_qty"}``. On invalid input,
        submission failure, or a non-simulation profile, ``{"status":
        "error", "error": <message>}`` -- fails closed, never raises for
        caller-controlled mistakes.
    """
    cfg = config or load_config()
    if not cfg.simulation:
        return {"status": "error", "error": _PAPER_ONLY_ERROR}

    clean_symbol = str(symbol or "").strip()
    if not clean_symbol.upper().endswith(".TWF"):
        return {"status": "error", "error": "only *.TWF futures symbols are supported"}

    side_token = str(side or "").strip().lower()
    if side_token not in ("buy", "sell"):
        return {"status": "error", "error": "side must be 'buy' or 'sell'"}

    if notional is not None:
        return {"status": "error", "error": "notional is not supported for futures; provide quantity"}
    if quantity is None:
        return {"status": "error", "error": "quantity is required"}
    try:
        qty_value = int(quantity)
    except (TypeError, ValueError):
        return {"status": "error", "error": "quantity must be an integer contract count"}
    if qty_value <= 0:
        return {"status": "error", "error": "quantity must be positive"}

    type_token = str(order_type or "").strip().lower()
    if type_token not in ("market", "limit"):
        return {"status": "error", "error": "order_type must be 'market' or 'limit'"}

    tif_token = str(time_in_force or "").strip().lower()
    if tif_token not in _FUTURES_TIME_IN_FORCE:
        return {"status": "error", "error": "time_in_force must be 'rod', 'ioc', or 'fok'"}
    if type_token == "market" and tif_token == "rod":
        return {"status": "error", "error": "a market order must use time_in_force 'ioc' or 'fok' (TAIFEX rejects market+rod)"}

    octype_token = str(octype or "").strip().lower()
    if octype_token not in ("auto", "new", "cover", "daytrade"):
        return {"status": "error", "error": "octype must be 'auto', 'new', 'cover', or 'daytrade'"}

    limit_value: float | None = None
    if type_token == "limit":
        if limit_price is None:
            return {"status": "error", "error": "limit order requires limit_price"}
        try:
            limit_value = float(limit_price)
        except (TypeError, ValueError):
            return {"status": "error", "error": "limit_price must be numeric"}
        if limit_value <= 0:
            return {"status": "error", "error": "limit_price must be positive"}

    sj = _require_shioaji()
    api = _login(cfg)
    try:
        contract = _resolve_contract(api, clean_symbol)
        if contract is None:
            return {"status": "error", "error": f"no Shioaji contract for {clean_symbol}"}

        account = _futopt_account(api)
        if account is None:
            return {"status": "error", "error": "no futures account signed in for this profile"}

        order = sj.FuturesOrder(
            price=limit_value if limit_value is not None else 0,
            quantity=qty_value,
            action=sj.Action.Buy if side_token == "buy" else sj.Action.Sell,
            price_type=sj.FuturesPriceType.LMT if type_token == "limit" else sj.FuturesPriceType.MKT,
            order_type={"rod": sj.OrderType.ROD, "ioc": sj.OrderType.IOC, "fok": sj.OrderType.FOK}[tif_token],
            octype={
                "auto": sj.FuturesOCType.Auto,
                "new": sj.FuturesOCType.New,
                "cover": sj.FuturesOCType.Cover,
                "daytrade": sj.FuturesOCType.DayTrade,
            }[octype_token],
            account=account,
        )
        with suppress_native_stdout():
            trade = api.place_order(contract, order)
            # Do NOT call api.update_status(trade=trade) here: confirmed live
            # (2026-07-02) that calling it immediately after place_order races
            # the SDK's own internal order-event handling and panics the Rust
            # extension ("Already borrowed: PyBorrowMutError") -- a
            # pyo3_runtime.PanicException, which subclasses BaseException
            # directly, not Exception, so it is NOT caught below and would
            # crash the whole process. trade.status.status is whatever the
            # exchange acknowledged synchronously (commonly PendingSubmit);
            # callers must poll get_open_orders() in a separate call for the
            # settled status (see ORDERS.md "Order Status").
    except Exception as exc:  # noqa: BLE001 - submission errors are reported, not raised
        return {"status": "error", "error": str(exc)}
    finally:
        _logout_best_effort(api)

    row = _trade_to_dict(trade)
    return {
        "status": "ok",
        "order_id": row["order_id"],
        "symbol": clean_symbol,
        "side": side_token,
        "profile": cfg.profile,
        "simulation": cfg.simulation,
        "order_type": type_token,
        "time_in_force": tif_token,
        "octype": octype_token,
        "quantity": qty_value,
        "limit_price": limit_value,
        "order_status": row["status"],
        "filled_qty": _obj_get(_obj_get(trade, "status"), "deal_quantity"),
    }


def cancel_order(
    config: ShioajiConfig | None = None,
    order_id: str = "",
    *,
    symbol: str | None = None,
) -> dict[str, Any]:
    """Cancel an open futures order on the configured Shioaji account.

    Simulation-only, same guard as :func:`place_order`. Shioaji's
    ``cancel_order`` takes a live ``Trade`` object, not a bare id, so this
    looks the order up via ``list_trades()`` by ``trade.order.id`` first --
    synced from the server first via ``update_status`` since this connector
    logs in fresh per call (see ``_login``): confirmed live (2026-07-02)
    that a plain ``list_trades()`` in a brand-new session is empty even for
    an order placed moments earlier in a different session/call.

    Args:
        config: Connector config; falls back to the saved config when ``None``.
        order_id: The Shioaji ``trade.order.id`` to cancel.
        symbol: Optional symbol, echoed back for caller bookkeeping only;
            lookup is purely by ``order_id``.

    Returns:
        On success ``{"status": "ok", "order_id", "symbol", "profile",
        "simulation", "cancelled"}``. On invalid input, an unmatched
        ``order_id``, cancel failure, or a non-simulation profile,
        ``{"status": "error", "error": <message>}`` -- fails closed, never
        raises.
    """
    cfg = config or load_config()
    if not cfg.simulation:
        return {"status": "error", "error": _PAPER_ONLY_ERROR}

    clean_id = str(order_id or "").strip()
    if not clean_id:
        return {"status": "error", "error": "order_id is required"}

    api = _login(cfg)
    try:
        _sync_trade_status(api)
        trades = _safe_call(api, "list_trades") or []
        target = next((t for t in trades if str(_obj_get(_obj_get(t, "order"), "id")) == clean_id), None)
        if target is None:
            return {"status": "error", "error": f"no open trade found for order_id {clean_id}"}
        with suppress_native_stdout():
            api.cancel_order(target)
    except Exception as exc:  # noqa: BLE001 - cancel errors are reported, not raised
        return {"status": "error", "error": str(exc)}
    finally:
        _logout_best_effort(api)

    return {
        "status": "ok",
        "order_id": clean_id,
        "symbol": symbol.strip() if isinstance(symbol, str) and symbol.strip() else None,
        "profile": cfg.profile,
        "simulation": cfg.simulation,
        "cancelled": True,
    }


def _futopt_account(api: Any) -> Any:
    """Return the logged-in session's futures/options account, or ``None``."""
    return getattr(api, "futopt_account", None)


def _sync_trade_status(api: Any) -> None:
    """Pull server-side order/deal state into this session's local trade cache.

    ``_login`` logs in fresh for every call (no session reuse across
    connector functions), so ``list_trades()`` reflects only what happened
    within the *current* session -- confirmed live (2026-07-02) that an
    order placed by a prior call is invisible to a new session's
    ``list_trades()`` without this sync first. Best-effort per account
    (``update_status`` failures must never block the caller from reading
    whatever the cache already has).
    """
    for account_attr in ("stock_account", "futopt_account"):
        account = getattr(api, account_attr, None)
        if account is not None:
            _safe_call(api, "update_status", account)


# ---------------------------------------------------------------------------
# SDK plumbing
# ---------------------------------------------------------------------------


def _require_shioaji():
    try:
        import shioaji
    except ModuleNotFoundError as exc:
        raise ShioajiDependencyError("shioaji is not installed; run `pip install shioaji`.") from exc
    return shioaji


def _login(cfg: ShioajiConfig):
    """Log in fresh for each call.

    No session caching here -- see ``backtest/loaders/shioaji_loader.py``'s
    ``_ensure_logged_in`` docstring for why eager/repeated logins within one
    process raced on the SDK's on-disk contract-cache lock files. Each
    top-level connector call here logs in once, so that race does not apply,
    but a caller that wants to batch multiple reads should reuse a single
    ``_login()`` result rather than calling these functions in a tight loop.
    """
    missing = _missing_fields(cfg)
    if missing:
        raise ShioajiConfigError(f"Shioaji connector not configured: missing {', '.join(missing)}.")
    sj = _require_shioaji()
    clear_stale_shioaji_locks()
    with suppress_native_stdout():
        api = sj.Shioaji(simulation=cfg.simulation)
        api.login(api_key=cfg.api_key, secret_key=cfg.secret_key, contracts_timeout=cfg.timeout)
    return api


def _logout_best_effort(api: Any) -> None:
    """Log out before returning so the SDK's background connection threads
    wind down promptly instead of lingering past this call's return -- a
    best-effort cleanup, never raises."""
    try:
        with suppress_native_stdout():
            api.logout()
    except Exception:  # noqa: BLE001 - logout failure must never mask a real result
        pass


def _strip_tw_suffix(code: str) -> str:
    return code.split(".")[0]


#: Known TAIFEX index-futures product categories under ``Contracts.Futures``.
_TW_FUTURES_PRODUCTS = ("TXF", "MXF", "TMF")


def _resolve_contract(api: Any, symbol: str):
    """Resolve a symbol to a Shioaji contract, routing by suffix.

    ``*.TWF`` -> ``api.Contracts.Futures.<PRODUCT>.<contract>`` (e.g.
    ``TXFR1.TWF`` -> ``Futures.TXF.TXFR1``); anything else -> the equity
    lookup ``api.Contracts.Stocks[stock_id]``. Returns ``None`` when the
    contract cannot be found.
    """
    if symbol.upper().endswith(".TWF"):
        contract_code = _strip_tw_suffix(symbol).upper()
        product = next((p for p in _TW_FUTURES_PRODUCTS if contract_code.startswith(p)), contract_code[:3])
        category = getattr(api.Contracts.Futures, product, None)
        if category is None:
            return None
        contract = getattr(category, contract_code, None)
        if contract is None:
            try:
                contract = category[contract_code]
            except Exception:
                contract = None
        return contract
    return api.Contracts.Stocks[_strip_tw_suffix(symbol)]


def _public_config(cfg: ShioajiConfig) -> dict[str, Any]:
    """Config snapshot with secrets redacted."""
    data = asdict(cfg)
    if data.get("secret_key"):
        data["secret_key"] = "***redacted***"
    if data.get("api_key"):
        data["api_key"] = data["api_key"][:4] + "***"
    return data


def _missing_fields(cfg: ShioajiConfig) -> list[str]:
    missing = []
    if not cfg.api_key:
        missing.append("api_key")
    if not cfg.secret_key:
        missing.append("secret_key")
    return missing


# ---------------------------------------------------------------------------
# Defensive field extraction (SDK returns objects; attribute names vary)
# ---------------------------------------------------------------------------


def _obj_get(obj: Any, name: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _account_to_dict(item: Any) -> dict[str, Any]:
    return {
        "account_type": _obj_get(item, "account_type"),
        "broker_id": _obj_get(item, "broker_id"),
        "account_id": _obj_get(item, "account_id"),
        "signed": _obj_get(item, "signed"),
    }


def _balance_to_dict(item: Any) -> dict[str, Any]:
    return {
        "acc_balance": _obj_get(item, "acc_balance"),
        "date": _obj_get(item, "date"),
        "errmsg": _obj_get(item, "errmsg"),
    }


def _position_to_dict(item: Any) -> dict[str, Any]:
    return {
        "code": _obj_get(item, "code"),
        "direction": str(_obj_get(item, "direction") or ""),
        "quantity": _obj_get(item, "quantity"),
        "price": _obj_get(item, "price"),
        "pnl": _obj_get(item, "pnl"),
        "last_price": _obj_get(item, "last_price"),
    }


def _trade_to_dict(item: Any) -> dict[str, Any]:
    order = _obj_get(item, "order")
    status = _obj_get(item, "status")
    contract = _obj_get(item, "contract")
    return {
        "order_id": _obj_get(order, "id"),
        "symbol": _obj_get(contract, "code"),
        "action": str(_obj_get(order, "action") or ""),
        "quantity": _obj_get(order, "quantity"),
        "price": _obj_get(order, "price"),
        "status": str(_obj_get(status, "status") or ""),
    }


def _snapshot_to_dict(item: Any) -> dict[str, Any]:
    return {
        "symbol": _obj_get(item, "code"),
        "last": _obj_get(item, "close"),
        "bid": _obj_get(item, "buy_price"),
        "ask": _obj_get(item, "sell_price"),
        "open": _obj_get(item, "open"),
        "high": _obj_get(item, "high"),
        "low": _obj_get(item, "low"),
        "volume": _obj_get(item, "total_volume"),
        "change_price": _obj_get(item, "change_price"),
        "change_rate": _obj_get(item, "change_rate"),
        "time": str(_obj_get(item, "ts") or ""),
    }


def _bar_row_to_dict(ts: pd.Timestamp, row: pd.Series) -> dict[str, Any]:
    return {
        "time": str(ts),
        "open": float(row["open"]),
        "high": float(row["high"]),
        "low": float(row["low"]),
        "close": float(row["close"]),
        "volume": float(row["volume"]),
    }


def _safe_call(obj: Any, name: str, *args: Any, **kwargs: Any) -> Any:
    """Call ``obj.name(*args, **kwargs)`` if it exists, else return ``None``."""
    fn = getattr(obj, name, None)
    if fn is None:
        return None
    with suppress_native_stdout():
        return fn(*args, **kwargs)
