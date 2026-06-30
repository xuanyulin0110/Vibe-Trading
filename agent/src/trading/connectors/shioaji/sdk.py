"""Read-only Shioaji (SinoPac, 永豐金證券) connector via the official ``shioaji`` SDK.

Wraps ``shioaji.Shioaji`` for the read operations the trading layer exposes
(account / positions / orders / quote / history). No order-placement method
is exposed here -- that is a later phase behind the mandate gate, mirroring
how ``connectors/tiger/sdk.py`` layers writes on top of a read-only base.

Unlike Tiger's paper-vs-live split (two different account-number formats
on the SAME connector), Shioaji has one account and a ``simulation``
boolean that selects which trading environment a login session talks to;
market data is the same real feed in both modes (see PREPARE.md/MARKET_DATA.md
in the bundled Shioaji skill). So ``profile`` here just toggles that
boolean -- there is no account-format mismatch to guard against.

Login uses ``api_key``/``secret_key`` only; no CA certificate is needed for
any read operation in this module (CA only gates order placement).
"""

from __future__ import annotations

import datetime as dt
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from src.config.paths import get_runtime_root

CONFIG_FILENAME = "shioaji.json"

#: Lock files older than this are assumed abandoned by a dead/killed process
#: rather than held by a genuinely in-progress download -- see
#: ``_clear_stale_shioaji_locks`` (same fix as ``backtest/loaders/shioaji_loader.py``,
#: duplicated rather than cross-imported since trading/ and backtest/ are
#: separate subsystems by repo convention).
_STALE_LOCK_SECONDS = 120.0


def _clear_stale_shioaji_locks(max_age_seconds: float = _STALE_LOCK_SECONDS) -> None:
    """Remove stale Shioaji contract-cache lock files before login.

    Confirmed empirically: ``shioaji.Shioaji().login()`` writes a
    ``contracts-*.parquet.lock`` file per contract type into ``SJ_HOME_PATH``
    (default ``~/.shioaji``) during the contract download and never removes
    it afterward, even on a clean process exit. A later login then hangs
    indefinitely waiting on a lock no live process holds -- reproduced
    repeatedly in testing. Only locks older than ``max_age_seconds`` are
    removed, so a genuinely concurrent in-progress download is not disturbed.
    """
    home = Path(os.environ.get("SJ_HOME_PATH") or (Path.home() / ".shioaji"))
    if not home.is_dir():
        return
    now = time.time()
    for lock_file in home.glob("*.lock"):
        try:
            if now - lock_file.stat().st_mtime > max_age_seconds:
                lock_file.unlink()
        except OSError:
            pass

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
    """Fetch recent trades/orders for the configured profile (read-only; no order placement here)."""
    cfg = config or load_config()
    api = _login(cfg)
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
    """Fetch a real-time top-of-book snapshot for ``symbol`` (e.g. ``2330.TW``)."""
    cfg = config or load_config()
    api = _login(cfg)
    stock_id = _strip_tw_suffix(symbol)
    contract = api.Contracts.Stocks[stock_id]
    if contract is None:
        _logout_best_effort(api)
        return {"status": "error", "error": f"no Shioaji contract for {symbol} (stock_id={stock_id})"}

    snapshots = _safe_call(api, "snapshots", [contract]) or []
    _logout_best_effort(api)
    if not snapshots:
        return {"status": "error", "error": f"no snapshot returned for {symbol}"}
    return {"status": "ok", "symbol": symbol, "quote": _snapshot_to_dict(snapshots[0])}


#: Canonical period token -> minutes per bar (None = use raw 1-minute bars).
_PERIOD_MINUTES = {
    "1m": 1, "5m": 5, "15m": 15, "30m": 30, "1h": 60, "4h": 240, "1d": None,
}


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
    29 calendar days (see ``backtest/loaders/shioaji_loader.py`` for the same
    chunking rule used in backtesting); this connector fetches the last
    ``limit`` calendar days in chunks and resamples up if ``period`` is
    coarser than 1 minute.
    """
    cfg = config or load_config()
    api = _login(cfg)
    stock_id = _strip_tw_suffix(symbol)
    contract = api.Contracts.Stocks[stock_id]
    if contract is None:
        _logout_best_effort(api)
        return {"status": "error", "error": f"no Shioaji contract for {symbol} (stock_id={stock_id})"}

    minutes = _PERIOD_MINUTES.get(period.strip())
    if period.strip() not in _PERIOD_MINUTES:
        _logout_best_effort(api)
        return {"status": "error", "error": f"unsupported period: {period!r}"}

    end = dt.date.today()
    start = end - dt.timedelta(days=max(int(limit), 1) + 5)  # pad for weekends/holidays
    minute_df = _fetch_minute_kbars(api, contract, start.isoformat(), end.isoformat())
    _logout_best_effort(api)
    if minute_df.empty:
        return {"status": "ok", "symbol": symbol, "period": period, "bars": []}

    bars_df = minute_df if minutes is None and period.strip() != "1d" else _resample(minute_df, period.strip())
    bars_df = bars_df.tail(int(limit))
    return {
        "status": "ok",
        "symbol": symbol,
        "period": period,
        "bars": [_bar_row_to_dict(ts, row) for ts, row in bars_df.iterrows()],
    }


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
    _clear_stale_shioaji_locks()
    api = sj.Shioaji(simulation=cfg.simulation)
    api.login(api_key=cfg.api_key, secret_key=cfg.secret_key, contracts_timeout=cfg.timeout)
    return api


def _logout_best_effort(api: Any) -> None:
    """Log out before returning so the SDK's background connection threads
    wind down promptly instead of lingering past this call's return -- a
    best-effort cleanup, never raises."""
    try:
        api.logout()
    except Exception:  # noqa: BLE001 - logout failure must never mask a real result
        pass


def _strip_tw_suffix(code: str) -> str:
    return code.split(".")[0]


def _fetch_minute_kbars(api: Any, contract: Any, start_date: str, end_date: str) -> pd.DataFrame:
    """Pull 1-minute K-bars in <=29-day chunks (Shioaji's per-request window limit)."""
    chunks: list[pd.DataFrame] = []
    cur = dt.date.fromisoformat(start_date)
    last = dt.date.fromisoformat(end_date)
    step = dt.timedelta(days=28)
    while cur <= last:
        chunk_end = min(cur + step, last)
        kbars = _safe_call(api, "kbars", contract, start=cur.isoformat(), end=chunk_end.isoformat())
        if kbars is not None and kbars.ts:
            chunks.append(pd.DataFrame({
                "open": kbars.Open, "high": kbars.High, "low": kbars.Low,
                "close": kbars.Close, "volume": kbars.Volume,
            }, index=pd.to_datetime(kbars.ts)))
        cur = chunk_end + dt.timedelta(days=1)
    if not chunks:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    return pd.concat(chunks).sort_index()


def _resample(minute_df: pd.DataFrame, period: str) -> pd.DataFrame:
    rule = "1D" if period == "1d" else f"{_PERIOD_MINUTES[period]}min"
    agg = minute_df.resample(rule).agg({
        "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum",
    })
    return agg.dropna(subset=["open", "high", "low", "close"])


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
    return fn(*args, **kwargs)
