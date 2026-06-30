"""Tests for the direct-SDK trading connectors (Tiger, Longbridge).

Layer A is read-only; these tests exercise the parts that do not require the
optional broker SDKs or live credentials: profile registration, the paper/live
identity guard, config resolution, read/write classification, secret redaction,
and the service dispatch degrading cleanly when nothing is configured.
"""

from __future__ import annotations

import pytest

from src.live.classification import ToolClass
from src.trading import profiles, service
from src.trading.connectors.alpaca import sdk as al
from src.trading.connectors.alpaca.classification import ALPACA_TOOL_CLASS
from src.trading.connectors.binance import sdk as bn
from src.trading.connectors.binance.classification import BINANCE_TOOL_CLASS
from src.trading.connectors.dhan import sdk as dh
from src.trading.connectors.dhan.classification import DHAN_TOOL_CLASS
from src.trading.connectors.futu import sdk as ft
from src.trading.connectors.futu.classification import FUTU_TOOL_CLASS
from src.trading.connectors.longbridge import sdk as lb
from src.trading.connectors.longbridge.classification import LONGBRIDGE_TOOL_CLASS
from src.trading.connectors.okx import sdk as ox
from src.trading.connectors.okx.classification import OKX_TOOL_CLASS
from src.trading.connectors.shioaji import sdk as sj
from src.trading.connectors.shioaji.classification import SHIOAJI_TOOL_CLASS
from src.trading.connectors.shoonya import sdk as sh
from src.trading.connectors.shoonya.classification import SHOONYA_TOOL_CLASS
from src.trading.connectors.tiger import sdk as tg
from src.trading.connectors.tiger.classification import TIGER_TOOL_CLASS

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------- #
# Profile registration
# --------------------------------------------------------------------------- #


def test_sdk_profiles_registered() -> None:
    """All broker connectors register paper and read-only live profiles."""
    ids = {p.id for p in profiles.list_profiles()}
    assert {
        "tiger-paper-sdk", "tiger-live-sdk-readonly",
        "longbridge-paper-sdk", "longbridge-live-sdk-readonly",
        "alpaca-paper-sdk", "alpaca-live-sdk-readonly",
        "okx-paper-sdk", "okx-live-sdk-readonly",
        "binance-paper-sdk", "binance-live-sdk-readonly",
        "futu-paper-sdk", "futu-live-sdk-readonly",
        "dhan-paper-sdk", "dhan-live-sdk-readonly",
        "shoonya-paper-sdk", "shoonya-live-sdk-readonly",
        "shioaji-paper-sdk", "shioaji-live-sdk-readonly",
    } <= ids


def test_no_discriminator_brokers_expose_no_live_trade_profile() -> None:
    """Brokers without a runtime paper/live discriminator (Longbridge, Dhan,
    Shoonya, Shioaji) must NOT register any live order-placing profile — the
    Longbridge precedent. A ``*-live-trade`` profile here would be a red-line
    regression. Shioaji has no account-format discriminator either (one
    account, a ``simulation`` boolean), so it belongs in this group too."""
    ids = {p.id for p in profiles.list_profiles()}
    for broker in ("longbridge", "dhan", "shoonya", "shioaji"):
        assert f"{broker}-live-trade" not in ids
        # No live profile for these brokers may advertise an order capability.
        for p in profiles.list_profiles():
            if p.connector == broker and p.environment == "live":
                assert not any(".place" in cap or "requires_mandate" in cap for cap in p.capabilities)


@pytest.mark.parametrize(
    "profile_id, connector, environment",
    [
        ("tiger-paper-sdk", "tiger", "paper"),
        ("tiger-live-sdk-readonly", "tiger", "live"),
        ("longbridge-paper-sdk", "longbridge", "paper"),
        ("longbridge-live-sdk-readonly", "longbridge", "live"),
        ("alpaca-paper-sdk", "alpaca", "paper"),
        ("alpaca-live-sdk-readonly", "alpaca", "live"),
        ("okx-paper-sdk", "okx", "paper"),
        ("okx-live-sdk-readonly", "okx", "live"),
        ("binance-paper-sdk", "binance", "paper"),
        ("binance-live-sdk-readonly", "binance", "live"),
        ("futu-paper-sdk", "futu", "paper"),
        ("futu-live-sdk-readonly", "futu", "live"),
        ("dhan-paper-sdk", "dhan", "paper"),
        ("dhan-live-sdk-readonly", "dhan", "live"),
        ("shoonya-paper-sdk", "shoonya", "paper"),
        ("shoonya-live-sdk-readonly", "shoonya", "live"),
        ("shioaji-paper-sdk", "shioaji", "paper"),
        ("shioaji-live-sdk-readonly", "shioaji", "live"),
    ],
)
def test_sdk_profiles_are_readonly_broker_sdk(profile_id, connector, environment) -> None:
    """Layer A profiles are broker_sdk transport and strictly read-only."""
    profile = profiles.profile_by_id(profile_id)
    assert profile.connector == connector
    assert profile.environment == environment
    assert profile.transport == "broker_sdk"
    assert profile.readonly is True
    # No order-placing / mandate-gated capability is advertised in Layer A.
    assert not any(".place" in cap or "requires_mandate" in cap for cap in profile.capabilities)


# --------------------------------------------------------------------------- #
# Tiger paper/live identity guard (17-digit account rule)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "account, is_paper",
    [
        ("20191106192858300", True),   # 17-digit paper
        ("51230321", False),            # prime/standard
        ("U12300123", False),           # global
        ("", False),
        ("2019110619285830", False),    # 16 digits
        ("201911061928583000", False),  # 18 digits
    ],
)
def test_tiger_is_paper_account(account, is_paper) -> None:
    assert tg.is_paper_account(account) is is_paper


def test_tiger_paper_profile_rejects_live_account() -> None:
    """A paper profile pointed at a non-17-digit account fails closed."""
    cfg = tg.TigerConfig(tiger_id="x", private_key_path="x", account="U12300123", profile="paper")
    with pytest.raises(tg.TigerProfileMismatchError):
        tg._assert_profile(cfg)


def test_tiger_live_profile_rejects_paper_account() -> None:
    """A live profile pointed at a 17-digit paper account fails closed."""
    cfg = tg.TigerConfig(tiger_id="x", private_key_path="x", account="20191106192858300", profile="live-readonly")
    with pytest.raises(tg.TigerProfileMismatchError):
        tg._assert_profile(cfg)


def test_tiger_paper_profile_accepts_paper_account() -> None:
    cfg = tg.TigerConfig(tiger_id="x", private_key_path="x", account="20191106192858300", profile="paper")
    tg._assert_profile(cfg)  # must not raise


# --------------------------------------------------------------------------- #
# Config resolution
# --------------------------------------------------------------------------- #


def test_tiger_build_config_merges_profile_then_overrides(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(tg, "get_runtime_root", lambda: tmp_path)
    cfg = tg.build_config({"profile": "paper"}, {"account": "20191106192858300"})
    assert cfg.profile == "paper"
    assert cfg.account == "20191106192858300"


def test_tiger_invalid_profile_rejected() -> None:
    with pytest.raises(tg.TigerConfigError):
        tg.TigerConfig.from_mapping({"profile": "live-trade-now"})


def test_longbridge_build_config_and_region(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(lb, "get_runtime_root", lambda: tmp_path)
    cfg = lb.build_config({"profile": "live-readonly", "region": "cn"}, None)
    assert cfg.profile == "live-readonly"
    assert cfg.region == "cn"


def test_longbridge_invalid_region_rejected() -> None:
    with pytest.raises(lb.LongbridgeConfigError):
        lb.LongbridgeConfig.from_mapping({"region": "moon"})


def test_longbridge_public_config_redacts_secrets() -> None:
    """Secret material must never appear in a status payload."""
    cfg = lb.LongbridgeConfig(app_key="abcd1234", app_secret="supersecret", access_token="tok-123")
    pub = lb._public_config(cfg)
    assert pub["app_secret"] == "***redacted***"
    assert pub["access_token"] == "***redacted***"
    assert "supersecret" not in str(pub)
    assert "tok-123" not in str(pub)
    assert pub["app_key"].endswith("***")


# --------------------------------------------------------------------------- #
# Read/write classification (live gate input)
# --------------------------------------------------------------------------- #


def test_tiger_order_ops_classified_write() -> None:
    for name in ("place_order", "cancel_order", "modify_order"):
        assert TIGER_TOOL_CLASS[name] is ToolClass.WRITE
    for name in ("get_assets", "get_positions", "get_bars"):
        assert TIGER_TOOL_CLASS[name] is ToolClass.READ


def test_longbridge_order_ops_classified_write() -> None:
    for name in ("submit_order", "cancel_order", "replace_order"):
        assert LONGBRIDGE_TOOL_CLASS[name] is ToolClass.WRITE
    for name in ("account_balance", "stock_positions", "candlesticks"):
        assert LONGBRIDGE_TOOL_CLASS[name] is ToolClass.READ


# --------------------------------------------------------------------------- #
# Service dispatch degrades cleanly when nothing is configured
# --------------------------------------------------------------------------- #


def test_service_check_connection_unconfigured_tiger(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(tg, "get_runtime_root", lambda: tmp_path)
    result = service.check_connection("tiger-paper-sdk")
    assert result["status"] == "error"
    assert "not configured" in result["error"]
    assert result["connector"] == "tiger"
    assert result["transport"] == "broker_sdk"


def test_service_check_connection_unconfigured_longbridge(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(lb, "get_runtime_root", lambda: tmp_path)
    result = service.check_connection("longbridge-paper-sdk")
    assert result["status"] == "error"
    assert "not configured" in result["error"]
    assert result["connector"] == "longbridge"
    assert result["transport"] == "broker_sdk"


# --------------------------------------------------------------------------- #
# Alpaca
# --------------------------------------------------------------------------- #


def test_alpaca_paper_live_host_and_flag() -> None:
    assert al.AlpacaConfig(profile="paper").is_paper is True
    assert al.AlpacaConfig(profile="paper").host == al.PAPER_HOST
    assert al.AlpacaConfig(profile="live-readonly").is_paper is False
    assert al.AlpacaConfig(profile="live-readonly").host == al.LIVE_HOST


def test_alpaca_invalid_feed_rejected() -> None:
    with pytest.raises(al.AlpacaConfigError):
        al.AlpacaConfig.from_mapping({"feed": "nasdaq"})


def test_alpaca_redacts_secrets() -> None:
    cfg = al.AlpacaConfig(api_key="AKFOURCHARS", secret_key="topsecret")
    pub = al._public_config(cfg)
    assert pub["secret_key"] == "***redacted***"
    assert "topsecret" not in str(pub)
    assert pub["api_key"].endswith("***")


def test_alpaca_classification() -> None:
    assert ALPACA_TOOL_CLASS["submit_order"] is ToolClass.WRITE
    assert ALPACA_TOOL_CLASS["cancel_order_by_id"] is ToolClass.WRITE
    assert ALPACA_TOOL_CLASS["get_account"] is ToolClass.READ


def test_alpaca_service_unconfigured(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(al, "get_runtime_root", lambda: tmp_path)
    result = service.check_connection("alpaca-paper-sdk")
    assert result["status"] == "error"
    assert result["connector"] == "alpaca"
    assert result["transport"] == "broker_sdk"


# --------------------------------------------------------------------------- #
# OKX
# --------------------------------------------------------------------------- #


def test_okx_flag_mapping() -> None:
    assert ox.OKXConfig(profile="paper").flag == "1"
    assert ox.OKXConfig(profile="live-readonly").flag == "0"
    assert ox.OKXConfig(profile="live").flag == "0"


def test_okx_redacts_secrets() -> None:
    cfg = ox.OKXConfig(api_key="KEYFOURXX", api_secret="sec", passphrase="pass")
    pub = ox._public_config(cfg)
    assert pub["api_secret"] == "***redacted***"
    assert pub["passphrase"] == "***redacted***"
    assert "sec" not in str(pub) or pub["api_secret"] == "***redacted***"


def test_okx_classification() -> None:
    assert OKX_TOOL_CLASS["place_order"] is ToolClass.WRITE
    assert OKX_TOOL_CLASS["cancel_order"] is ToolClass.WRITE
    assert OKX_TOOL_CLASS["get_account_balance"] is ToolClass.READ


def test_okx_service_unconfigured(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(ox, "get_runtime_root", lambda: tmp_path)
    result = service.check_connection("okx-paper-sdk")
    assert result["status"] == "error"
    assert result["connector"] == "okx"


# --------------------------------------------------------------------------- #
# Binance
# --------------------------------------------------------------------------- #


def test_binance_testnet_host_mapping() -> None:
    assert bn.BinanceConfig(profile="paper").is_testnet is True
    assert "testnet" in bn.BinanceConfig(profile="paper").host
    assert bn.BinanceConfig(profile="live-readonly").is_testnet is False
    assert bn.BinanceConfig(profile="live-readonly").host == "https://api.binance.com"


def test_binance_classification() -> None:
    assert BINANCE_TOOL_CLASS["create_order"] is ToolClass.WRITE
    assert BINANCE_TOOL_CLASS["cancel_order"] is ToolClass.WRITE
    assert BINANCE_TOOL_CLASS["fetch_balance"] is ToolClass.READ


def test_binance_service_unconfigured(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(bn, "get_runtime_root", lambda: tmp_path)
    result = service.check_connection("binance-paper-sdk")
    assert result["status"] == "error"
    assert result["connector"] == "binance"


# --------------------------------------------------------------------------- #
# Futu (local OpenD gateway)
# --------------------------------------------------------------------------- #


def test_futu_trd_env_mapping() -> None:
    assert ft.FutuConfig(profile="paper").trd_env_name == "SIMULATE"
    assert ft.FutuConfig(profile="live-readonly").trd_env_name == "REAL"


def test_futu_classification() -> None:
    assert FUTU_TOOL_CLASS["place_order"] is ToolClass.WRITE
    assert FUTU_TOOL_CLASS["modify_order"] is ToolClass.WRITE
    assert FUTU_TOOL_CLASS["unlock_trade"] is ToolClass.WRITE
    assert FUTU_TOOL_CLASS["position_list_query"] is ToolClass.READ


def test_futu_service_unconfigured_gateway_down(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(ft, "get_runtime_root", lambda: tmp_path)
    result = service.check_connection("futu-paper-sdk")
    # OpenD gateway is not running in CI → clean error, not a crash.
    assert result["status"] == "error"
    assert result["connector"] == "futu"
    assert result["transport"] == "broker_sdk"


def test_binance_redacts_secrets() -> None:
    cfg = bn.BinanceConfig(api_key="ABCD1234", api_secret="topsecret")
    pub = bn._public_config(cfg)
    assert pub["api_secret"] == "***redacted***"
    assert "topsecret" not in str(pub)
    assert pub["api_key"].endswith("***")


def test_binance_assert_host_consistent_profiles_pass() -> None:
    """Host property is the guard: paper→testnet host, live→api.binance.com.

    The host is derived from the profile (paper→``testnet_host``,
    live→``api.binance.com``), so a paper profile structurally cannot resolve to
    the live host. ``_assert_host`` is defense-in-depth over that derivation and
    must accept both consistent profiles without raising.
    """
    bn._assert_host(bn.BinanceConfig(profile="paper"))
    bn._assert_host(bn.BinanceConfig(profile="live-readonly"))
    assert "testnet" in bn.BinanceConfig(profile="paper").host
    assert bn.BinanceConfig(profile="live-readonly").host == "https://api.binance.com"


def test_okx_invalid_profile_rejected() -> None:
    with pytest.raises(ox.OKXConfigError):
        ox.OKXConfig.from_mapping({"profile": "go-live-now"})


# --------------------------------------------------------------------------- #
# Live gate: order ops are WRITE-pinned through the real classifier + registry
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "broker, order_op",
    [
        ("tiger", "place_order"),
        ("longbridge", "submit_order"),
        ("alpaca", "submit_order"),
        ("okx", "place_order"),
        ("binance", "create_order"),
        ("futu", "place_order"),
        ("dhan", "place_order"),
        ("shoonya", "place_order"),
    ],
)
def test_order_ops_write_pinned_via_registry(broker, order_op) -> None:
    """Every broker's order op resolves WRITE through the shared classifier."""
    from src.live import registry
    from src.live.classification import classify_tool

    curated = registry._BROKER_CURATED_MAPS[broker]
    assert classify_tool(order_op, None, curated) is ToolClass.WRITE


def test_unknown_op_does_not_classify_read() -> None:
    """An unmapped op resolves to UNKNOWN (never READ); the registry then treats
    UNKNOWN as WRITE (fail-closed) when wrapping the live channel."""
    from src.live import registry
    from src.live.classification import classify_tool

    curated = registry._BROKER_CURATED_MAPS["okx"]
    verdict = classify_tool("some_unmapped_future_tool", None, curated)
    assert verdict is not ToolClass.READ
    assert verdict in (ToolClass.WRITE, ToolClass.UNKNOWN)


# --------------------------------------------------------------------------- #
# Period mapping (generic token → per-SDK token)
# --------------------------------------------------------------------------- #


def test_period_maps_distinguish_minute_from_month() -> None:
    """The 1m (minute) vs 1M (month) tokens must not collide in any map."""
    assert tg._PERIOD_MAP["1m"] == "1min" and tg._PERIOD_MAP["1M"] == "month"
    assert ox._BAR_MAP["1m"] == "1m" and ox._BAR_MAP["1M"] == "1M"
    assert ft._KLTYPE_MAP["1m"] == "K_1M" and ft._KLTYPE_MAP["1M"] == "K_MON"


# --------------------------------------------------------------------------- #
# Read-path mapping with stubbed SDK clients (no broker SDK installed)
# --------------------------------------------------------------------------- #


class _FakeLbTrade:
    def today_orders(self):
        return [
            {"order_id": "1", "symbol": "700.HK", "status": "NewStatus", "quantity": 100},
            {"order_id": "2", "symbol": "700.HK", "status": "FilledStatus", "quantity": 100},
            {"order_id": "3", "symbol": "AAPL.US", "status": "CanceledStatus", "quantity": 5},
        ]


def test_longbridge_open_orders_filters_terminal(monkeypatch) -> None:
    monkeypatch.setattr(lb, "_trade_context", lambda cfg: _FakeLbTrade())
    out = lb.get_open_orders(lb.LongbridgeConfig(app_key="k", app_secret="s", access_token="t"))
    ids = [o["order_id"] for o in out["open_orders"]]
    assert ids == ["1"]  # filled + cancelled dropped


def test_longbridge_status_normalization_variants() -> None:
    """Terminal-status filtering must work across SDK string forms."""
    for terminal in ("Filled", "FilledStatus", "OrderStatus.Filled", "CANCELED", "Rejected"):
        assert not lb._is_open_order({"status": terminal})
    for live in ("NewStatus", "PartialFilledStatus", "PartialFilled", "WaitToNew"):
        assert lb._is_open_order({"status": live})


class _FakeTigerQuote:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def get_bars(self, symbols, period=None, limit=None):
        self.calls.append({"period": period, "limit": limit})
        return []


def test_tiger_history_month_does_not_collapse_to_minute(monkeypatch) -> None:
    """Regression: ``1M`` (month) must map to ``month``, not ``1min``."""
    fake = _FakeTigerQuote()
    monkeypatch.setattr(tg, "_quote_client", lambda cfg: fake)
    monkeypatch.setattr(tg, "_assert_profile", lambda cfg: None)
    cfg = tg.TigerConfig(tiger_id="x", private_key_path="x", account="20191106192858300", profile="paper")
    tg.get_historical_bars("AAPL", config=cfg, period="1M", limit=12)
    assert fake.calls[-1]["period"] == "month"
    assert fake.calls[-1]["limit"] == 12


def test_trading_history_tool_exposes_period_and_limit() -> None:
    from src.tools.trading_connector_tool import TradingHistoryTool

    props = TradingHistoryTool.parameters["properties"]
    assert "period" in props and "limit" in props


class _FakeOkxMarket:
    def get_candlesticks(self, instId=None, bar=None, limit=None):
        return {"code": "0", "data": [["1700000000000", "100", "110", "90", "105", "12", "1200", "1200", "1"]]}


def test_okx_history_maps_candles_and_period(monkeypatch) -> None:
    monkeypatch.setattr(ox, "_market_client", lambda cfg: _FakeOkxMarket())
    out = ox.get_historical_bars("BTC-USDT", config=ox.OKXConfig(api_key="k", api_secret="s", passphrase="p"), period="1h")
    assert out["period"] == "1h" and out["bar"] == "1H"
    assert len(out["bars"]) == 1
    bar = out["bars"][0]
    assert bar["open"] == "100" and bar["close"] == "105" and bar["confirm"] == "1"


# --------------------------------------------------------------------------- #
# Dhan + Shoonya: structural paper-only cap (no runtime discriminator)
#
# Like Longbridge, these brokers expose no sandbox / no runtime paper/live
# discriminator (same token/login reaches the same real account). The order
# path is therefore structurally capped at paper: any non-paper config is
# refused at the first line, so a flipped ``profile`` override can never reach a
# live order. Paper orders are simulated locally (neither broker has a sandbox).
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("mod, Config", [(dh, dh.DhanConfig), (sh, sh.ShoonyaConfig)])
@pytest.mark.parametrize("profile", ["live", "live-readonly"])
def test_in_broker_place_order_refuses_non_paper(mod, Config, profile) -> None:
    """A non-paper config is refused before any SDK call (fail-closed)."""
    result = mod.place_order(Config(profile=profile), symbol="RELIANCE", side="buy", quantity=1)
    assert result["status"] == "error"
    assert "paper-only" in result["error"]


@pytest.mark.parametrize("mod, Config", [(dh, dh.DhanConfig), (sh, sh.ShoonyaConfig)])
def test_in_broker_cancel_order_refuses_non_paper(mod, Config) -> None:
    result = mod.cancel_order(Config(profile="live"), "ORD1")
    assert result["status"] == "error"
    assert "paper-only" in result["error"]


@pytest.mark.parametrize("mod, Config", [(dh, dh.DhanConfig), (sh, sh.ShoonyaConfig)])
def test_in_broker_paper_place_order_simulated_locally(mod, Config) -> None:
    """Paper config simulates locally — no real money, no SDK call."""
    result = mod.place_order(Config(profile="paper"), symbol="RELIANCE", side="buy", quantity=10)
    assert result["status"] == "ok"
    assert result["is_paper"] is True
    assert result["order_status"] == "simulated_fill"
    assert result["paper_guard"] == "simulated_locally"


@pytest.mark.parametrize("mod, Config", [(dh, dh.DhanConfig), (sh, sh.ShoonyaConfig)])
def test_in_broker_paper_cancel_order_simulated(mod, Config) -> None:
    result = mod.cancel_order(Config(profile="paper"), "ORD1")
    assert result["status"] == "ok"
    assert result["cancelled"] is True
    assert result["is_paper"] is True


def test_in_broker_order_ops_classified_write() -> None:
    for name in ("place_order", "modify_order", "cancel_order"):
        assert DHAN_TOOL_CLASS[name] is ToolClass.WRITE
        assert SHOONYA_TOOL_CLASS[name] is ToolClass.WRITE
    for name in ("get_positions", "get_holdings"):
        assert DHAN_TOOL_CLASS[name] is ToolClass.READ
        assert SHOONYA_TOOL_CLASS[name] is ToolClass.READ


def test_dhan_redacts_access_token() -> None:
    cfg = dh.DhanConfig(client_id="C1", access_token="tok-abcdefgh-secret")
    pub = dh._public_config(cfg)
    assert "secret" not in str(pub)
    assert pub["access_token"].endswith("***")


def test_shoonya_redacts_secrets() -> None:
    cfg = sh.ShoonyaConfig(
        user_id="USER1", password="pw", vendor_code="V", api_secret="sec", totp_secret="totp"
    )
    pub = sh._public_config(cfg)
    for secret in ("password", "api_secret", "totp_secret"):
        assert pub[secret] == "***redacted***"
    assert "sec" not in str(pub) or pub["api_secret"] == "***redacted***"
    assert pub["user_id"].endswith("***")


def test_dhan_invalid_profile_rejected() -> None:
    with pytest.raises(dh.DhanConfigError):
        dh.DhanConfig.from_mapping({"profile": "go-live"})


def test_shoonya_invalid_profile_rejected() -> None:
    with pytest.raises(sh.ShoonyaConfigError):
        sh.ShoonyaConfig.from_mapping({"profile": "go-live"})


def test_dhan_service_unconfigured(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(dh, "get_runtime_root", lambda: tmp_path)
    result = service.check_connection("dhan-paper-sdk")
    assert result["status"] == "error"
    assert result["connector"] == "dhan"
    assert result["transport"] == "broker_sdk"


def test_shoonya_service_unconfigured(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(sh, "get_runtime_root", lambda: tmp_path)
    result = service.check_connection("shoonya-paper-sdk")
    assert result["status"] == "error"
    assert result["connector"] == "shoonya"
    assert result["transport"] == "broker_sdk"


# --------------------------------------------------------------------------- #
# Shioaji
# --------------------------------------------------------------------------- #


def test_shioaji_clear_stale_locks_removes_old_not_recent(tmp_path, monkeypatch) -> None:
    import os
    import time

    monkeypatch.setenv("SJ_HOME_PATH", str(tmp_path))
    old_lock = tmp_path / "contracts-1.5.4-STK-TW.parquet.lock"
    recent_lock = tmp_path / "contracts-1.5.4-FUT-TW.parquet.lock"
    old_lock.touch()
    recent_lock.touch()
    old_time = time.time() - 999
    os.utime(old_lock, (old_time, old_time))

    sj._clear_stale_shioaji_locks(max_age_seconds=120.0)

    assert not old_lock.exists()
    assert recent_lock.exists()


def test_shioaji_simulation_flag_mapping() -> None:
    assert sj.ShioajiConfig(profile="paper").simulation is True
    assert sj.ShioajiConfig(profile="live-readonly").simulation is False


def test_shioaji_build_config_merges_profile_then_overrides(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(sj, "get_runtime_root", lambda: tmp_path)
    cfg = sj.build_config({"profile": "live-readonly"}, {"api_key": "real-key"})
    assert cfg.profile == "live-readonly"
    assert cfg.api_key == "real-key"


def test_shioaji_invalid_profile_rejected() -> None:
    with pytest.raises(sj.ShioajiConfigError):
        sj.ShioajiConfig.from_mapping({"profile": "go-live"})


def test_shioaji_redacts_secrets() -> None:
    cfg = sj.ShioajiConfig(api_key="abcd1234", secret_key="supersecret", profile="paper")
    pub = sj._public_config(cfg)
    assert pub["secret_key"] == "***redacted***"
    assert "supersecret" not in str(pub)
    assert pub["api_key"].endswith("***")


def test_shioaji_order_placement_not_implemented() -> None:
    """This phase is read-only -- no place_order/cancel_order function exists."""
    assert not hasattr(sj, "place_order")
    assert not hasattr(sj, "cancel_order")


def test_shioaji_classification() -> None:
    for name in ("snapshots", "kbars", "list_positions", "account_balance"):
        assert SHIOAJI_TOOL_CLASS[name] is ToolClass.READ


def test_shioaji_service_unconfigured(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(sj, "get_runtime_root", lambda: tmp_path)
    result = service.check_connection("shioaji-paper-sdk")
    assert result["status"] == "error"
    assert "not configured" in result["error"]
    assert result["connector"] == "shioaji"
    assert result["transport"] == "broker_sdk"


class _FakeShioajiContract:
    code = "2330"


class _FakeShioajiContracts:
    Stocks = {"2330": _FakeShioajiContract()}


class _FakeShioajiSnapshot:
    code = "2330"
    close = 602.0
    buy_price = 601.0
    sell_price = 603.0
    open = 600.0
    high = 605.0
    low = 598.0
    total_volume = 12345
    change_price = 2.0
    change_rate = 0.33
    ts = 1234567890


class _FakeShioajiApi:
    Contracts = _FakeShioajiContracts()

    def __init__(self) -> None:
        self.kbars_calls: list[dict] = []

    def snapshots(self, contracts):
        return [_FakeShioajiSnapshot()]

    def list_positions(self):
        return []

    def account_balance(self):
        return None

    def list_trades(self):
        return []

    def logout(self):
        return True


def test_shioaji_get_quote_maps_snapshot_fields(monkeypatch) -> None:
    fake = _FakeShioajiApi()
    monkeypatch.setattr(sj, "_login", lambda cfg: fake)
    cfg = sj.ShioajiConfig(api_key="k", secret_key="s", profile="paper")

    result = sj.get_quote("2330.TW", config=cfg)

    assert result["status"] == "ok"
    assert result["quote"]["symbol"] == "2330"
    assert result["quote"]["last"] == 602.0
    assert result["quote"]["bid"] == 601.0
    assert result["quote"]["ask"] == 603.0


def test_shioaji_get_quote_strips_tw_suffix(monkeypatch) -> None:
    """The bare Shioaji contract lookup must use the stock id, not the .TW-suffixed symbol."""
    fake = _FakeShioajiApi()
    monkeypatch.setattr(sj, "_login", lambda cfg: fake)
    cfg = sj.ShioajiConfig(api_key="k", secret_key="s", profile="paper")

    result = sj.get_quote("2330.TWO", config=cfg)
    assert result["status"] == "ok"


def test_shioaji_get_positions_empty_is_ok(monkeypatch) -> None:
    fake = _FakeShioajiApi()
    monkeypatch.setattr(sj, "_login", lambda cfg: fake)
    cfg = sj.ShioajiConfig(api_key="k", secret_key="s", profile="paper")

    result = sj.get_positions(config=cfg)
    assert result["status"] == "ok"
    assert result["positions"] == []


def test_shioaji_history_resamples_minute_kbars_to_daily(monkeypatch) -> None:
    import pandas as pd

    class _KBars:
        ts = [
            pd.Timestamp("2026-01-02 09:00:00").value,
            pd.Timestamp("2026-01-02 09:01:00").value,
            pd.Timestamp("2026-01-03 09:00:00").value,
        ]
        Open = [600.0, 601.0, 610.0]
        High = [602.0, 603.0, 612.0]
        Low = [599.0, 600.0, 609.0]
        Close = [601.0, 602.0, 611.0]
        Volume = [100, 200, 300]

    fake = _FakeShioajiApi()
    fake.kbars = lambda contract, start, end: _KBars()
    monkeypatch.setattr(sj, "_login", lambda cfg: fake)
    cfg = sj.ShioajiConfig(api_key="k", secret_key="s", profile="paper")

    result = sj.get_historical_bars("2330.TW", config=cfg, period="1d", limit=10)

    assert result["status"] == "ok"
    assert len(result["bars"]) == 2
    assert result["bars"][0]["open"] == 600.0
    assert result["bars"][0]["high"] == 603.0  # max(602, 603) across the day's two bars
    assert result["bars"][0]["volume"] == 300.0  # 100 + 200 summed
