"""Built-in Shioaji (SinoPac) connector profiles.

Read-only paper (simulation) and live (production, read-only) profiles only
in this phase. Order placement is deferred to a later phase behind the
mandate gate, mirroring how ``connectors/tiger/profiles.py`` layers
paper/live trade profiles on top of a read-only base.
"""

from __future__ import annotations

from src.trading.types import READ_CAPABILITIES, TradingProfile

SHIOAJI_PROFILES: tuple[TradingProfile, ...] = (
    TradingProfile(
        id="shioaji-paper-sdk",
        connector="shioaji",
        label="Shioaji 模擬 · SinoPac Simulation",
        environment="paper",
        transport="broker_sdk",
        capabilities=READ_CAPABILITIES,
        readonly=True,
        config={"profile": "paper"},
        notes=(
            "Reads Taiwan equity quotes/account/positions via the official "
            "shioaji SDK against SinoPac's simulation environment. Market "
            "data is the same real feed as production; only account "
            "balance/positions/P&L are simulated (zero/default)."
        ),
    ),
    TradingProfile(
        id="shioaji-live-sdk-readonly",
        connector="shioaji",
        label="Shioaji 正式 · SinoPac Live Read-Only",
        environment="live",
        transport="broker_sdk",
        capabilities=READ_CAPABILITIES,
        readonly=True,
        config={"profile": "live-readonly"},
        notes=(
            "Reads a real SinoPac account's quotes/balance/positions. Order "
            "placement is not exposed in this profile."
        ),
    ),
)
