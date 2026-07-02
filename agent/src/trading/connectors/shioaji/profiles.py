"""Built-in Shioaji (SinoPac) connector profiles.

Read-only paper/live profiles plus a paper-trade profile (futures order
placement against the real SinoPac simulation environment). No live-trade
profile: Shioaji has no runtime paper/live discriminator (one account, a
``simulation`` boolean), so -- following the Longbridge/Dhan/Shoonya
precedent -- it stays structurally capped at paper until CA activation and
mandate-gate wiring exist for a real live order path.
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
    TradingProfile(
        id="shioaji-paper-trade",
        connector="shioaji",
        label="Shioaji 模擬下單 · SinoPac Simulation Trade",
        environment="paper",
        transport="broker_sdk",
        capabilities=READ_CAPABILITIES + ("orders.place",),
        readonly=False,
        config={"profile": "paper"},
        notes=(
            "Places TAIFEX futures orders (TXF/MXF/TMF) against SinoPac's real "
            "simulation environment via the official shioaji SDK -- not a "
            "locally-faked fill. No CA cert needed (simulation orders skip CA "
            "signing). Quantity in contracts only, no notional path."
        ),
    ),
)
