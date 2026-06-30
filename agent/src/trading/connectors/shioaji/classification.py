"""Curated read/write classification for Shioaji SDK operations.

The trading layer classifies each connector operation as READ or WRITE so the
live gate can keep writes behind the mandate. Shioaji is a direct-SDK
connector (not MCP), so the keys here are the connector's own operation
names. This phase exposes read-only operations only -- order placement
(``place_order``/``cancel_order``) is not implemented yet, so there is
nothing to list as WRITE; anything not listed and not a known read resolves
to WRITE (fail-closed) when the live gate consults this map, which is the
correct default once order placement is added in a later phase.
"""

from __future__ import annotations

from src.live.classification import ToolClass

#: Shioaji SDK operation read/write catalog.
SHIOAJI_TOOL_CLASS: dict[str, ToolClass] = {
    # READ
    "list_accounts": ToolClass.READ,
    "account_balance": ToolClass.READ,
    "list_positions": ToolClass.READ,
    "list_trades": ToolClass.READ,
    "snapshots": ToolClass.READ,
    "kbars": ToolClass.READ,
}
