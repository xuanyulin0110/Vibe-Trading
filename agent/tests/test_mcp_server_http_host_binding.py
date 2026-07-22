"""Regression tests for the HTTP-family transports' host binding and choice.

Found 2026-07-02 setting up a docker-compose service for the SSE transport:
``mcp_server.py``'s SSE branch bound FastMCP's default of ``127.0.0.1``.
Inside a container that's the *container's* loopback, not the bridge-network
interface Docker's port mapping actually connects to -- confirmed live
(``docker logs`` showed "Uvicorn running on http://127.0.0.1:8900", and a
curl from the host got "Connection reset by peer": nothing was listening on
the address the port forward reaches).

Found again 2026-07-05: a real Antigravity CLI (`agy`) connection to the SSE
endpoint failed with "Method Not Allowed" on `initialize` -- modern clients
POST `initialize` straight to the URL (single-endpoint Streamable HTTP), so
the docker-compose service defaults to `--transport http` served at `/mcp`.

Found again 2026-07-08: agy and Claude Code both failed to reconnect over a
Tailscale IP with "HTTP 421 Misdirected Request" -- the Host/Origin
DNS-rebinding guard only allowed loopback, so LAN/Tailscale/WireGuard
clients (`docs/OPERATIONS.md` section 5) never had a chance.

After merging upstream's hardened network stack (2026-07-23: uvicorn +
``_build_network_app`` guard, GHSA-p3c9), the *code* defaults are
deliberately upstream's secure loopback-only ones; this deployment's posture
now lives in ``docker-compose.yml`` (``--host 0.0.0.0`` +
``VIBE_TRADING_MCP_ALLOWED_HOSTS=*``). These tests pin both halves so a
future merge can't silently regress either the CLI plumbing or the compose
flags that keep the three incidents above fixed.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

_COMPOSE = Path(__file__).resolve().parents[2] / "docker-compose.yml"


@pytest.fixture
def mcp_server_module(monkeypatch: pytest.MonkeyPatch):
    import mcp_server

    monkeypatch.setattr(mcp_server, "_get_registry", lambda: MagicMock())
    monkeypatch.setattr(mcp_server.mcp, "run", MagicMock())
    return mcp_server


class TestNetworkTransportPlumbing:
    @pytest.mark.parametrize(
        "cli_transport, fastmcp_transport",
        [("http", "streamable-http"), ("sse", "sse")],
    )
    def test_cli_host_and_port_reach_uvicorn(
        self,
        mcp_server_module,
        monkeypatch: pytest.MonkeyPatch,
        cli_transport: str,
        fastmcp_transport: str,
    ) -> None:
        """--host/--port must reach uvicorn.run, and 'http' must map to
        FastMCP's 'streamable-http' app (the /mcp single endpoint)."""
        seen: dict[str, object] = {}

        def fake_build(transport: str, allowed_hosts: list[str]):
            seen["transport"] = transport
            seen["allowed_hosts"] = allowed_hosts
            return MagicMock()

        fake_uvicorn = MagicMock()
        monkeypatch.setattr(mcp_server_module, "_build_network_app", fake_build)
        monkeypatch.setitem(sys.modules, "uvicorn", fake_uvicorn)
        monkeypatch.setattr(
            sys, "argv",
            ["vibe-trading-mcp", "--transport", cli_transport,
             "--host", "0.0.0.0", "--port", "9000"],
        )
        mcp_server_module.main()
        _, kwargs = fake_uvicorn.run.call_args
        assert kwargs["host"] == "0.0.0.0"
        assert kwargs["port"] == 9000
        assert seen["transport"] == fastmcp_transport
        mcp_server_module.mcp.run.assert_not_called()

    def test_stdio_transport_does_not_pass_host_or_port(
        self, mcp_server_module, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(sys, "argv", ["vibe-trading-mcp"])
        mcp_server_module.main()
        args, kwargs = mcp_server_module.mcp.run.call_args
        assert args == ()
        assert kwargs == {}


class TestAllowedHostsParsing:
    def test_wildcard_passes_through(self) -> None:
        from mcp_server import _parse_allowed_hosts

        assert "*" in _parse_allowed_hosts("*")

    def test_comma_separated_hosts(self) -> None:
        from mcp_server import _parse_allowed_hosts

        parsed = _parse_allowed_hosts("10.0.0.195, 100.78.149.102")
        assert "10.0.0.195" in parsed
        assert "100.78.149.102" in parsed

    def test_unset_defaults_to_loopback_only(self) -> None:
        """Upstream's secure default: DNS-rebinding pages must be rejected
        when the operator hasn't opted into a wider allowlist."""
        from mcp_server import _parse_allowed_hosts

        parsed = _parse_allowed_hosts(None)
        assert "127.0.0.1" in parsed
        assert "*" not in parsed


class TestComposeDeploymentPosture:
    """The compose file, not the code, now carries this deployment's
    network posture -- pin it like frontend's viteProxy config test."""

    def test_mcp_service_binds_all_interfaces(self) -> None:
        # Container-loopback binding is unreachable through Docker's port
        # mapping (incident 2026-07-02).
        assert "--host 0.0.0.0" in _COMPOSE.read_text()

    def test_mcp_service_uses_streamable_http(self) -> None:
        # Modern clients POST initialize to a single endpoint
        # (incident 2026-07-05).
        assert "--transport http" in _COMPOSE.read_text()

    def test_mcp_service_widens_host_guard(self) -> None:
        # Loopback-only guard 421s every LAN/Tailscale/WireGuard client
        # (incident 2026-07-08).
        assert "VIBE_TRADING_MCP_ALLOWED_HOSTS=*" in _COMPOSE.read_text()
