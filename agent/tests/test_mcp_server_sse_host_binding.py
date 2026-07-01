"""Regression test for the SSE transport's host binding.

Found 2026-07-02 setting up a docker-compose service for the SSE transport:
``mcp_server.py``'s SSE branch called ``mcp.run(transport="sse", port=...)``
with no ``host``, so FastMCP fell back to its own default of ``127.0.0.1``.
Inside a container that's the *container's* loopback, not the bridge-network
interface Docker's port mapping actually connects to -- confirmed live
(``docker logs`` showed "Uvicorn running on http://127.0.0.1:8900", and a
curl from the host got "Connection reset by peer": nothing was listening on
the address the port forward reaches). SSE only exists for network clients in
the first place (stdio already covers same-machine), so a loopback-only
default silently defeated the whole point of the transport in the one
deployment shape (Docker) this project actually runs it in.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def mcp_server_module(monkeypatch: pytest.MonkeyPatch):
    import mcp_server

    monkeypatch.setattr(mcp_server, "_get_registry", lambda: MagicMock())
    monkeypatch.setattr(mcp_server.mcp, "run", MagicMock())
    return mcp_server


class TestSseHostBinding:
    def test_default_host_is_all_interfaces_not_loopback(
        self, mcp_server_module, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(sys, "argv", ["vibe-trading-mcp", "--transport", "sse"])
        mcp_server_module.main()
        _, kwargs = mcp_server_module.mcp.run.call_args
        assert kwargs["host"] == "0.0.0.0"
        assert kwargs["transport"] == "sse"
        assert kwargs["port"] == 8900

    def test_host_is_overridable(
        self, mcp_server_module, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(
            sys, "argv",
            ["vibe-trading-mcp", "--transport", "sse", "--host", "127.0.0.1", "--port", "9000"],
        )
        mcp_server_module.main()
        _, kwargs = mcp_server_module.mcp.run.call_args
        assert kwargs["host"] == "127.0.0.1"
        assert kwargs["port"] == 9000

    def test_stdio_transport_does_not_pass_host_or_port(
        self, mcp_server_module, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(sys, "argv", ["vibe-trading-mcp"])
        mcp_server_module.main()
        args, kwargs = mcp_server_module.mcp.run.call_args
        assert args == ()
        assert kwargs == {}
