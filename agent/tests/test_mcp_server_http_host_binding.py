"""Regression tests for the HTTP-family transports' host binding and choice.

Found 2026-07-02 setting up a docker-compose service for the SSE transport:
``mcp_server.py``'s SSE branch called ``mcp.run(transport="sse", port=...)``
with no ``host``, so FastMCP fell back to its own default of ``127.0.0.1``.
Inside a container that's the *container's* loopback, not the bridge-network
interface Docker's port mapping actually connects to -- confirmed live
(``docker logs`` showed "Uvicorn running on http://127.0.0.1:8900", and a
curl from the host got "Connection reset by peer": nothing was listening on
the address the port forward reaches). HTTP-family transports only exist for
network clients in the first place (stdio already covers same-machine), so a
loopback-only default silently defeated the whole point in the one deployment
shape (Docker) this project actually runs it in.

Found again 2026-07-05: a real Antigravity CLI (`agy`) connection to the SSE
endpoint failed with "Method Not Allowed" on `initialize`. SSE's legacy
two-endpoint design only accepts GET on `/sse` (to open the event stream) and
POST on a separate session-scoped `/messages/` path; `agy` instead POSTs
`initialize` straight to the URL it was given, which is how a client speaking
the modern single-endpoint Streamable HTTP transport behaves. Confirmed via
`docker logs` (FastMCP logged the SSE 404s) and a raw curl POST against
`/mcp`, which returned 200 with a valid `initialize` result. Switched the
`mcp-sse` docker-compose service's default from `sse` to `http` (Streamable
HTTP) accordingly; `sse` is kept as a selectable transport for any client that
still needs the legacy shape, and both must bind `0.0.0.0` for the same
Docker-port-mapping reason above.
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


class TestHttpFamilyHostBinding:
    @pytest.mark.parametrize("transport", ["http", "sse"])
    def test_default_host_is_all_interfaces_not_loopback(
        self, mcp_server_module, monkeypatch: pytest.MonkeyPatch, transport: str,
    ) -> None:
        monkeypatch.setattr(sys, "argv", ["vibe-trading-mcp", "--transport", transport])
        mcp_server_module.main()
        _, kwargs = mcp_server_module.mcp.run.call_args
        assert kwargs["host"] == "0.0.0.0"
        assert kwargs["transport"] == transport
        assert kwargs["port"] == 8900

    @pytest.mark.parametrize("transport", ["http", "sse"])
    def test_host_is_overridable(
        self, mcp_server_module, monkeypatch: pytest.MonkeyPatch, transport: str,
    ) -> None:
        monkeypatch.setattr(
            sys, "argv",
            ["vibe-trading-mcp", "--transport", transport, "--host", "127.0.0.1", "--port", "9000"],
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
