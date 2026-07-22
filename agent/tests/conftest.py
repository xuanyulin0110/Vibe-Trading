"""Shared fixtures and sys.path setup for all tests."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Ensure agent/ is on sys.path so imports like `backtest.*` and `src.*` work.
AGENT_DIR = Path(__file__).resolve().parent.parent
if str(AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(AGENT_DIR))


@pytest.fixture(autouse=True)
def _reset_env_config():
    """Clear the cached EnvConfig before each test so monkeypatch.setenv works."""
    from src.config.accessor import reset_env_config
    reset_env_config()
    yield
    reset_env_config()


@pytest.fixture(autouse=True)
def _keyless_auth_env(monkeypatch: pytest.MonkeyPatch):
    """Neutralize any real API auth key from the developer's agent/.env.

    Upstream's API tests assume a keyless dev environment (loopback bypasses
    auth only when no key is configured); a real ``API_AUTH_KEY`` in the
    developer's ``.env`` would 401 every one of them. Set to "" rather than
    delenv: the api_server startup preflight re-loads ``.env`` with
    ``override=False``, so an *existing* empty var blocks restoration while a
    deleted one gets refilled from the file. Tests that exercise auth still
    work — they ``monkeypatch.setenv`` their own key, which wins over this.
    """
    monkeypatch.setenv("API_AUTH_KEY", "")
    monkeypatch.setenv("VIBE_TRADING_API_KEY", "")
