"""Regression test: VIBE_TRADING_DATA_DIR must actually redirect runtime state.

Found live 2026-07-08 debugging a real user's Telegram pairing failure:
tests/test_channels_runtime.py's pairing test set this env var expecting it
to sandbox src/channels/pairing/store.py's writes, but src/config/paths.py
never read it -- get_runtime_root() always resolved to the real
``~/.vibe-trading`` regardless. Every test exercising the pairing store was
silently writing real pending-code/approved-sender entries into the test
runner's actual home directory instead of an isolated tmp_path, which also
made cross-test pollution possible (a leftover pending entry from one test
leaking into another test's "no pending requests" assertion).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.config.paths import get_runtime_root


def test_data_dir_env_var_overrides_default_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    override = tmp_path / "isolated-runtime"
    monkeypatch.setenv("VIBE_TRADING_DATA_DIR", str(override))

    assert get_runtime_root() == override


def test_explicit_config_path_wins_over_data_dir_env_var(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VIBE_TRADING_DATA_DIR", str(tmp_path / "ignored"))
    explicit = tmp_path / "explicit" / "agent.json"

    assert get_runtime_root(explicit) == explicit.parent


def test_no_env_var_falls_back_to_home_dir(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VIBE_TRADING_DATA_DIR", raising=False)

    assert get_runtime_root() == Path.home() / ".vibe-trading"
