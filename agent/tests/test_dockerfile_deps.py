"""Regression test: the Docker image must actually install what src/channels
imports.

Found live 2026-07-08: the Telegram integration (notifier.py,
chat_commands.py, ChannelRuntime's /deploy interception) was built and
tested this session, TELEGRAM_BOT_TOKEN was set in agent/.env, and
VIBE_TRADING_CHANNELS_AUTO_START=1 -- yet every container boot logged
"telegram channel not available: ModuleNotFoundError: No module named
'telegram'" and ran with "No channels enabled". python-telegram-bot is only
declared in pyproject.toml's optional "telegram"/"channels" extras, not the
base dependency list; the Dockerfile's plain `pip install -e .` never pulled
it in, so the feature was never actually reachable in this deployment
despite passing tests (those exercise config/dispatch logic, not the real
import).
"""

from __future__ import annotations

from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def test_dockerfile_installs_telegram_extra() -> None:
    dockerfile = (_REPO_ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert '-e ".[telegram]"' in dockerfile or "-e '.[telegram]'" in dockerfile
