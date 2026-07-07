"""Channel config loading helpers."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from src.config.loader import load_agent_config

#: Env vars checked (in order) for the Telegram bot token. Keeping the token
#: in ``agent/.env`` alongside every other secret (SJ_API_KEY, finlab token…)
#: beats a second secret location in agent.json.
_TELEGRAM_TOKEN_ENVS = ("TELEGRAM_BOT_TOKEN", "VIBE_TRADING_TELEGRAM_TOKEN")


def load_channels_config(config_path: Path | None = None) -> dict[str, Any]:
    """Load the operator IM channel config from the structured agent config.

    Args:
        config_path: Optional explicit config path.

    Returns:
        A plain dictionary suitable for :class:`src.channels.manager.ChannelManager`.
    """
    config = load_agent_config(config_path)
    data = config.channels.model_dump(mode="json", by_alias=False)
    return _apply_env_overrides(data)


def _apply_env_overrides(data: dict[str, Any]) -> dict[str, Any]:
    """Overlay secrets provided via environment variables.

    An env-provided Telegram token replaces the file token, and implies
    ``enabled: true`` when the file didn't carry a token of its own (an
    explicit file config with its own token keeps its own ``enabled`` flag,
    so an operator who deliberately disabled a configured channel is never
    surprised by an env var flipping it back on).
    """
    token = next(
        (os.environ[name].strip() for name in _TELEGRAM_TOKEN_ENVS if os.environ.get(name, "").strip()),
        "",
    )
    if not token:
        return data
    telegram = dict(data.get("telegram") or {})
    file_had_token = bool(str(telegram.get("token") or "").strip())
    telegram["token"] = token
    if not file_had_token:
        telegram["enabled"] = True
        # Buttons are what make /deploy interactive; the env-only quick-setup
        # path turns them on (a file config with its own token keeps its own
        # flags -- model_dump always materializes defaults, so setdefault
        # can't distinguish "unset" here).
        telegram["inline_keyboards"] = True
    data["telegram"] = telegram
    return data
