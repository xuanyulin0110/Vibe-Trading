"""Regression test: /deploy must actually be routed by the Telegram adapter.

Found live 2026-07-08: a real user paired successfully (confirmed approved),
then "/deploy help" got no reply at all -- not an error, nothing. Root
cause: Telegram treats any leading-"/" message as a command (filters.COMMAND
in python-telegram-bot), which excludes it from the adapter's plain-text
handler (registered with ``& ~filters.COMMAND``). The only other handler
that could catch it, ``_forward_command``, is gated by
``TELEGRAM_BUS_SLASH_COMMAND_RE`` -- and "deploy" was never added to that
allowlist, despite the entire /deploy chat integration (chat_commands.py,
ChannelRuntime's interception, all its unit tests) being built and tested
at the MessageBus level, which never touches this adapter-level regex.
Every other layer worked; the message just never got forwarded onto the
bus in the first place.
"""

from __future__ import annotations

import pytest


class TestTelegramBusSlashCommandRegex:
    @pytest.mark.parametrize(
        "content",
        [
            "/deploy",
            "/deploy help",
            "/deploy status",
            "/deploy pos",
            "/deploy start 2330",
            "/deploy@Bob_the_traderbot status",
        ],
    )
    def test_deploy_commands_match_the_bus_forwarding_regex(self, content: str) -> None:
        pytest.importorskip("telegram")
        from src.channels.telegram import TelegramChannel

        assert TelegramChannel.TELEGRAM_BUS_SLASH_COMMAND_RE.match(content) is not None

    def test_deploy_is_registered_in_the_command_menu(self) -> None:
        pytest.importorskip("telegram")
        from src.channels.telegram import TelegramChannel

        names = {cmd.command for cmd in TelegramChannel.BOT_COMMANDS}
        assert "deploy" in names
