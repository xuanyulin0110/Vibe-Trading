"""Regression tests: an unapproved DM sender must be able to self-approve.

Found live 2026-07-08: a real user's Telegram "/pairing approve <code>"
kept silently failing -- every attempt just issued a fresh pairing code
instead of approving the one they already had. Root cause: BOTH
BaseChannel._handle_message and (independently) the Telegram adapter's
_process_forward_command checked ``is_allowed(sender_id)`` BEFORE looking at
message content, so a "/pairing approve <code>" from a not-yet-approved
sender never reached the pairing handler at all -- it hit the same "you're
not approved, here's a fresh code" branch as any other message. Since
self-approval is the only way an unapproved sender becomes approved, this
was an unbreakable deadlock for every channel sharing the base class, not
just Telegram.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from src.channels.base import BaseChannel
from src.channels.bus.events import OutboundMessage
from src.channels.bus.queue import MessageBus


class _FakeChannel(BaseChannel):
    """Minimal concrete BaseChannel for exercising _handle_message directly."""

    def __init__(self, bus: MessageBus) -> None:
        super().__init__(config={"allow_from": []}, bus=bus)
        self.sent: list[OutboundMessage] = []

    async def start(self) -> None:
        self._running = True

    async def stop(self) -> None:
        self._running = False

    async def send(self, msg: OutboundMessage) -> None:
        self.sent.append(msg)


@pytest.fixture(autouse=True)
def _sandbox_pairing_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VIBE_TRADING_DATA_DIR", str(tmp_path / "data"))


def test_unapproved_sender_pairing_command_reaches_the_bus() -> None:
    async def scenario() -> None:
        bus = MessageBus()
        channel = _FakeChannel(bus)

        await channel._handle_message(
            sender_id="stranger",
            chat_id="chat-1",
            content="/pairing approve ABCD-1234",
            is_dm=True,
        )

        msg = await asyncio.wait_for(bus.consume_inbound(), timeout=1)
        assert msg.content == "/pairing approve ABCD-1234"
        # The bug: this used to be swallowed and replaced with a fresh pairing
        # code, never reaching the bus at all.
        assert channel.sent == []

    asyncio.run(scenario())


def test_unapproved_sender_ordinary_message_still_gets_a_pairing_code() -> None:
    """The fix only carves out /pairing -- everything else keeps the gate."""

    async def scenario() -> None:
        bus = MessageBus()
        channel = _FakeChannel(bus)

        await channel._handle_message(
            sender_id="stranger",
            chat_id="chat-1",
            content="hello there",
            is_dm=True,
        )

        assert bus.inbound_size == 0
        assert len(channel.sent) == 1
        assert "pairing code" in channel.sent[0].content.lower()

    asyncio.run(scenario())


def test_approved_sender_ordinary_message_reaches_the_bus() -> None:
    async def scenario() -> None:
        from src.channels.pairing import approve_code, generate_code

        bus = MessageBus()
        channel = _FakeChannel(bus)
        code = generate_code(channel.name, "friend")
        approve_code(code)

        await channel._handle_message(
            sender_id="friend",
            chat_id="chat-1",
            content="hello",
            is_dm=True,
        )

        msg = await asyncio.wait_for(bus.consume_inbound(), timeout=1)
        assert msg.content == "hello"

    asyncio.run(scenario())


class TestTelegramForwardCommandBootstrap:
    """Same deadlock, one layer up: Telegram's own slash-command router had
    an independent is_allowed gate ahead of BaseChannel's, so fixing only
    base.py wasn't enough -- telegram.py needed the same carve-out."""

    def test_process_forward_command_lets_pairing_through_when_unapproved(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        pytest.importorskip("telegram")
        from types import SimpleNamespace

        from src.channels.telegram import TelegramChannel, TelegramConfig

        async def scenario() -> None:
            bus = MessageBus()
            channel = TelegramChannel(TelegramConfig(token="test-token", allow_from=[]), bus)

            handled: list[tuple[str, str, bool]] = []

            async def fake_handle_message(*, sender_id, chat_id, content, is_dm=False, **_kw):
                handled.append((sender_id, content, is_dm))

            monkeypatch.setattr(channel, "_handle_message", fake_handle_message)
            monkeypatch.setattr(channel, "_remember_thread_context", lambda _msg: None)

            message = SimpleNamespace(
                text="/pairing approve ABCD-1234",
                message_id=1,
                chat_id="chat-1",
                chat=SimpleNamespace(type="private", is_forum=False),
                message_thread_id=None,
                reply_to_message=None,
            )
            user = SimpleNamespace(id=999, username="stranger", first_name="Stranger")
            update = SimpleNamespace(message=message, effective_user=user)

            await channel._process_forward_command(update, context=None)

            assert handled == [("999|stranger", "/pairing approve ABCD-1234", True)]

        asyncio.run(scenario())
