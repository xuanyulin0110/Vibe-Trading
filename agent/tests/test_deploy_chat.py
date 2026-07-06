"""/deploy chat commands + IM notifier + channel-runtime interception."""

from __future__ import annotations

import asyncio
import json

import pytest

from src.deploy import chat_commands, notifier
from src.deploy import store as deploy_store
from src.deploy.executor import TickOutcome


@pytest.fixture
def tmp_stores(monkeypatch, tmp_path):
    monkeypatch.setattr(deploy_store, "get_runtime_root", lambda: tmp_path)
    monkeypatch.setattr(notifier, "get_runtime_root", lambda: tmp_path)
    return tmp_path


@pytest.fixture
def approved(monkeypatch):
    monkeypatch.setattr(chat_commands, "is_approved", lambda channel, sender: True)


def _mk_deployment(**kw):
    defaults = dict(
        run_id="runX", symbol="TXFR1.TWF", market="tw_futures", environment="paper",
        interval="1D", allocated_capital=1_000_000.0, max_order_qty=5,
        max_daily_orders=10, max_order_notional=10_000_000.0,
    )
    defaults.update(kw)
    return deploy_store.create_deployment(**defaults)


class _StubScheduler:
    def __init__(self):
        self.flattened: list[str] = []
        self.toggles = 0

    def run_once(self, deployment_id, *, dry_run):
        return TickOutcome(
            "dry_run", "planned", bar_ts="2026-07-04", weight=1.0,
            target_qty=2, current_qty=0,
            orders=[{"side": "buy", "quantity": 2, "symbol": "TMF202607.TWF"}],
        )

    def flatten(self, deployment_id):
        self.flattened.append(deployment_id)
        return [{"response": {"status": "ok"}}]

    def on_deployment_toggled(self):
        self.toggles += 1


@pytest.fixture
def stub_scheduler(monkeypatch):
    from src.deploy import api as deploy_api

    stub = _StubScheduler()
    monkeypatch.setattr(deploy_api, "_scheduler", stub)
    return stub


def _run(content: str) -> tuple[str, list]:
    return chat_commands.handle_deploy_command("telegram", "user1", "chat1", content)


class TestDeployChatCommands:
    def test_unapproved_sender_is_refused(self, tmp_stores, monkeypatch) -> None:
        monkeypatch.setattr(chat_commands, "is_approved", lambda c, s: False)
        reply, buttons = _run("/deploy status")
        assert "配對" in reply
        assert buttons == []

    def test_is_deploy_command(self) -> None:
        assert chat_commands.is_deploy_command("/deploy")
        assert chat_commands.is_deploy_command("/deploy status")
        assert not chat_commands.is_deploy_command("/deployments")
        assert not chat_commands.is_deploy_command("hello /deploy")

    def test_status_lists_deployments_with_buttons(self, tmp_stores, approved) -> None:
        _mk_deployment()
        reply, buttons = _run("/deploy status")
        assert "TXFR1.TWF" in reply
        assert any("/deploy start TXFR1.TWF" in row[0] for row in buttons)
        assert any("/deploy kill" in row[0] for row in buttons)

    def test_start_stop_toggle(self, tmp_stores, approved, stub_scheduler) -> None:
        dep = _mk_deployment()
        reply, _ = _run("/deploy start TXFR1.TWF")
        assert "啟動" in reply
        assert deploy_store.get_deployment(dep.id).enabled is True
        reply, _ = _run(f"/deploy stop {dep.id}")  # by id also works
        assert deploy_store.get_deployment(dep.id).enabled is False
        assert stub_scheduler.toggles == 2

    def test_kill_and_resume(self, tmp_stores, approved) -> None:
        reply, _ = _run("/deploy kill")
        assert deploy_store.kill_switch_engaged() is True
        assert "已啟動" in reply
        reply, _ = _run("/deploy resume")
        assert deploy_store.kill_switch_engaged() is False

    def test_flatten_requires_symbol_twice(self, tmp_stores, approved, stub_scheduler) -> None:
        _mk_deployment()
        reply, _ = _run("/deploy flatten TXFR1.TWF")
        assert "兩次" in reply
        assert stub_scheduler.flattened == []
        reply, _ = _run("/deploy flatten TXFR1.TWF TXFR1.TWF")
        assert len(stub_scheduler.flattened) == 1

    def test_dryrun_reports_planned_orders(self, tmp_stores, approved, stub_scheduler) -> None:
        _mk_deployment()
        reply, _ = _run("/deploy dryrun TXFR1.TWF")
        assert "Dry-run" in reply
        assert "buy 2" in reply

    def test_notify_on_off(self, tmp_stores, approved) -> None:
        reply, _ = _run("/deploy notify on")
        assert notifier.is_subscribed("telegram", "chat1")
        reply, _ = _run("/deploy notify off")
        assert not notifier.is_subscribed("telegram", "chat1")

    def test_unknown_subcommand_shows_help(self, tmp_stores, approved) -> None:
        reply, _ = _run("/deploy bogus")
        assert "/deploy status" in reply


class TestNotifier:
    def test_subscription_roundtrip(self, tmp_stores) -> None:
        notifier.subscribe("telegram", "42")
        notifier.subscribe("telegram", "42")  # idempotent
        assert notifier.list_subscribers() == [{"channel": "telegram", "chat_id": "42"}]
        notifier.unsubscribe("telegram", "42")
        assert notifier.list_subscribers() == []

    @pytest.mark.parametrize(
        "event,expected",
        [
            ({"type": "tick", "symbol": "X", "status": "failed", "reason": "boom"}, "執行失敗"),
            ({"type": "tick", "symbol": "X", "status": "blocked", "reason": "cap"}, "被擋下"),
            ({"type": "tick", "symbol": "X", "status": "ok", "orders": 2, "bar_ts": "t"}, "2 筆委託"),
            ({"type": "fill", "fill": {"code": "TMF", "action": "Buy", "quantity": 1, "price": 23000}}, "成交"),
            ({"type": "flatten", "deployment_id": "d1"}, "平倉"),
            ({"type": "kill_switch", "engaged": True}, "緊急停止"),
        ],
    )
    def test_format_pushworthy_events(self, event, expected) -> None:
        assert expected in notifier.format_event(event)

    @pytest.mark.parametrize(
        "event",
        [
            {"type": "tick", "status": "ok", "orders": 0},  # converged no-op
            {"type": "tick", "status": "skipped"},
            {"type": "created", "deployment_id": "d1"},
            {"type": "toggled", "deployment_id": "d1"},
        ],
    )
    def test_quiet_events_are_dropped(self, event) -> None:
        assert notifier.format_event(event) is None

    def test_publish_sends_to_subscribed_chats(self, tmp_stores, monkeypatch) -> None:
        notifier.subscribe("telegram", "42")

        sent = []

        class _Bus:
            async def publish_outbound(self, msg):
                sent.append(msg)

        loop = asyncio.new_event_loop()
        try:
            monkeypatch.setattr(notifier, "_channel_bus", lambda: _Bus())
            monkeypatch.setattr(notifier, "_loop", loop)
            notifier.publish({"type": "kill_switch", "engaged": True})
            loop.run_until_complete(asyncio.sleep(0.05))
        finally:
            loop.close()
        assert len(sent) == 1
        assert sent[0].channel == "telegram"
        assert sent[0].chat_id == "42"
        assert "緊急停止" in sent[0].content

    def test_publish_without_runtime_is_silent(self, tmp_stores, monkeypatch) -> None:
        notifier.subscribe("telegram", "42")
        monkeypatch.setattr(notifier, "_channel_bus", lambda: None)
        notifier.publish({"type": "kill_switch", "engaged": True})  # must not raise


class TestRuntimeInterception:
    def test_deploy_command_never_reaches_llm_session(self, tmp_stores, monkeypatch) -> None:
        """A /deploy message is answered deterministically -- the session
        service (LLM path) must not be touched."""
        from src.channels.bus.events import InboundMessage
        from src.channels.bus.queue import MessageBus
        from src.channels.runtime import ChannelRuntime

        monkeypatch.setattr(chat_commands, "is_approved", lambda c, s: True)

        class _ExplodingSessionService:
            def create_session(self, **kw):  # pragma: no cover - must not run
                raise AssertionError("LLM session must not be created for /deploy")

            def send_message(self, *a, **kw):  # pragma: no cover
                raise AssertionError("LLM must not be invoked for /deploy")

        bus = MessageBus()
        runtime = ChannelRuntime(
            bus=bus,
            session_service=_ExplodingSessionService(),
            manager=None,
            session_map_path=tmp_stores / "sessions.json",
        )

        async def _exercise():
            msg = InboundMessage(
                channel="telegram", sender_id="u1", chat_id="c1", content="/deploy status",
            )
            await runtime._handle_inbound(msg)
            return await asyncio.wait_for(bus.consume_outbound(), timeout=5)

        out = asyncio.new_event_loop().run_until_complete(_exercise())
        assert out.metadata.get("_deploy_command") is True
        assert "部署" in out.content
