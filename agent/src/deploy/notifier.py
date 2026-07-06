"""Push deploy events to subscribed IM chats (Telegram etc.).

Subscription is per-chat and explicit (``/deploy notify on`` from an approved
chat) -- stored in ``~/.vibe-trading/deploy_notify.json`` so it survives
restarts and never needs the upstream channel config schema touched.

``publish(event)`` is called from the deploy scheduler's worker threads with
the same event dicts the SSE bus gets; only operator-significant events are
forwarded (orders actually placed, blocks, failures, fills, flatten, kill
switch) -- a converged no-op tick at intraday cadence would be pure noise.
Delivery is strictly best-effort: if the channel runtime isn't running there
is nothing to send through, and trading must never block on chat.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
from typing import Any

from src.config.paths import get_runtime_root

logger = logging.getLogger(__name__)

STORE_FILENAME = "deploy_notify.json"

_LOCK = threading.Lock()
_loop: asyncio.AbstractEventLoop | None = None


def _store_path():
    return get_runtime_root() / STORE_FILENAME


def _load() -> list[dict[str, str]]:
    path = _store_path()
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return [s for s in data.get("subscribers", []) if s.get("channel") and s.get("chat_id")]
    except (OSError, json.JSONDecodeError):
        logger.warning("ignoring invalid deploy notify store at %s", path)
        return []


def _save(subscribers: list[dict[str, str]]) -> None:
    path = _store_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(
        json.dumps({"subscribers": subscribers}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, path)


def subscribe(channel: str, chat_id: str) -> None:
    with _LOCK:
        subs = _load()
        entry = {"channel": channel, "chat_id": str(chat_id)}
        if entry not in subs:
            subs.append(entry)
            _save(subs)


def unsubscribe(channel: str, chat_id: str) -> None:
    with _LOCK:
        subs = _load()
        kept = [s for s in subs if not (s["channel"] == channel and s["chat_id"] == str(chat_id))]
        if len(kept) != len(subs):
            _save(kept)


def is_subscribed(channel: str, chat_id: str) -> bool:
    return any(
        s["channel"] == channel and s["chat_id"] == str(chat_id) for s in _load()
    )


def list_subscribers() -> list[dict[str, str]]:
    return _load()


def bind_loop() -> None:
    """Capture the server's event loop (called at startup, like the SSE bus)."""
    global _loop
    _loop = asyncio.get_event_loop()


def format_event(event: dict[str, Any]) -> str | None:
    """Human message for an event, or None when it isn't worth pushing."""
    kind = event.get("type")
    dep = event.get("deployment_id", "")

    if kind == "tick":
        status = event.get("status")
        label = event.get("symbol") or dep
        if status == "failed":
            return f"❌ {label} 執行失敗\n{event.get('reason')}"
        if status == "blocked":
            return f"🚫 {label} 被擋下\n{event.get('reason')}"
        if status == "ok" and event.get("orders"):
            return (
                f"📈 {label} 已執行 {event.get('orders')} 筆委託"
                f"（bar {event.get('bar_ts', '')}）"
            )
        # Converged no-op ticks stay quiet -- visible in /deploy status, not
        # the notification feed (pure noise at intraday cadence).
        return None
    if kind == "fill":
        fill = event.get("fill") or {}
        return (
            f"✅ 成交 {fill.get('code', '')} {fill.get('action', fill.get('side', ''))} "
            f"{fill.get('quantity', '')} @ {fill.get('price', '')}"
        )
    if kind == "flatten":
        return f"⚪ 部署 {dep} 已停止並平倉"
    if kind == "kill_switch":
        return "🔴 全域緊急停止已啟動" if event.get("engaged") else "🟢 全域緊急停止已解除"
    if kind == "order_placed":
        return (
            f"📤 部署 {dep} 下單 {event.get('side')} {event.get('quantity')} "
            f"{event.get('symbol')}（{event.get('note', '')}）".rstrip("（）")
        )
    return None


def publish(event: dict[str, Any]) -> None:
    """Fan a deploy event out to every subscribed chat (thread-safe)."""
    try:
        message = format_event(event)
        if not message:
            return
        subscribers = _load()
        if not subscribers:
            return
        bus = _channel_bus()
        loop = _loop
        if bus is None or loop is None or loop.is_closed():
            logger.debug("deploy notify dropped (channel runtime not running): %s", message)
            return
        from src.channels.bus.events import OutboundMessage

        for sub in subscribers:
            out = OutboundMessage(
                channel=sub["channel"],
                chat_id=sub["chat_id"],
                content=message,
                metadata={"_deploy_notify": True},
            )
            asyncio.run_coroutine_threadsafe(bus.publish_outbound(out), loop)
    except Exception:  # noqa: BLE001 - notifications must never break trading
        logger.exception("deploy notify failed")


def _channel_bus():
    """The api_server's channel MessageBus, if the runtime has been built."""
    try:
        import api_server

        return getattr(api_server, "_channel_bus", None)
    except Exception:  # noqa: BLE001 - notifier works without the api server
        return None
