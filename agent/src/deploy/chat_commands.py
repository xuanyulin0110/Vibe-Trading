"""Deterministic ``/deploy`` chat commands for IM channels (Telegram etc.).

Intercepted by ``ChannelRuntime`` BEFORE any LLM routing -- these commands
control the deterministic deployment runtime and must stay exactly as
deterministic as the runtime itself. The reply is plain text plus optional
button rows; a button press comes back as an inbound message whose content
is the button label, so buttons are just pre-typed commands.

Authorization: only pairing-approved senders (the same trust boundary the
platform already uses before letting a chat talk to the LLM agent) may run
ANY ``/deploy`` command. Destructive actions keep the same typed-confirmation
discipline as the web UI: ``/deploy flatten`` requires the symbol twice.
"""

from __future__ import annotations

import logging
from typing import Any

from src.channels.pairing.store import is_approved

from src.deploy import notifier, store
from src.deploy.market_calendar import TW_FUTURES

logger = logging.getLogger(__name__)

COMMAND_PREFIX = "/deploy"

HELP_TEXT = """可用指令：
/deploy status — 所有部署狀態
/deploy pos — 券商實際倉位查詢（模擬/正式）
/deploy start <代號> — 啟動部署
/deploy stop <代號> — 停止部署
/deploy dryrun <代號> — 試算下一個 tick（不下單）
/deploy flatten <代號> <代號> — 停止並平倉（代號需輸入兩次確認）
/deploy kill — 全域緊急停止
/deploy resume — 解除全域停止
/deploy notify on|off — 在此對話開/關實盤推播
/deploy help — 顯示本說明"""


def is_deploy_command(content: str) -> bool:
    text = (content or "").strip()
    return text == COMMAND_PREFIX or text.startswith(COMMAND_PREFIX + " ")


def _scheduler():
    from src.deploy import api as deploy_api

    return deploy_api._scheduler


def _find_deployment(token: str) -> store.Deployment | None:
    token = token.strip().upper()
    for dep in store.list_deployments():
        if dep.id == token.lower() or dep.symbol == token:
            return dep
    return None


def _status_line(dep: store.Deployment) -> str:
    env = "正式" if dep.environment == "live" else "模擬"
    state = "🟢運行" if dep.enabled else "⚪停止"
    line = f"{state} {dep.symbol} [{env}/{dep.interval}]"
    if dep.market == TW_FUTURES and dep.sessions == "day_night":
        line += " 日+夜盤"
    if dep.last_tick_status:
        line += f" | 最近: {dep.last_tick_status}"
    if dep.last_error:
        line += f" ⚠ {dep.last_error[:80]}"
    return line


def handle_deploy_command(
    channel: str, sender_id: str, chat_id: str, content: str,
) -> tuple[str, list[list[str]]]:
    """Execute one ``/deploy`` command. Returns (reply_text, button_rows).

    Runs synchronously (callers use a worker thread); dryrun/flatten do real
    broker I/O and can take seconds.
    """
    if not is_approved(channel, sender_id):
        return (
            "此對話尚未配對授權，無法操作實盤部署。請先用 /pairing 完成配對。",
            [],
        )

    parts = (content or "").strip().split()
    sub = parts[1].lower() if len(parts) > 1 else "status"
    args = parts[2:]

    try:
        if sub in ("help", "?"):
            return HELP_TEXT, []
        if sub == "status":
            return _cmd_status(channel, chat_id)
        if sub in ("pos", "positions", "倉位"):
            return _cmd_positions()
        if sub in ("start", "stop"):
            return _cmd_toggle(sub, args)
        if sub == "dryrun":
            return _cmd_dryrun(args)
        if sub == "flatten":
            return _cmd_flatten(args)
        if sub in ("kill", "resume"):
            store.set_kill_switch(sub == "kill")
            event = {"type": "kill_switch", "engaged": store.kill_switch_engaged()}
            notifier.publish(event)  # other subscribed chats hear it too
            try:
                from src.deploy.api import event_bus

                event_bus.publish(event)  # web UI SSE stays in sync
            except Exception:  # noqa: BLE001 - chat must work without the API app
                pass
            state = "已啟動 🔴" if store.kill_switch_engaged() else "已解除 🟢"
            return f"全域緊急停止{state}", []
        if sub == "notify":
            return _cmd_notify(channel, chat_id, args)
        return f"未知指令 {sub!r}。\n\n{HELP_TEXT}", []
    except Exception as exc:  # noqa: BLE001 - chat surface must reply, not die
        logger.exception("deploy chat command failed: %s", content)
        return f"指令執行失敗：{type(exc).__name__}: {exc}", []


def _cmd_status(channel: str, chat_id: str) -> tuple[str, list[list[str]]]:
    deployments = store.list_deployments()
    lines: list[str] = []
    if store.kill_switch_engaged():
        lines.append("🔴 全域緊急停止已啟動")
    if not deployments:
        lines.append("目前沒有任何部署。到網頁 Reports 挑一個回測結果建立。")
        return "\n".join(lines), []
    lines.extend(_status_line(d) for d in deployments)
    subscribed = notifier.is_subscribed(channel, chat_id)
    lines.append(f"\n推播：此對話{'已開啟 🔔' if subscribed else '未開啟（/deploy notify on）'}")

    buttons: list[list[str]] = []
    for dep in deployments[:6]:  # keyboard sanity cap
        action = "stop" if dep.enabled else "start"
        label = f"{COMMAND_PREFIX} {action} {dep.symbol}"
        if len(label.encode("utf-8")) <= 64:  # Telegram callback_data cap
            buttons.append([label])
    buttons.append(
        [f"{COMMAND_PREFIX} resume" if store.kill_switch_engaged() else f"{COMMAND_PREFIX} kill"]
    )
    return "\n".join(lines), buttons


def _cmd_positions() -> tuple[str, list[list[str]]]:
    """Live broker positions, per environment, with deployment attribution."""
    from src.deploy import contracts
    from src.trading.connectors.shioaji import sdk

    scheduler = _scheduler()
    if scheduler is None:
        return "排程器未運行，無法查詢倉位。", []

    deployments = store.list_deployments()
    environments = sorted({d.environment for d in deployments}) or ["paper"]
    lines: list[str] = []
    for env in environments:
        header = "【模擬倉】" if env == "paper" else "【正式環境】"
        try:
            with scheduler.sessions.use(env) as api:
                payload = sdk.get_positions(None, api=api)
        except Exception as exc:  # noqa: BLE001 - report, don't die
            lines.append(f"{header} 查詢失敗：{exc}")
            continue
        positions = list(payload.get("positions") or [])
        if not positions:
            lines.append(f"{header} 無持倉")
            continue
        lines.append(header)
        env_symbols = {d.symbol for d in deployments if d.environment == env}
        env_products = {
            contracts.product_of(d.symbol)
            for d in deployments
            if d.environment == env and d.market == TW_FUTURES
        }
        for pos in positions:
            code = str(pos.get("code") or "")
            qty = pos.get("quantity")
            direction = str(pos.get("direction") or "")
            side = "多" if "buy" in direction.lower() or direction in ("1", "Action.Buy") else (
                "空" if "sell" in direction.lower() else direction or "?"
            )
            is_futures_code = contracts.product_of(code) in env_products
            unit = "口" if is_futures_code else "張"
            owned = (
                is_futures_code
                or f"{code}.TW" in env_symbols
                or code in env_symbols
            )
            tag = "" if owned else "（非部署部位）"
            entry = pos.get("price")
            last = pos.get("last_price")
            pnl = pos.get("pnl")
            line = f"  {code} {side} {qty}{unit} 均價{entry} 現價{last}"
            if pnl is not None:
                try:
                    line += f" 未實現{float(pnl):+,.0f}"
                except (TypeError, ValueError):
                    line += f" 未實現{pnl}"
            lines.append(line + tag)
    return "\n".join(lines), []


def _cmd_toggle(sub: str, args: list[str]) -> tuple[str, list[list[str]]]:
    if not args:
        return f"用法：/deploy {sub} <代號>", []
    dep = _find_deployment(args[0])
    if dep is None:
        return f"找不到部署 {args[0]}", []
    store.update_deployment(dep.id, enabled=sub == "start", paused_reason=None)
    scheduler = _scheduler()
    if scheduler is not None:
        scheduler.on_deployment_toggled()
    verb = "已啟動 🟢" if sub == "start" else "已停止 ⚪"
    return f"{dep.symbol} {verb}", []


def _cmd_dryrun(args: list[str]) -> tuple[str, list[list[str]]]:
    if not args:
        return "用法：/deploy dryrun <代號>", []
    dep = _find_deployment(args[0])
    if dep is None:
        return f"找不到部署 {args[0]}", []
    scheduler = _scheduler()
    if scheduler is None:
        return "排程器未運行，無法試算。", []
    outcome = scheduler.run_once(dep.id, dry_run=True)
    orders = outcome.orders or []
    lines = [
        f"Dry-run {dep.symbol}（未下單）",
        f"bar: {outcome.bar_ts} | 訊號權重: {outcome.weight}",
        f"目標/現況: {outcome.target_qty}/{outcome.current_qty}",
        f"結果: {outcome.status} — {outcome.reason}",
    ]
    for order in orders:
        lines.append(f"  將下單: {order.get('side')} {order.get('quantity')} {order.get('symbol')}")
    return "\n".join(lines), []


def _cmd_flatten(args: list[str]) -> tuple[str, list[list[str]]]:
    if len(args) < 2 or args[0].strip().upper() != args[1].strip().upper():
        target = args[0] if args else "<代號>"
        return (
            f"平倉需要輸入代號兩次確認：/deploy flatten {target} {target}\n"
            "（停用部署並以市價單將持倉全部歸零，無法還原）",
            [],
        )
    dep = _find_deployment(args[0])
    if dep is None:
        return f"找不到部署 {args[0]}", []
    scheduler = _scheduler()
    if scheduler is None:
        return "排程器未運行，無法平倉。", []
    results = scheduler.flatten(dep.id)
    oks = sum(1 for r in results if (r.get("response") or {}).get("status") == "ok")
    return f"{dep.symbol} 已停止並送出平倉（{oks} 筆委託成功）。用 /deploy status 確認。", []


def _cmd_notify(channel: str, chat_id: str, args: list[str]) -> tuple[str, list[list[str]]]:
    mode = args[0].lower() if args else ""
    if mode == "on":
        notifier.subscribe(channel, chat_id)
        return "已開啟此對話的實盤推播 🔔（成交/失敗/平倉/緊急停止都會通知）", []
    if mode == "off":
        notifier.unsubscribe(channel, chat_id)
        return "已關閉此對話的實盤推播。", []
    return "用法：/deploy notify on|off", []


__all__: list[Any] = ["is_deploy_command", "handle_deploy_command", "COMMAND_PREFIX"]
