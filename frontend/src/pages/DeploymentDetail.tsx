import { useCallback, useEffect, useRef, useState } from "react";
import { Link, useParams } from "react-router-dom";
import {
  Activity,
  AlertTriangle,
  ArrowLeft,
  CircleStop,
  FlaskConical,
  Loader2,
  OctagonX,
  Play,
} from "lucide-react";
import {
  api,
  type DeploymentHistory,
  type DeploymentItem,
  type DeploymentTickRecord,
  type PriceBar,
  type TradeMarker,
} from "@/lib/api";
import { CandlestickChart } from "@/components/charts/CandlestickChart";
import { EquityChart } from "@/components/charts/EquityChart";
import { ConfirmDialog } from "@/pages/Deployments";
import { useSSE } from "@/hooks/useSSE";
import { cn } from "@/lib/utils";

export function DeploymentDetail() {
  const { deploymentId = "" } = useParams();
  const [dep, setDep] = useState<DeploymentItem | null>(null);
  const [history, setHistory] = useState<DeploymentHistory>({ ticks: [], fills: [] });
  const [equity, setEquity] = useState<Array<{ time: string; equity: number; drawdown: number }>>([]);
  const [bars, setBars] = useState<PriceBar[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const [flattenOpen, setFlattenOpen] = useState(false);
  const [dryRunResult, setDryRunResult] = useState<DeploymentTickRecord | null>(null);
  const sse = useSSE();
  const mounted = useRef(true);

  const load = useCallback(async () => {
    try {
      const [depRes, histRes, eqRes] = await Promise.all([
        api.getDeployment(deploymentId),
        api.getDeploymentHistory(deploymentId),
        api.getDeploymentEquity(deploymentId),
      ]);
      if (!mounted.current) return;
      setDep(depRes);
      setHistory(histRes);
      let peak = Number.NEGATIVE_INFINITY;
      setEquity(
        eqRes.points.map((p) => {
          peak = Math.max(peak, p.equity);
          const drawdown = peak > 0 ? (p.equity - peak) / peak : 0;
          return { time: p.ts, equity: p.equity, drawdown };
        }),
      );
      setError(null);
    } catch (err) {
      if (mounted.current) setError(err instanceof Error ? err.message : "load failed");
    }
  }, [deploymentId]);

  useEffect(() => {
    mounted.current = true;
    load();
    api
      .getDeploymentBars(deploymentId)
      .then((res) => mounted.current && setBars(res.bars))
      .catch(() => {});
    sse.connect(api.deploymentEventsUrl(), {
      message: (event) => {
        if (event.deployment_id === deploymentId) load();
      },
    });
    return () => {
      mounted.current = false;
      sse.disconnect();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [deploymentId, load]);

  const markers: TradeMarker[] = history.ticks.flatMap((tick) =>
    (tick.orders ?? [])
      .filter((o) => (o as { response?: { status?: string } }).response?.status === "ok")
      .map((o) => {
        const order = o as { side?: string; requested?: number; response?: { limit_price?: number } };
        return {
          time: String(tick.bar_ts ?? tick.executed_at ?? ""),
          side: (order.side === "sell" ? "SELL" : "BUY") as "BUY" | "SELL",
          price: Number(tick.signal_close ?? 0),
          qty: Number(order.requested ?? 0),
          reason: String((o as { kind?: string }).kind ?? "signal"),
        };
      }),
  );

  const act = async (action: "start" | "stop" | "dry" | "flatten", confirmSymbol?: string) => {
    setBusy(action);
    setError(null);
    try {
      if (action === "start") await api.startDeployment(deploymentId);
      if (action === "stop") await api.stopDeployment(deploymentId);
      if (action === "dry") setDryRunResult(await api.runDeploymentOnce(deploymentId, true));
      if (action === "flatten" && confirmSymbol) {
        await api.flattenDeployment(deploymentId, confirmSymbol);
      }
      await load();
    } catch (err) {
      setError(err instanceof Error ? err.message : `${action} failed`);
    } finally {
      setBusy(null);
    }
  };

  if (!dep) {
    return (
      <div className="flex h-[60vh] items-center justify-center text-muted-foreground">
        {error ?? "Loading…"}
      </div>
    );
  }

  return (
    <div className="min-h-screen p-6 lg:p-8">
      <div className="mx-auto flex w-full max-w-6xl flex-col gap-6">
        <section className="flex flex-col gap-4 border-b pb-6 lg:flex-row lg:items-end lg:justify-between">
          <div>
            <Link to="/deployments" className="inline-flex items-center gap-1 text-sm text-muted-foreground hover:text-foreground">
              <ArrowLeft className="h-4 w-4" /> 部署清單
            </Link>
            <div className="mt-2 flex flex-wrap items-center gap-2">
              <h1 className="font-mono text-2xl font-bold">{dep.symbol}</h1>
              <span
                className={cn(
                  "rounded px-2 py-0.5 text-xs font-medium",
                  dep.environment === "live" ? "bg-danger/10 text-danger" : "bg-muted text-muted-foreground",
                )}
              >
                {dep.environment === "live" ? "正式" : "模擬"}
              </span>
              <span className="rounded bg-muted px-2 py-0.5 text-xs">{dep.interval}</span>
              <span
                className={cn(
                  "inline-flex items-center gap-1 rounded px-2 py-0.5 text-xs font-medium",
                  dep.enabled ? "bg-success/10 text-success" : "bg-muted text-muted-foreground",
                )}
              >
                <Activity className="h-3 w-3" /> {dep.enabled ? "運行中" : "已停止"}
              </span>
            </div>
            <p className="mt-1 text-sm text-muted-foreground">
              策略 <Link to={`/runs/${dep.run_id}`} className="font-mono hover:text-primary">{dep.run_id}</Link>
              {" · "}配置資金 {dep.allocated_capital.toLocaleString()} TWD
              {" · "}上限：單筆 {dep.max_order_qty}、日 {dep.max_daily_orders} 筆、名目 {dep.max_order_notional.toLocaleString()}
            </p>
          </div>
          <div className="flex items-center gap-2">
            <button
              type="button"
              disabled={busy !== null}
              onClick={() => act(dep.enabled ? "stop" : "start")}
              className={cn(
                "inline-flex items-center gap-1.5 rounded-md border px-4 py-2 text-sm font-medium",
                dep.enabled ? "border-danger text-danger hover:bg-danger/10" : "hover:bg-muted",
              )}
            >
              {busy === "start" || busy === "stop" ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : dep.enabled ? (
                <CircleStop className="h-4 w-4" />
              ) : (
                <Play className="h-4 w-4" />
              )}
              {dep.enabled ? "停止" : "啟動"}
            </button>
            <button
              type="button"
              disabled={busy !== null}
              onClick={() => act("dry")}
              className="inline-flex items-center gap-1.5 rounded-md border px-4 py-2 text-sm font-medium hover:bg-muted"
            >
              {busy === "dry" ? <Loader2 className="h-4 w-4 animate-spin" /> : <FlaskConical className="h-4 w-4" />}
              Dry-run 預覽
            </button>
            <button
              type="button"
              disabled={busy !== null}
              onClick={() => setFlattenOpen(true)}
              className="inline-flex items-center gap-1.5 rounded-md border border-danger px-4 py-2 text-sm font-semibold text-danger hover:bg-danger/10"
            >
              <OctagonX className="h-4 w-4" /> 停止並平倉
            </button>
          </div>
        </section>

        {error ? (
          <div className="flex items-center gap-2 rounded-md border border-amber-500/30 bg-amber-500/5 p-4 text-sm">
            <AlertTriangle className="h-4 w-4 text-amber-600" /> {error}
          </div>
        ) : null}

        {dryRunResult ? (
          <section className="rounded-md border bg-muted/20 p-4 text-sm">
            <div className="flex items-center justify-between">
              <h2 className="font-medium">Dry-run 結果（未下單）</h2>
              <button type="button" className="text-xs text-muted-foreground" onClick={() => setDryRunResult(null)}>
                關閉
              </button>
            </div>
            <pre className="mt-2 overflow-x-auto rounded bg-background p-3 font-mono text-xs">
              {JSON.stringify(dryRunResult, null, 2)}
            </pre>
          </section>
        ) : null}

        {equity.length >= 2 ? (
          <section className="rounded-md border p-4">
            <h2 className="mb-3 text-sm font-medium text-muted-foreground">實盤權益曲線</h2>
            <EquityChart data={equity} height={260} />
          </section>
        ) : null}

        {bars.length > 0 ? (
          <section className="rounded-md border p-4">
            <h2 className="mb-3 text-sm font-medium text-muted-foreground">
              {dep.symbol} 近期走勢與實際成交
            </h2>
            <CandlestickChart data={bars} markers={markers} height={420} />
          </section>
        ) : null}

        <section className="rounded-md border p-4">
          <h2 className="mb-3 text-sm font-medium text-muted-foreground">執行歷史</h2>
          {history.ticks.length === 0 ? (
            <p className="text-sm text-muted-foreground">尚無執行紀錄。</p>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-left text-sm">
                <thead className="text-xs uppercase text-muted-foreground">
                  <tr>
                    <th className="py-2 pr-4">Bar</th>
                    <th className="py-2 pr-4">訊號</th>
                    <th className="py-2 pr-4">目標/現況</th>
                    <th className="py-2 pr-4">狀態</th>
                    <th className="py-2 pr-4">委託</th>
                    <th className="py-2 pr-4">耗時</th>
                  </tr>
                </thead>
                <tbody>
                  {[...history.ticks].reverse().map((tick, i) => (
                    <tr key={i} className="border-t">
                      <td className="py-2 pr-4 font-mono text-xs">{tick.bar_ts}</td>
                      <td className="py-2 pr-4 font-mono">{tick.signal_weight?.toFixed(3) ?? "-"}</td>
                      <td className="py-2 pr-4 font-mono">
                        {tick.target_qty ?? "-"} / {tick.current_qty ?? "-"}
                      </td>
                      <td
                        className={cn(
                          "py-2 pr-4",
                          tick.status === "ok" && "text-success",
                          (tick.status === "blocked" || tick.status === "failed") && "text-danger",
                        )}
                      >
                        {tick.status ?? tick.phase}
                        {tick.note ? <span className="block text-xs text-muted-foreground">{tick.note}</span> : null}
                      </td>
                      <td className="py-2 pr-4">{tick.orders?.length ?? 0}</td>
                      <td className="py-2 pr-4 font-mono text-xs">
                        {tick.elapsed_seconds ? `${tick.elapsed_seconds.toFixed(1)}s` : "-"}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </section>

        <section className="rounded-md border p-4">
          <h2 className="mb-3 text-sm font-medium text-muted-foreground">成交明細（order_deal_event）</h2>
          {history.fills.length === 0 ? (
            <p className="text-sm text-muted-foreground">尚無成交回報。</p>
          ) : (
            <pre className="max-h-64 overflow-auto rounded bg-muted/20 p-3 font-mono text-xs">
              {JSON.stringify(history.fills.slice(-50), null, 2)}
            </pre>
          )}
        </section>
      </div>

      {flattenOpen ? (
        <ConfirmDialog
          title="停止並平倉"
          body={`停用此部署並以市價單將 ${dep.symbol} 的持倉全部歸零。此動作無法還原。請輸入標的代號確認。`}
          confirmLabel="平倉"
          danger
          typedConfirmation={dep.symbol}
          onCancel={() => setFlattenOpen(false)}
          onConfirm={() => {
            setFlattenOpen(false);
            act("flatten", dep.symbol);
          }}
        />
      ) : null}
    </div>
  );
}
