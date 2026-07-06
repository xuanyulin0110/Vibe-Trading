import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Link, useSearchParams } from "react-router-dom";
import {
  Activity,
  AlertTriangle,
  CircleStop,
  Loader2,
  OctagonX,
  Play,
  Plus,
  RefreshCw,
  Rocket,
} from "lucide-react";
import { api, type CreateDeploymentBody, type DeploymentItem, type DeploymentListResponse } from "@/lib/api";
import { MiniEquityChart } from "@/components/charts/MiniEquityChart";
import { useSSE } from "@/hooks/useSSE";
import { cn } from "@/lib/utils";

const POLL_MS = 30_000;

export function Deployments() {
  const [data, setData] = useState<DeploymentListResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const [killConfirm, setKillConfirm] = useState(false);
  const [searchParams, setSearchParams] = useSearchParams();
  const [showCreate, setShowCreate] = useState(() => Boolean(searchParams.get("create")));
  const sse = useSSE();
  const mounted = useRef(true);

  const load = useCallback(async () => {
    try {
      const next = await api.listDeployments();
      if (mounted.current) {
        setData(next);
        setError(null);
      }
    } catch (err) {
      if (mounted.current) setError(err instanceof Error ? err.message : "load failed");
    } finally {
      if (mounted.current) setLoading(false);
    }
  }, []);

  useEffect(() => {
    mounted.current = true;
    load();
    const timer = window.setInterval(load, POLL_MS);
    // Live updates: any deployment event refreshes the list.
    sse.connect(api.deploymentEventsUrl(), { message: () => { load(); } });
    return () => {
      mounted.current = false;
      window.clearInterval(timer);
      sse.disconnect();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [load]);

  const toggle = async (dep: DeploymentItem) => {
    setBusy(dep.id);
    try {
      if (dep.enabled) await api.stopDeployment(dep.id);
      else await api.startDeployment(dep.id);
      await load();
    } catch (err) {
      setError(err instanceof Error ? err.message : "toggle failed");
    } finally {
      setBusy(null);
    }
  };

  const setKill = async (engaged: boolean) => {
    setKillConfirm(false);
    try {
      await api.setDeployKillSwitch(engaged);
      await load();
    } catch (err) {
      setError(err instanceof Error ? err.message : "kill switch failed");
    }
  };

  const deployments = data?.deployments ?? [];
  const running = useMemo(
    () => deployments.filter((d) => d.enabled).length,
    [deployments],
  );

  return (
    <div className="min-h-screen p-6 lg:p-8">
      <div className="mx-auto flex w-full max-w-6xl flex-col gap-6">
        <section className="flex flex-col gap-4 border-b pb-6 lg:flex-row lg:items-end lg:justify-between">
          <div className="space-y-3">
            <div className="inline-flex items-center gap-2 rounded-md border px-2.5 py-1 text-xs font-medium text-muted-foreground">
              <Rocket className="h-3.5 w-3.5" />
              Deterministic Deployments
            </div>
            <div>
              <h1 className="text-3xl font-bold tracking-tight">實盤部署</h1>
              <p className="mt-2 max-w-2xl text-sm text-muted-foreground">
                回測策略以確定性排程直接執行（無 LLM 參與下單決策）。訊號、換算與費用口徑與回測引擎共用同一份程式碼。
              </p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={() => setShowCreate(true)}
              className="inline-flex items-center gap-2 rounded-md border px-4 py-2 text-sm font-medium transition hover:bg-muted"
            >
              <Plus className="h-4 w-4" /> 新增部署
            </button>
            {data?.kill_switch ? (
              <button
                type="button"
                onClick={() => setKill(false)}
                className="inline-flex items-center gap-2 rounded-md border border-danger bg-danger/10 px-4 py-2 text-sm font-semibold text-danger"
              >
                <OctagonX className="h-4 w-4" /> 解除全域停止
              </button>
            ) : (
              <button
                type="button"
                onClick={() => setKillConfirm(true)}
                className="inline-flex items-center gap-2 rounded-md border border-danger px-4 py-2 text-sm font-semibold text-danger transition hover:bg-danger/10"
              >
                <OctagonX className="h-4 w-4" /> 全域緊急停止
              </button>
            )}
            <button
              type="button"
              onClick={load}
              className="inline-flex items-center gap-2 rounded-md border px-3 py-2 text-sm"
              aria-label="refresh"
            >
              <RefreshCw className="h-4 w-4" />
            </button>
          </div>
        </section>

        {data?.kill_switch ? (
          <div className="flex items-center gap-2 rounded-md border border-danger bg-danger/10 p-4 text-sm font-medium text-danger">
            <OctagonX className="h-5 w-5" /> 全域緊急停止已啟動：所有部署暫停排程與下單。
          </div>
        ) : null}

        {error ? (
          <div className="flex items-center gap-2 rounded-md border border-amber-500/30 bg-amber-500/5 p-4 text-sm">
            <AlertTriangle className="h-4 w-4 text-amber-600" /> {error}
          </div>
        ) : null}

        {loading ? (
          <div className="grid gap-3 md:grid-cols-2">
            {[1, 2].map((i) => (
              <div key={i} className="h-40 animate-pulse rounded-md border bg-muted/40" />
            ))}
          </div>
        ) : deployments.length === 0 ? (
          <section className="rounded-md border border-dashed p-10 text-center">
            <Rocket className="mx-auto h-8 w-8 text-muted-foreground" />
            <h2 className="mt-3 font-medium">還沒有任何部署</h2>
            <p className="mt-1 text-sm text-muted-foreground">
              從 Reports 挑一個回測結果，點「部署此策略」，或按右上角「新增部署」。
            </p>
          </section>
        ) : (
          <section className="grid gap-4 md:grid-cols-2">
            {deployments.map((dep) => (
              <DeploymentCard
                key={dep.id}
                dep={dep}
                busy={busy === dep.id}
                onToggle={() => toggle(dep)}
              />
            ))}
          </section>
        )}

        <p className="text-xs text-muted-foreground">
          運行中 {running} / {deployments.length}。Session：
          {Object.entries(data?.sessions ?? {}).map(([env, s]) => (
            <span key={env} className="ml-2">
              {env}: {s.failed ? `失敗（${s.failed}）` : s.connected ? "已連線" : "閒置"}
            </span>
          ))}
        </p>
      </div>

      {killConfirm ? (
        <ConfirmDialog
          title="全域緊急停止"
          body="停止所有部署的排程與下單（不平倉）。持倉將維持現狀，可再逐一使用「停止並平倉」。確定？"
          confirmLabel="確定停止"
          danger
          onCancel={() => setKillConfirm(false)}
          onConfirm={() => setKill(true)}
        />
      ) : null}

      {showCreate ? (
        <CreateDeploymentDialog
          initialRunId={searchParams.get("create") || ""}
          onClose={(created) => {
            setShowCreate(false);
            if (searchParams.get("create")) {
              searchParams.delete("create");
              setSearchParams(searchParams, { replace: true });
            }
            if (created) load();
          }}
        />
      ) : null}
    </div>
  );
}

function DeploymentCard({
  dep,
  busy,
  onToggle,
}: {
  dep: DeploymentItem;
  busy: boolean;
  onToggle: () => void;
}) {
  const [equity, setEquity] = useState<Array<{ time: string; equity: number }>>([]);
  useEffect(() => {
    api
      .getDeploymentEquity(dep.id)
      .then((res) => setEquity(res.points.map((p) => ({ time: p.ts, equity: p.equity }))))
      .catch(() => {});
  }, [dep.id]);

  const statusTone =
    dep.last_error || dep.last_tick_status === "failed"
      ? "text-danger"
      : dep.enabled
        ? "text-success"
        : "text-muted-foreground";

  return (
    <article className="rounded-md border p-4">
      <div className="flex items-start justify-between gap-3">
        <div>
          <div className="flex flex-wrap items-center gap-2">
            <Link to={`/deployments/${dep.id}`} className="font-mono font-semibold hover:text-primary">
              {dep.symbol}
            </Link>
            <Pill tone={dep.environment === "live" ? "danger" : "neutral"}>
              {dep.environment === "live" ? "正式" : "模擬"}
            </Pill>
            <Pill tone="neutral">{dep.interval}</Pill>
            {dep.market === "tw_futures" && dep.sessions === "day_night" ? (
              <Pill tone="neutral">日+夜盤</Pill>
            ) : null}
          </div>
          <p className="mt-1 text-xs text-muted-foreground">
            策略 <Link to={`/runs/${dep.run_id}`} className="font-mono hover:text-primary">{dep.run_id}</Link>
            {" · "}配置資金 {dep.allocated_capital.toLocaleString()}
          </p>
        </div>
        <button
          type="button"
          disabled={busy}
          onClick={onToggle}
          className={cn(
            "inline-flex items-center gap-1.5 rounded-md border px-3 py-1.5 text-sm font-medium transition",
            dep.enabled ? "border-danger text-danger hover:bg-danger/10" : "hover:bg-muted",
          )}
        >
          {busy ? (
            <Loader2 className="h-4 w-4 animate-spin" />
          ) : dep.enabled ? (
            <CircleStop className="h-4 w-4" />
          ) : (
            <Play className="h-4 w-4" />
          )}
          {dep.enabled ? "停止" : "啟動"}
        </button>
      </div>

      {equity.length >= 2 ? (
        <div className="mt-3">
          <MiniEquityChart data={equity} height={64} />
        </div>
      ) : null}

      <div className={cn("mt-3 flex items-center gap-2 text-xs", statusTone)}>
        <Activity className="h-3.5 w-3.5" />
        {dep.last_tick_status
          ? `最近執行 ${dep.last_tick_status}${dep.last_tick_at ? ` @ ${dep.last_tick_at.slice(0, 19)}` : ""}`
          : "尚未執行"}
        {dep.last_error ? <span className="truncate">— {dep.last_error}</span> : null}
      </div>
    </article>
  );
}

function Pill({ children, tone }: { children: React.ReactNode; tone: "danger" | "neutral" | "success" }) {
  return (
    <span
      className={cn(
        "inline-flex items-center rounded px-2 py-0.5 text-xs font-medium",
        tone === "danger" && "bg-danger/10 text-danger",
        tone === "success" && "bg-success/10 text-success",
        tone === "neutral" && "bg-muted text-muted-foreground",
      )}
    >
      {children}
    </span>
  );
}

export function ConfirmDialog({
  title,
  body,
  confirmLabel,
  danger,
  typedConfirmation,
  onCancel,
  onConfirm,
}: {
  title: string;
  body: string;
  confirmLabel: string;
  danger?: boolean;
  typedConfirmation?: string; // must type this exact string to enable confirm
  onCancel: () => void;
  onConfirm: () => void;
}) {
  const [typed, setTyped] = useState("");
  const blocked = Boolean(typedConfirmation) && typed.trim().toUpperCase() !== typedConfirmation?.toUpperCase();
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4">
      <div className="w-full max-w-md rounded-md border bg-background p-5 shadow-lg">
        <h2 className="font-semibold">{title}</h2>
        <p className="mt-2 text-sm text-muted-foreground">{body}</p>
        {typedConfirmation ? (
          <input
            value={typed}
            onChange={(e) => setTyped(e.target.value)}
            placeholder={`輸入 ${typedConfirmation} 確認`}
            className="mt-3 w-full rounded-md border bg-transparent px-3 py-2 font-mono text-sm"
          />
        ) : null}
        <div className="mt-4 flex justify-end gap-2">
          <button type="button" onClick={onCancel} className="rounded-md border px-4 py-2 text-sm">
            取消
          </button>
          <button
            type="button"
            disabled={blocked}
            onClick={onConfirm}
            className={cn(
              "rounded-md border px-4 py-2 text-sm font-semibold disabled:opacity-40",
              danger ? "border-danger bg-danger/10 text-danger" : "hover:bg-muted",
            )}
          >
            {confirmLabel}
          </button>
        </div>
      </div>
    </div>
  );
}

function CreateDeploymentDialog({
  initialRunId,
  onClose,
}: {
  initialRunId: string;
  onClose: (created: boolean) => void;
}) {
  const [form, setForm] = useState({
    run_id: initialRunId,
    environment: "paper" as "paper" | "live",
    sessions: "day" as "day" | "day_night",
    allocated_capital: "",
    max_order_qty: "",
    max_daily_orders: "",
    max_order_notional: "",
    confirm_symbol: "",
  });
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const submit = async () => {
    setSubmitting(true);
    setError(null);
    const body: CreateDeploymentBody = {
      run_id: form.run_id.trim(),
      environment: form.environment,
      sessions: form.sessions,
      allocated_capital: Number(form.allocated_capital),
      max_order_qty: Number(form.max_order_qty),
      max_daily_orders: Number(form.max_daily_orders),
      max_order_notional: Number(form.max_order_notional),
    };
    if (form.environment === "live") body.confirm_symbol = form.confirm_symbol.trim().toUpperCase();
    try {
      await api.createDeployment(body);
      onClose(true);
    } catch (err) {
      setError(err instanceof Error ? err.message : "create failed");
    } finally {
      setSubmitting(false);
    }
  };

  const field = (label: string, key: keyof typeof form, placeholder: string, type = "text") => (
    <label className="block text-sm">
      <span className="text-muted-foreground">{label}</span>
      <input
        type={type}
        value={form[key] as string}
        onChange={(e) => setForm((f) => ({ ...f, [key]: e.target.value }))}
        placeholder={placeholder}
        className="mt-1 w-full rounded-md border bg-transparent px-3 py-2 text-sm"
      />
    </label>
  );

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4">
      <div className="max-h-[90vh] w-full max-w-lg overflow-y-auto rounded-md border bg-background p-5 shadow-lg">
        <h2 className="font-semibold">新增部署</h2>
        <p className="mt-1 text-xs text-muted-foreground">
          頻率繼承該回測的 interval；安全上限為必填、超限的委託會整筆拒絕。
        </p>
        <div className="mt-4 space-y-3">
          {field("回測 Run ID", "run_id", "例如 tsmc_kd_macd_ma")}
          <div className="flex gap-2">
            {(["paper", "live"] as const).map((env) => (
              <button
                key={env}
                type="button"
                onClick={() => setForm((f) => ({ ...f, environment: env }))}
                className={cn(
                  "flex-1 rounded-md border px-3 py-2 text-sm font-medium",
                  form.environment === env && (env === "live" ? "border-danger bg-danger/10 text-danger" : "border-primary bg-primary/10"),
                )}
              >
                {env === "paper" ? "模擬倉（simulation）" : "正式環境（live，需 CA）"}
              </button>
            ))}
          </div>
          {form.environment === "live" ? (
            <div className="rounded-md border border-danger/40 bg-danger/5 p-3 text-xs text-danger">
              正式環境會動用真實資金。需要 shioaji.json 已設定 CA 憑證，且必須在下方輸入標的代號（例如
              TXFR1.TWF）作為打字確認。
            </div>
          ) : null}
          <div className="flex gap-2">
            {(["day", "day_night"] as const).map((s) => (
              <button
                key={s}
                type="button"
                onClick={() => setForm((f) => ({ ...f, sessions: s }))}
                className={cn(
                  "flex-1 rounded-md border px-3 py-2 text-sm",
                  form.sessions === s && "border-primary bg-primary/10",
                )}
              >
                {s === "day" ? "日盤" : "日+夜盤（期貨）"}
              </button>
            ))}
          </div>
          {field("配置資金（TWD）", "allocated_capital", "1000000", "number")}
          {field("單筆最大口數/股數", "max_order_qty", "期貨口數或現股股數", "number")}
          {field("每日最大下單次數", "max_daily_orders", "10", "number")}
          {field("單筆名目金額上限（TWD）", "max_order_notional", "5000000", "number")}
          {form.environment === "live"
            ? field("打字確認：輸入標的代號", "confirm_symbol", "TXFR1.TWF")
            : null}
        </div>
        {error ? <p className="mt-3 text-sm text-danger">{error}</p> : null}
        <div className="mt-4 flex justify-end gap-2">
          <button type="button" onClick={() => onClose(false)} className="rounded-md border px-4 py-2 text-sm">
            取消
          </button>
          <button
            type="button"
            disabled={submitting || !form.run_id.trim()}
            onClick={submit}
            className="rounded-md border border-primary bg-primary/10 px-4 py-2 text-sm font-semibold disabled:opacity-40"
          >
            {submitting ? <Loader2 className="h-4 w-4 animate-spin" /> : "建立"}
          </button>
        </div>
      </div>
    </div>
  );
}
