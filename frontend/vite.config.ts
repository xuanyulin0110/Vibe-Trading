import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";
import path from "path";

const PROXY_PATHS = [
  "/auth",
  "/sessions",
  "/swarm/presets",
  "/swarm/runs",
  "/qveris",
  "/settings/llm",
  "/settings/data-sources",
  "/channels",
  "/mandate",
  "/live",
  "/upload",
  "/shadow-reports",
  "/deployments",
];

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), "");
  const apiTarget = env.VITE_API_URL || "http://127.0.0.1:8899";
  // The backend only trusts loopback peers unless API_AUTH_KEY is set (see
  // api_server.py's _validate_api_auth): this dev-mode proxy runs in its own
  // container, so its requests to the backend arrive over the Docker bridge
  // network, not loopback, and get rejected once a key is configured. Inject
  // the key here so the browser never needs to know it -- the key only ever
  // lives in server-side env (agent/.env), never in client-visible code.
  const apiAuthKey = process.env.API_AUTH_KEY || "";
  const withAuthHeader = <T extends Record<string, unknown>>(proxyOpts: T): T => ({
    ...proxyOpts,
    configure(proxy: { on: (event: string, cb: (proxyReq: { setHeader: (name: string, value: string) => void }) => void) => void }) {
      if (!apiAuthKey) return;
      proxy.on("proxyReq", (proxyReq) => {
        proxyReq.setHeader("Authorization", `Bearer ${apiAuthKey}`);
      });
    },
  });
  // changeOrigin false (the default) -- the backend's cross-site-POST guard
  // (api_server.py's _reject_cross_site_browser_request) checks the browser's
  // Origin header against the request's Host header to tell a same-site fetch
  // from a cross-site one. Rewriting Host would point it at this proxy's
  // Docker-internal target ("vibe-trading:8899"), which can never equal a
  // real browser Origin (e.g. "100.78.149.102:5899") for ANY address --
  // confirmed live 2026-07-08: every POST/PUT/DELETE from a LAN/Tailscale/
  // WireGuard browser 403'd with "Cross-site request denied" (surfaced to the
  // user as a generic "Add API key in Settings" message, since the frontend
  // maps every 401/403 to that string regardless of the real backend detail).
  // Only loopback-origin browsers were ever exempt, via a separate bypass.
  // This backend has no TrustedHostMiddleware/virtual-hosting that needs Host
  // rewritten to match the proxy target, so leaving it as the browser sent it
  // is safe and is what makes the Origin/Host comparison meaningful again.
  const apiProxy = withAuthHeader({ target: apiTarget, changeOrigin: false });
  const apiProxyWithHtmlFallback = {
    ...apiProxy,
    bypass(req: { headers: { accept?: string } }) {
      if (req.headers.accept?.includes("text/html")) {
        return "/index.html";
      }
    },
  };

  return {
    plugins: [react()],
    resolve: {
      alias: { "@": path.resolve(__dirname, "./src") },
    },
    server: {
      port: 5899,
      proxy: {
        ...Object.fromEntries(PROXY_PATHS.map((p) => [p, apiProxy])),
        // SPA RunDetail page — only the two-segment ``/runs/{id}``
        // form should fall back to ``index.html`` on browser navigation.
        // ``/runs/{id}/code`` and ``/runs/{id}/pine`` are API-only and
        // must keep proxying to the backend even when Accept is text/html.
        "^/runs/[^/]+/?$": apiProxyWithHtmlFallback,
        "/runs": apiProxy,
        "/correlation": apiProxyWithHtmlFallback,
        "^/alpha(?:/|$)": apiProxy,
      },
    },
    build: {
      rollupOptions: {
        output: {
          manualChunks: {
            "vendor-react": ["react", "react-dom", "react-router-dom"],
            "vendor-charts": ["echarts"],
          },
        },
      },
    },
  };
});
