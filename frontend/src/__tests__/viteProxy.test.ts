import fs from "node:fs";
import path from "node:path";
import { describe, expect, it } from "vitest";

describe("Vite API proxy config", () => {
  const configPath = path.resolve(__dirname, "../../vite.config.ts");
  const config = fs.readFileSync(configPath, "utf8");

  it("proxies channel runtime endpoints", () => {
    expect(config).toContain('"/channels"');
  });

  it("proxies settings endpoints", () => {
    expect(config).toContain('"/settings/llm"');
    expect(config).toContain('"/settings/data-sources"');
  });

  it("does not rewrite the Host header to the Docker-internal proxy target", () => {
    // changeOrigin: true would make api_server.py's cross-site-POST guard
    // (Origin vs Host comparison) reject every non-loopback browser's writes
    // with 403 "Cross-site request denied" -- found live 2026-07-08 testing
    // the Deployments start button over Tailscale.
    expect(config).toContain("changeOrigin: false");
    expect(config).not.toContain("changeOrigin: true");
  });

  it("proxies authentication endpoints", () => {
    expect(config).toContain('"/auth"');
  });
});
