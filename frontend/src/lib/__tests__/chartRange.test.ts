import { parseBarTime, computeRangeStart, RANGE_DAYS } from "../chartRange";

describe("parseBarTime", () => {
  it("parses a date-only bar time", () => {
    expect(parseBarTime("2026-04-01")).toBe(new Date("2026-04-01").getTime());
  });

  it("parses a space-separated intraday bar time as if it were ISO 8601", () => {
    expect(parseBarTime("2026-04-01 09:05:00")).toBe(new Date("2026-04-01T09:05:00").getTime());
  });
});

describe("computeRangeStart", () => {
  it("returns 0 for ALL regardless of data", () => {
    expect(computeRangeStart(["2026-04-01", "2026-05-01"], "ALL")).toBe(0);
  });

  it("returns 0 when there is 0 or 1 bar", () => {
    expect(computeRangeStart([], "1M")).toBe(0);
    expect(computeRangeStart(["2026-04-01"], "1M")).toBe(0);
  });

  it("returns 0 when the whole series already fits within the range", () => {
    const times = ["2026-04-01", "2026-04-05", "2026-04-10"];
    expect(computeRangeStart(times, "1Y")).toBe(0);
  });

  it("scopes a daily-bar series to the last N calendar days (existing behavior preserved)", () => {
    // 1 bar per day for 100 days -- "1M" (30 days) should start near bar 70.
    const times = Array.from({ length: 100 }, (_, i) => {
      const d = new Date(Date.UTC(2026, 0, 1) + i * 86400000);
      return d.toISOString().slice(0, 10);
    });
    const start = computeRangeStart(times, "1M");
    // Bars 0..69 are older than 30 days before the last bar; expect the
    // zoom window to begin somewhere around the 70th bar (index/length*100).
    expect(start).toBeGreaterThan(60);
    expect(start).toBeLessThan(75);
  });

  it("scopes a 5-minute-bar series to the last N calendar days, not a fixed bar count", () => {
    // The bug this fixes: a fixed 22-bar "1M" window on 5m data covered
    // under two hours. 3 days of 5m bars (78 bars/day * 3 = 234 bars) with
    // range "1M" (30 days) must show ALL of them, since none are older than
    // 30 days -- a bars-per-day heuristic tuned for daily bars would instead
    // start the zoom partway through day 3.
    const times: string[] = [];
    for (let day = 0; day < 3; day++) {
      for (let bar = 0; bar < 78; bar++) {
        const minutes = bar * 5;
        const h = String(Math.floor(minutes / 60)).padStart(2, "0");
        const m = String(minutes % 60).padStart(2, "0");
        times.push(`2026-04-0${day + 1} ${h}:${m}:00`);
      }
    }
    expect(computeRangeStart(times, "1M")).toBe(0);
  });

  it("still narrows a long intraday series when it genuinely spans more than the range", () => {
    // 60 days of 1 bar/day worth of intraday timestamps standing in for a
    // longer history -- "1M" must exclude the older bars.
    const times = Array.from({ length: 60 }, (_, i) => {
      const d = new Date(Date.UTC(2026, 0, 1) + i * 86400000);
      return `${d.toISOString().slice(0, 10)} 09:00:00`;
    });
    const start = computeRangeStart(times, "1M");
    expect(start).toBeGreaterThan(0);
    expect(start).toBeLessThan(100);
  });
});

describe("RANGE_DAYS", () => {
  it("is calendar days, not a trading-day/bar count", () => {
    expect(RANGE_DAYS["1M"]).toBe(30);
    expect(RANGE_DAYS["3M"]).toBe(90);
    expect(RANGE_DAYS["6M"]).toBe(180);
    expect(RANGE_DAYS["1Y"]).toBe(365);
    expect(RANGE_DAYS.ALL).toBe(Infinity);
  });
});
