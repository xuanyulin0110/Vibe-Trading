export type ChartRange = "1M" | "3M" | "6M" | "1Y" | "ALL";

// Calendar days, not a bar count -- a fixed bar count ("1M" = last 22 bars)
// silently assumed one bar per trading day. For an intraday-interval run
// (5m/15m/30m/1h/4h) that showed "1M" as the last 22 bars = under two
// hours of data instead of a month, with no way to reach a real month of
// history via these buttons at all. Computing the zoom window from the
// bars' own timestamps instead works for any interval automatically.
export const RANGE_DAYS: Record<ChartRange, number> = { "1M": 30, "3M": 90, "6M": 180, "1Y": 365, ALL: Infinity };

const MS_PER_DAY = 86400000;

/**
 * Parse a bar's "time" string into epoch milliseconds.
 *
 * Bar times are "YYYY-MM-DD" or "YYYY-MM-DD HH:MM:SS" (space, not "T" --
 * see agent/src/ui_services.py's _format_bar_timestamp). Date's
 * space-separated form isn't reliably parsed across browsers; normalize to
 * ISO 8601 first.
 */
export function parseBarTime(value: string): number {
  return new Date(value.includes(" ") ? value.replace(" ", "T") : value).getTime();
}

/**
 * Compute the initial dataZoom start percentage for a range button, from
 * the bars' own timestamps rather than an assumed bars-per-day count.
 *
 * @param times Bar "time" strings in chronological order.
 * @param range Selected range button.
 * @returns A 0-100 percentage suitable for ECharts dataZoom's `start`.
 */
export function computeRangeStart(times: string[], range: ChartRange): number {
  const days = RANGE_DAYS[range];
  if (days === Infinity || times.length <= 1) return 0;
  const lastMs = parseBarTime(times[times.length - 1]);
  const cutoffMs = lastMs - days * MS_PER_DAY;
  const startIdx = times.findIndex((t) => parseBarTime(t) >= cutoffMs);
  return startIdx > 0 ? (startIdx / times.length) * 100 : 0;
}
