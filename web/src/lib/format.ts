// Number formatting shared by the charts and stat readouts.

/** 12345 → "12.3k", 1234567 → "1.23M" */
export function formatQps(value: number): string {
  if (value >= 1e6) return `${(value / 1e6).toFixed(2)}M`;
  if (value >= 1e4) return `${(value / 1e3).toFixed(1)}k`;
  if (value >= 1e3) return `${(value / 1e3).toFixed(2)}k`;
  return value.toFixed(0);
}

/** Microseconds, switching to ms or s where that reads better. */
export function formatUs(us: number): string {
  if (us >= 1e6) return `${(us / 1e6).toFixed(2)} s`;
  if (us >= 1e3) return `${(us / 1e3).toFixed(2)} ms`;
  if (us >= 10) return `${us.toFixed(0)} µs`;
  return `${us.toFixed(1)} µs`;
}

/** Log-axis tick labels: 1k, 2k, 5k, 10k, … */
export function formatTick(value: number): string {
  if (value >= 1e6) return `${value / 1e6}M`;
  if (value >= 1e3) return `${value / 1e3}k`;
  return String(value);
}
