/* Value formatting, shared by every chart, table and tile.
 *
 * Split out of charts.js so it can be imported by modules that must not pull in the SVG
 * layer - and so the pure formatting rules can be unit-tested under `node --test`
 * without a DOM. Deliberately dependency-free: relative imports only, no `state`, no
 * `document`.
 */

/**
 * The one string shown for a value that was never recorded.
 *
 * Distinct from "—" (an absent *number*) and from "N/A" (a value that is recorded and
 * genuinely does not apply, e.g. an operator with no KV model). Three different
 * statements, three different strings.
 */
export const MISSING_LABEL = "(not recorded)";

/** The placeholder a record carries when a dimension does not apply to it. */
export const NOT_APPLICABLE = "n/a";

export const Format = {
  seconds(v) {
    if (v === null || v === undefined || Number.isNaN(v)) return "—";
    if (v < 1) return `${(v * 1000).toFixed(0)} ms`;
    if (v < 90) return `${v.toFixed(2)} s`;
    if (v < 5400) return `${(v / 60).toFixed(1)} min`;
    return `${(v / 3600).toFixed(2)} h`;
  },
  hours(v) {
    return v === null || v === undefined || Number.isNaN(v) ? "—" : `${v.toFixed(2)} h`;
  },
  gb(v) {
    return v === null || v === undefined || Number.isNaN(v) ? "—" : `${v.toFixed(1)} GB`;
  },
  num(v, digits = 2) {
    if (v === null || v === undefined || Number.isNaN(v)) return "—";
    if (Number.isInteger(v)) return v.toLocaleString();
    if (Math.abs(v) >= 1000) return v.toLocaleString(undefined, { maximumFractionDigits: 0 });
    return v.toFixed(digits);
  },
  int(v) {
    return v === null || v === undefined || Number.isNaN(v) ? "—" : Math.round(v).toLocaleString();
  },
  ratio(v) {
    return v === null || v === undefined || Number.isNaN(v) ? "—" : `${v.toFixed(2)}×`;
  },
  clock(ts) {
    return new Date(ts * 1000).toLocaleTimeString();
  },
  bytes(v) {
    if (v === null || v === undefined || Number.isNaN(v)) return "—";
    const units = ["B", "KB", "MB", "GB", "TB"];
    let i = 0;
    let n = v;
    while (n >= 1024 && i < units.length - 1) {
      n /= 1024;
      i += 1;
    }
    return `${n.toFixed(n < 10 && i > 0 ? 1 : 0)} ${units[i]}`;
  },
  /**
   * A model id shown as its last path segment: "meta-llama/Llama-3.1-70B" -> "Llama-3.1-70B".
   *
   * The "n/a" sentinel an operator with no KV backend carries (TraditionalFilter,
   * PythonExtract - see collector.cr_label_for) is mapped to "N/A" rather than split
   * on "/" into a bare "a".
   */
  modelName(v) {
    if (v === null || v === undefined || v === "") return MISSING_LABEL;
    if (v === NOT_APPLICABLE) return "N/A";
    return String(v).split("/").pop();
  },
  /**
   * An operator id shown as its family: "TextQaFilter-KvTextQABackend-..." ->
   * "TextQaFilter". Same shape of hazard as `modelName`, on "-" instead of "/".
   */
  operatorShortName(v) {
    if (v === null || v === undefined || v === "") return MISSING_LABEL;
    return String(v).split("-")[0];
  },
  truncate(s, n = 60) {
    const str = String(s ?? "");
    return str.length <= n ? str : `${str.slice(0, n - 1)}…`;
  },
  /** Unix seconds -> "3s ago" / "2m ago" / "1h ago" - for last-heartbeat/registered-at
   * columns (Workers tab), where "how stale is this" matters more than the wall clock. */
  relativeTime(ts) {
    if (ts === null || ts === undefined || Number.isNaN(ts)) return "—";
    const deltaS = Date.now() / 1000 - ts;
    if (deltaS < 0) return "just now";
    if (deltaS < 60) return `${Math.round(deltaS)}s ago`;
    if (deltaS < 3600) return `${Math.round(deltaS / 60)}m ago`;
    if (deltaS < 86400) return `${Math.round(deltaS / 3600)}h ago`;
    return `${Math.round(deltaS / 86400)}d ago`;
  },
};
