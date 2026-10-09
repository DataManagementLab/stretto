/* The breakdown card's arithmetic, as pure functions.
 *
 * Split out of analysis.js so it can be unit-tested under `node --test`, and so the one
 * place a number is divided is a function with a name rather than a closure inside a
 * render call. No DOM, no `state`, relative imports only.
 *
 * The rule the whole module exists to enforce: **normalization never changes what is
 * measured.** Picking "per tuple" divides an already-computed cell and does nothing
 * else - it must not switch the data source, the level, the phase scope, or the
 * stacking. A normalization that changed any of those would make a round trip through the
 * control land on a different view than it started from.
 */

/** Sum, treating non-finite entries as 0 - mirrors state.js's `sum`. */
export function sum(values) {
  return values.reduce((a, b) => a + (Number.isFinite(b) ? b : 0), 0);
}

/**
 * The query-level component that contains a given operator phase.
 *
 * Operator runs tagged `execution` happen inside the `execution` span, `profiling` runs
 * inside `profiling`, and so on - so the honest thing to compare operator time against
 * is the matching component, not end-to-end.
 */
export const PHASE_COMPONENT = {
  execution: "time_execution",
  profiling: "time_profiling",
  configuring: "time_configuring",
  all: "time_end_to_end",
};

/** Total query-level time for a set of query records, in seconds. */
export function phaseTotal(records, phaseMode) {
  const column = PHASE_COMPONENT[phaseMode] ?? "time_end_to_end";
  return sum(records.map((r) => r.phase_components?.[column] ?? 0));
}

/**
 * How far a group's operator total runs past the phase span containing it, or 0.
 *
 * Operator runs happen *inside* their phase span, so operator total <= phase total holds
 * by construction and the "Outside operators" remainder is the difference. The two are
 * timed separately off the same clock, though, so rounding can invert them by
 * microseconds - which is why `remainderFor` clamps at zero.
 *
 * That clamp alone cannot tell rounding noise from a real inconsistency, which would
 * otherwise render silently as a chart with no remainder band.
 *
 * This is the predicate that separates the two cases: rounding stays inside the
 * tolerance and reports 0; anything past it is a real inconsistency for the caller to
 * surface. It reports the raw excess in seconds, not a ratio, so a group with a
 * near-zero phase total cannot manufacture a large one.
 */
export const OVERSHOOT_TOLERANCE = 0.01;

export function remainderOvershoot(phaseSeconds, operatorSeconds) {
  if (!Number.isFinite(phaseSeconds) || !Number.isFinite(operatorSeconds)) return 0;
  const excess = operatorSeconds - phaseSeconds;
  return excess > Math.abs(phaseSeconds) * OVERSHOOT_TOLERANCE ? excess : 0;
}

/* ── Normalization ────────────────────────────────────────────────────────── */

export const NORMALIZATIONS = [
  { value: "total", label: "Total" },
  { value: "per_query", label: "Per query" },
  { value: "per_tuple", label: "Per tuple" },
];

export const DEFAULT_NORMALIZATION = "total";

/**
 * The divisor for one group, or `null` when the normalization is not defined for it.
 *
 * - `total` divides by 1.
 * - `per_query` needs a query-side count. At phase level each record *is* a query row,
 *   so it is that group's record count. At operator level the records are buckets
 *   (config × phase × operator × model × ratio), not queries, so the count has to come
 *   from the matching query group - and when the grouping cannot be expressed at query
 *   level there is no honest answer, so it is `null` rather than a wrong number.
 * - `per_tuple` divides by the tuples the numerator's own rows consumed. Note this is
 *   the sum of per-call batch sizes (`n_input_rows` per `operator_run`), not distinct
 *   query rows: an operator invoked once per batch contributes its batch size each time.
 *
 * `null` propagates to a `null` cell, which renders as "—" in the table and as an
 * explicitly-undefined bar in the chart. A real zero stays a zero.
 */
export function divisorFor(mode, { records = [], queryRecords = null } = {}) {
  if (mode === "total") return 1;
  if (mode === "per_query") {
    if (queryRecords === null) return null;
    return queryRecords.length || null;
  }
  if (mode === "per_tuple") {
    return sum(records.map((r) => r.input_rows)) || null;
  }
  return 1;
}

/**
 * Divide a computed cell by its group's divisor. The single point at which any
 * normalization is applied.
 *
 * Because the divisor is per *group*, not per series, a stacked chart stays additive
 * under normalization: Σₛ(vₛ / T) === (Σₛ vₛ) / T. That is what makes split and
 * no-split agree. A pooled ratio per series (Σ runtimeₛ / Σ input_rowsₛ) would not
 * add up.
 */
export function normalize(value, divisor) {
  if (value === null || value === undefined) return null;
  if (divisor === null || divisor === undefined || divisor === 0) return null;
  return value / divisor;
}

/** Which normalizations make sense for a (measure, level) pair, and why not. */
export function normalizationAvailability(measureValue, level) {
  const out = {};
  for (const { value } of NORMALIZATIONS) out[value] = { enabled: true, reason: "" };

  if (level === "phase") {
    // Query rows carry no tuple count. Borrowing the operator-side sum would divide a
    // phase total by a divisor drawn from a different record set with a different phase
    // scope.
    out.per_tuple = {
      enabled: false,
      reason: "Phase spans do not record a tuple count.",
    };
  }
  if (measureValue === "input_rows") {
    out.per_tuple = { enabled: false, reason: "Tuples per tuple is always 1." };
  }
  return out;
}

/** The axis title for a (measure, normalization) pair - a units change, so it composes. */
export function axisTitle(measureAxis, mode) {
  if (mode === "per_query") return `${measureAxis} / query`;
  if (mode === "per_tuple") return `${measureAxis} / tuple`;
  return measureAxis;
}
