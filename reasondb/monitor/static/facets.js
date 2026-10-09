/* Group-by and filter over any list of records, driven from the URL hash.
 *
 * Every analysis chart in the dashboard answers the same shape of question: "split this
 * measurement by whatever the sweep varied, optionally restricted to some slice of it".
 * The dimensions differ per run - one task varies sample size, another varies storage
 * step, a third only varies the guarantee - so they are *derived* from the data rather
 * than hard-coded, and a dimension that takes a single value is hidden entirely (there
 * is nothing to compare, and offering it just adds noise).
 *
 * Records arrive already stamped with their configuration by the collector (see
 * CONFIG_DIMENSIONS in reasondb/monitor/collector.py); a coordinator job's `spec`
 * contributes the rest - sample_size, step_idx, press_name, use_indexes, cost_type -
 * simply by being flat scalars. A new producer flag still has to be named in the calling
 * tab's candidate list, though: `deriveDimensions` only considers the names it is given.
 * The Run tab's configuration panel is the exception - it enumerates the spec itself.
 *
 * State lives in the hash: `?group=executor,sample_size&f.benchmark=artwork|movie`.
 */

import { checkList, details, field, h, segmented } from "/static/ui.js";
import { Format } from "/static/charts.js";
import { setParams, state } from "/static/state.js";
// The pure half lives in dimensions.js so it is testable without a DOM. Re-exported
// below because the rest of the dashboard imports these names from facets.js.
import {
  DEFAULT_ROLE_MODE,
  DEFAULT_RUN_MODE,
  NOT_APPLICABLE,
  ROLE_MODES,
  RUN_MODES,
  applyFilters,
  applyRoleMode,
  applyRunMode,
  earlierRunCount,
  hiddenByRunMode,
  compositeKey,
  compositeLabel,
  dimensionKey,
  MISSING_KEY,
  MISSING_LABEL,
  compareValues,
  deriveDimensions,
  dimensionLabel,
  groupRecords,
  presentDimensions,
  hiddenByRoleMode,
  specDimensions,
  valueLabel,
  withJobDimensions,
} from "./dimensions.js";

export {
  DEFAULT_ROLE_MODE,
  DEFAULT_RUN_MODE,
  NOT_APPLICABLE,
  ROLE_MODES,
  RUN_MODES,
  applyFilters,
  applyRoleMode,
  applyRunMode,
  earlierRunCount,
  hiddenByRunMode,
  compositeKey,
  compositeLabel,
  dimensionKey,
  MISSING_KEY,
  MISSING_LABEL,
  compareValues,
  deriveDimensions,
  dimensionLabel,
  groupRecords,
  presentDimensions,
  hiddenByRoleMode,
  specDimensions,
  valueLabel,
  withJobDimensions,
};

/** Above this many distinct values, a filter collapses instead of listing every chip. */
const INLINE_VALUE_LIMIT = 12;


/* ── Selection, stored in the URL hash ─────────────────────────────────────── */

/** Which label-pass mode this facet bar is in. Defaults to excluding them. */
export function readRoleMode(scope) {
  const raw = state.route.params.get(`${scope}pass`);
  return ROLE_MODES.some((m) => m.value === raw) ? raw : DEFAULT_ROLE_MODE;
}

function writeRoleMode(scope, mode) {
  setParams({ [`${scope}pass`]: mode === DEFAULT_ROLE_MODE ? null : mode });
}

/** Which run mode this facet bar is in. Defaults to every run of this task. */
export function readRunMode(scope) {
  const raw = state.route.params.get(`${scope}run`);
  return RUN_MODES.some((m) => m.value === raw) ? raw : DEFAULT_RUN_MODE;
}

function writeRunMode(scope, mode) {
  setParams({ [`${scope}run`]: mode === DEFAULT_RUN_MODE ? null : mode });
}

/** The run this process is collecting for, or null for a viewer with no run of its own. */
export function liveRunId() {
  return state.run?.live_run_id ?? null;
}

/** Read the group-by / filter selection for one facet bar out of the current route. */
export function readSelection(scope, dimensions) {
  const known = new Set(dimensions.map((d) => d.name));
  const groupParam = state.route.params.get(`${scope}group`) || "";
  const group = groupParam
    .split(",")
    .map((s) => s.trim())
    .filter((s) => s && known.has(s));
  const filters = new Map();
  for (const [key, raw] of state.route.params.entries()) {
    if (!key.startsWith(`${scope}f.`)) continue;
    const name = key.slice(`${scope}f.`.length);
    if (!known.has(name)) continue;
    filters.set(name, new Set(raw.split("|").filter(Boolean)));
  }
  return { group, filters };
}

function writeSelection(scope, selection, dimensions) {
  const entries = { [`${scope}group`]: selection.group.join(",") || null };
  for (const dim of dimensions) {
    const chosen = selection.filters.get(dim.name);
    entries[`${scope}f.${dim.name}`] = chosen && chosen.size ? [...chosen].join("|") : null;
  }
  setParams(entries);
}


/**
 * The label-pass control, plus the count of what it is hiding.
 *
 * Rendered only when the records actually contain a labelling pass, so a plain
 * single-process run never grows a control for something that never happened. The count
 * is not optional: excluding data by default is only honest if the amount excluded is
 * on screen.
 */
function roleControl(wrap, { scope, roleMode, roleHidden, rerender }) {
  if (roleHidden === null || roleHidden === undefined) return;
  wrap.append(
    field(
      "Pass",
      segmented(ROLE_MODES, roleMode, (mode) => {
        writeRoleMode(scope, mode);
        rerender();
      }),
    ),
  );
  if (roleHidden > 0) {
    wrap.append(
      h("span", {
        class: "facet-summary on",
        text: `${roleHidden.toLocaleString()} label-pass record(s) hidden`,
        title:
          "A labelling pass runs every query to produce the ground truth this sweep is " +
          "scored against. It is not a measured point, so it is excluded from these " +
          "statistics by default.",
      }),
    );
  }
}

/**
 * The run scope control, plus how much of what is on screen came from earlier runs.
 *
 * Rendered only when the dashboard is actually holding more than one run - a coordinator
 * started fresh in an empty output directory never grows a control for history it does
 * not have. The count is not optional, for the same reason the label-pass count is not,
 * but it runs the other way: this default *includes* data the live process did not
 * produce, and pooling is only honest if the amount pooled in is on screen.
 */
function runControl(wrap, { scope, runMode, runHidden, runCount, rerender }) {
  if (!runHidden) return;
  wrap.append(
    field(
      "Runs",
      segmented(RUN_MODES, runMode, (mode) => {
        writeRunMode(scope, mode);
        rerender();
      }),
    ),
  );
  // Stated in both directions. Included data has to be as visible as excluded data:
  // the default pools earlier runs into these charts, and a reader who does not know
  // that would read a restarted task's totals as this process's own work.
  const runs = `${runCount.toLocaleString()} earlier run${runCount === 1 ? "" : "s"}`;
  wrap.append(
    h("span", {
      class: "facet-summary on",
      text:
        runMode === "all"
          ? `includes ${runHidden.toLocaleString()} record(s) from ${runs}`
          : `${runHidden.toLocaleString()} record(s) from ${runs} hidden`,
      title:
        "This dashboard was seeded with the telemetry of the earlier runs in the same " +
        "output directory, which is one coordinator task - so these are earlier attempts " +
        "at the same experiment, and they are included by default. Switch to \"This run\" " +
        "to narrow to the live process, or group by Run to compare them.",
    }),
  );
}

/**
 * The control bar: "Group by" chips, then one filter row per varying dimension.
 * `scope` namespaces the hash keys so two bars on one tab do not collide.
 */
export function facetBar(
  parent,
  {
    scope,
    dimensions,
    selection,
    rerender,
    note,
    roleMode,
    roleHidden,
    runMode,
    runHidden,
    runCount,
  },
) {
  const wrap = h("div", { class: "facets" });
  // Before the "nothing varies" early return: a run can have nothing to group by and
  // still be hiding a labelling pass or an earlier run, and those counts must never go
  // unshown.
  runControl(wrap, { scope, runMode, runHidden, runCount, rerender });
  roleControl(wrap, { scope, roleMode, roleHidden, rerender });
  if (!dimensions.length) {
    wrap.append(
      h("p", {
        class: "card-sub",
        text:
          note ??
          "Nothing varies across the runs recorded so far, so there is nothing to group or filter by yet.",
      }),
    );
    parent.append(wrap);
    return wrap;
  }

  const groupable = dimensions.filter((d) => d.groupable);
  const chosenGroup = new Set(selection.group);
  wrap.append(
    field(
      "Group by",
      checkList(
        groupable.map((d) => ({ value: d.name, label: d.label })),
        chosenGroup,
        (name) => {
          const next = selection.group.includes(name)
            ? selection.group.filter((g) => g !== name)
            : [...selection.group, name];
          writeSelection(scope, { ...selection, group: next }, dimensions);
          rerender();
        },
      ),
    ),
  );

  const filterRow = h("div", { class: "facet-filters" });
  for (const dim of dimensions) {
    const chosen = selection.filters.get(dim.name) ?? new Set();
    const toggle = (value) => {
      const next = new Map(selection.filters);
      const set = new Set(next.get(dim.name) ?? []);
      if (set.has(value)) set.delete(value);
      else set.add(value);
      next.set(dim.name, set);
      writeSelection(scope, { ...selection, filters: next }, dimensions);
      rerender();
    };
    const options = dim.values.map((v) => ({
      value: String(v),
      label: valueLabel(dim.name, v),
      title: String(v),
    }));

    // A dimension with a handful of values shows its chips outright. One with dozens -
    // `query` on a 70-query benchmark - would otherwise bury the whole control surface
    // under a wall of chips, so it collapses behind a summary that says how many are
    // selected. Open state persists across re-renders like every other control.
    if (options.length <= INLINE_VALUE_LIMIT) {
      filterRow.append(field(dim.label, checkList(options, chosen, toggle)));
      continue;
    }
    const summary = chosen.size ? `${dim.label}: ${chosen.size} selected` : `${dim.label}: all (${options.length})`;
    filterRow.append(
      details(
        `${scope}${dim.name}`,
        h("span", { class: chosen.size ? "facet-summary on" : "facet-summary", text: summary }),
        h("div", { class: "facet-values" }, checkList(options, chosen, toggle)),
      ),
    );
  }
  // Group-by stays a single visible row - it is the control you reach for constantly.
  // Filters collapse behind a summary: with both record kinds feeding one panel there
  // can be a dozen dimensions, and a wall of chips would push the charts below the fold
  // for the common case where nothing is filtered at all. The summary always says how
  // many filters are active, so a collapsed panel can never hide that it is filtering.
  const activeCounts = [...selection.filters.entries()].filter(([, s]) => s.size);
  const filterSummary = activeCounts.length
    ? `Filters: ${activeCounts.map(([name, s]) => `${dimensionLabel(name)} (${s.size})`).join(", ")}`
    : `Filters: none — ${dimensions.length} available`;
  wrap.append(
    details(
      `${scope}filters`,
      h("span", {
        class: activeCounts.length ? "facet-summary on" : "facet-summary",
        text: filterSummary,
      }),
      filterRow,
    ),
  );

  if (activeCounts.length || selection.group.length) {
    const clear = h("button", { type: "button", class: "link-button", text: "Reset" });
    clear.addEventListener("click", () => {
      writeSelection(scope, { group: [], filters: new Map() }, dimensions);
      rerender();
    });
    wrap.append(clear);
  }
  parent.append(wrap);
  return wrap;
}

/** Convenience: derive dimensions, read the selection, draw the bar, return both. */
export function facetize(parent, { scope, records, candidates, rerender, note }) {
  // Run scope comes first, for the same reason the pass filter does: an excluded record
  // must not contribute group-by chips either. Under "This run" the `run_id` dimension
  // collapses to a single value and self-hides, so the Run chip appears exactly when
  // there is more than one run to compare.
  const runMode = readRunMode(scope);
  const live = liveRunId();
  // Always measured against "current", whatever the mode: this is "how much of this is
  // not the live run", the number the control reports either as included or as hidden.
  const runHidden = hiddenByRunMode(records, "current", live);
  const runCount = earlierRunCount(records, live);
  const inRun = applyRunMode(records, runMode, live);
  // The pass filter second: a label pass's guarantee-less executor_start would otherwise
  // add a dead "Precision target: —" chip to a run that never varied the guarantee.
  const roleMode = readRoleMode(scope);
  const hasLabelRecords = inRun.some((r) => r.role === "label");
  const scoped = applyRoleMode(inRun, roleMode);
  const dimensions = deriveDimensions(scoped, candidates);
  const selection = readSelection(scope, dimensions);
  facetBar(parent, {
    scope,
    dimensions,
    selection,
    rerender,
    note,
    roleMode,
    roleHidden: hasLabelRecords ? hiddenByRoleMode(inRun, roleMode) : null,
    runMode,
    runHidden,
    runCount,
  });
  const filtered = applyFilters(scoped, selection);
  return { dimensions, selection, filtered, groups: groupRecords(filtered, selection.group) };
}
