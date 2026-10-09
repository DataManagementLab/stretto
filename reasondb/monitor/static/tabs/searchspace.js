/* Search space tab: what the optimizer was allowed to choose between, per logical step.
 *
 * The Query tab shows the plan that was *picked*. This shows everything that was on the
 * table and was not — which is the only way to tell "the optimizer made a good call"
 * apart from "the good option was never in the list" (e.g. a search space missing its
 * vanilla baselines would hand the gold role to the best *compressed* operator).
 *
 * Gold is highlighted per logical step. It is defined positionally — the profiler takes
 * the last candidate of a step, relying on `UnoptimizedPhysicalPlanStep` having sorted
 * them ascending by quality — so this reports the flag the producer recorded rather than
 * recomputing a maximum here. If the two ever disagree, the tab calls it out instead of
 * hiding it behind its own recomputation.
 */

import { Format, barChart, crColorScale, crTickLabel, sortCrLabels, table } from "/static/charts.js";
import { MISSING_LABEL } from "/static/format.js";
import { chartCard, h, notice, panel, segmented, tiles } from "/static/ui.js";
import {
  cardControl,
  getJson,
  hiddenSet,
  setCardControl,
  state,
  sum,
  toggleSeries,
  uniq,
} from "/static/state.js";
import { facetize, valueLabel, withJobDimensions } from "/static/facets.js";
import { jobSpecIndex } from "/static/analysis.js";

const SPACE_DIMENSIONS = [
  "benchmark",
  "split",
  "executor",
  "precision",
  "recall",
  "run_id",
  "worker_id",
  "job_id",
  "query",
  "logical_type",
  "sample_size",
  "adaptive_sampling",
  "step_idx",
  "state_plan",
  "tune_parameters",
  "reorder",
  "approach",
  "use_indexes",
];

/** The compression variant of one candidate, using the same vocabulary as everywhere
 *  else: vanilla is a mode, not a ratio, and is checked first. */
function crLabelOf(candidate) {
  if (candidate.vanilla) return "vanilla";
  const cr = candidate.effective_compression_ratio;
  return cr === null || cr === undefined ? "n/a" : `cr${cr}`;
}

/**
 * Distinct search spaces, keyed by what the optimizer actually chose between. Two
 * logical steps offering the same candidate list are the same search space, however
 * many queries and configurations produced them — that collapse is the point, since a
 * 70-query sweep otherwise reports 200+ near-identical rows.
 */
function distinctSpaces(steps) {
  const map = new Map();
  for (const step of steps) {
    const ids = (step.candidates || []).map((c) => c.operator);
    const key = `${step.logical_type}|${ids.join(",")}`;
    let entry = map.get(key);
    if (!entry) {
      entry = {
        key,
        logical_type: step.logical_type,
        candidates: step.candidates || [],
        occurrences: 0,
        queries: new Set(),
        configurations: new Set(),
        examples: [],
      };
      map.set(key, entry);
    }
    entry.occurrences += 1;
    if (step.query) entry.queries.add(step.query);
    if (step.run_key) entry.configurations.add(step.run_key);
    if (entry.examples.length < 3 && step.logical_expression) {
      entry.examples.push(step.logical_expression);
    }
  }
  return [...map.values()].sort((a, b) => b.occurrences - a.occurrences);
}

function goldOf(space) {
  return (space.candidates || []).find((c) => c.gold) ?? null;
}

/** A candidate flagged gold that is not the highest-quality one is a real defect. */
function goldDisagreement(space) {
  const gold = goldOf(space);
  const scored = (space.candidates || []).filter((c) => Number.isFinite(c.quality));
  if (!gold || !scored.length) return null;
  const best = scored.reduce((a, b) => (b.quality > a.quality ? b : a));
  return best.operator === gold.operator ? null : { gold, best };
}

function candidateTable(host, space) {
  table(host, {
    id: `space-${space.key}`,
    columns: [
      {
        key: "role",
        label: "",
        format: (v) =>
          v
            ? h("span", { class: `role-tag ${v}`, text: v === "gold" ? "GOLD" : "LLM pick" })
            : document.createTextNode(""),
      },
      { key: "operator", label: "Operator", wrap: true },
      { key: "crLabel", label: "Compression", format: crTickLabel },
      { key: "model_name", label: "Model", wrap: true, format: Format.modelName },
      { key: "quality", label: "Quality", format: (v) => Format.num(v, 2) },
      { key: "fake_cost", label: "Fake cost", format: (v) => Format.num(v, 2) },
    ],
    rows: (space.candidates || []).map((c) => ({
      ...c,
      crLabel: crLabelOf(c),
      role: c.gold ? "gold" : c.estimated_best ? "estimate" : null,
    })),
    // Quality descending is how you read a search space: the ceiling first.
    sortKey: "quality",
    sortDir: "desc",
    rowClass: (row) => (row.role === "gold" ? "gold-row" : ""),
  });
}

function spaceCards(view, spaces) {
  for (const space of spaces) {
    const gold = goldOf(space);
    const disagreement = goldDisagreement(space);
    const body = panel(view, {
      title: `${space.logical_type ?? MISSING_LABEL} — ${space.candidates.length} candidates`,
      sub:
        `${Format.int(space.occurrences)} step(s) across ` +
        `${Format.int(space.queries.size)} quer${space.queries.size === 1 ? "y" : "ies"} and ` +
        `${Format.int(space.configurations.size)} configuration(s).` +
        (gold ? ` Gold: ${gold.operator}` : " No gold recorded."),
    });
    if (disagreement) {
      notice(
        body,
        `Gold is ${disagreement.gold.operator} (quality ` +
          `${Format.num(disagreement.gold.quality, 2)}) but the highest-quality candidate is ` +
          `${disagreement.best.operator} (${Format.num(disagreement.best.quality, 2)}). The ` +
          "profiler takes the last candidate after a quality sort, so these must agree — " +
          "labels for this step are being derived from the wrong operator.",
        "error",
      );
    }
    if (space.examples.length) {
      body.append(
        h("p", { class: "card-sub mono wrap", text: `e.g. ${space.examples[0]}` }),
      );
    }
    const host = h("div", {});
    body.append(host);
    candidateTable(host, space);
  }
}

/** How the search space is distributed over compression ratios, per logical operator. */
function shapeCard(view, spaces, rerender) {
  const id = "space-shape";
  const metric = cardControl(id, "metric", "count");
  const rows = [];
  for (const space of spaces) {
    for (const c of space.candidates || []) {
      rows.push({
        logical_type: space.logical_type ?? MISSING_LABEL,
        crLabel: crLabelOf(c),
        quality: c.quality,
        gold: !!c.gold,
      });
    }
  }
  const cats = uniq(rows.map((r) => r.logical_type)).sort();
  const crLabels = sortCrLabels(uniq(rows.map((r) => r.crLabel)));
  const colors = crColorScale(crLabels);

  chartCard(view, {
    id,
    title: "Search space shape",
    sub:
      metric === "count"
        ? "How many candidates each logical operator can choose between, split by compression ratio."
        : "The quality ceiling each compression ratio contributes per logical operator.",
    wide: true,
    controls: () => [
      segmented(
        [
          { value: "count", label: "Candidates" },
          { value: "quality", label: "Best quality" },
        ],
        metric,
        (v) => setCardControl(id, "metric", v, rerender),
      ),
    ],
    render: (host, height) =>
      barChart(host, {
        height,
        categories: cats,
        stacked: metric === "count",
        hidden: hiddenSet(id),
        series: crLabels.map((cr) => ({
          key: cr,
          label: crTickLabel(cr),
          colorVar: colors[cr],
          values: cats.map((cat) => {
            const matching = rows.filter((r) => r.logical_type === cat && r.crLabel === cr);
            if (!matching.length) return 0;
            if (metric === "count") return matching.length;
            const scored = matching.map((r) => r.quality).filter(Number.isFinite);
            return scored.length ? Math.max(...scored) : 0;
          }),
        })),
        format: metric === "count" ? Format.int : (v) => Format.num(v, 2),
        yTitle: metric === "count" ? "Candidates" : "Best quality",
        emptyMessage: "No search spaces recorded yet.",
        onToggleSeries: (key) => toggleSeries(id, key, rerender),
      }),
    tableSpec: () => ({
      columns: [
        { key: "logical_type", label: "Logical operator" },
        { key: "crLabel", label: "Compression", format: crTickLabel },
        { key: "candidates", label: "Candidates", format: Format.int },
        { key: "best", label: "Best quality", format: (v) => Format.num(v, 2) },
        { key: "goldHere", label: "Holds gold" },
      ],
      rows: cats.flatMap((cat) =>
        crLabels.map((cr) => {
          const matching = rows.filter((r) => r.logical_type === cat && r.crLabel === cr);
          const scored = matching.map((r) => r.quality).filter(Number.isFinite);
          return {
            logical_type: cat,
            crLabel: cr,
            candidates: matching.length,
            best: scored.length ? Math.max(...scored) : null,
            goldHere: matching.some((r) => r.gold) ? "yes" : "no",
          };
        }).filter((r) => r.candidates),
      ),
      sortKey: "candidates",
    }),
  });
}

export function renderSearchSpace(view, rerender) {
  if (state.searchSpace === null) {
    notice(view, "Loading search spaces…");
    return;
  }
  const steps = state.searchSpace || [];
  if (!steps.length) {
    notice(
      view,
      "No search spaces recorded yet. They are captured while a query is being " +
        "configured, so the first one appears once a run has configured its first " +
        "logical step. Runs from before this was recorded show nothing here.",
    );
    return;
  }

  // Most of SPACE_DIMENSIONS is job-spec axes (the sweep state, the profiling budget,
  // the optimizer's own knobs) that live only on the job spec, never on a search-space
  // record -- and `deriveDimensions` skips any name absent from the records, so the
  // spec must be joined first.
  const joined = withJobDimensions(steps, jobSpecIndex());
  const facets = facetize(view, {
    scope: "s.",
    records: joined,
    candidates: SPACE_DIMENSIONS,
    rerender,
    note: "Every recorded step shares one configuration, so there is nothing to filter by.",
  });
  const spaces = distinctSpaces(facets.filtered);
  const broken = spaces.filter(goldDisagreement).length;

  tiles(view, [
    { label: "Logical steps", value: Format.int(facets.filtered.length) },
    { label: "Distinct search spaces", value: Format.int(spaces.length) },
    {
      label: "Candidates / step",
      value: spaces.length
        ? Format.num(sum(spaces.map((s) => s.candidates.length)) / spaces.length, 1)
        : "—",
    },
    {
      label: "Gold operators",
      value: Format.int(uniq(spaces.map((s) => goldOf(s)?.operator)).length),
      note: broken ? `${broken} disagree with best quality` : null,
    },
  ]);

  shapeCard(view, spaces, rerender);
  spaceCards(view, spaces);
}

export async function loadSearchSpace(rerender) {
  try {
    const payload = await getJson("/api/search-space");
    state.searchSpace = payload.steps || [];
  } catch {
    state.searchSpace = [];
  }
  if (state.route.tab === "searchspace") rerender();
}
