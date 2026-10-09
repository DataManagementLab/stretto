"""Render every experiment in a cluster config as an HTML report: what each one is for,
and which axes it holds fixed versus sweeps.

The report combines ``scripts/cluster.yaml``, the producer each experiment names,
``experiments.py`` and the engine's ``resolve_*`` defaults into one view.

Nothing in the output is hand-written, so it cannot drift from the producers:

- the experiment list, its per-experiment flags and its node assignment come from
  ``reasondb.coordinator.cluster``, the same parser both launcher scripts read;
- each experiment's **purpose** is its producer module's own ``__doc__``;
- each experiment's **axis table** comes from really calling that producer's
  ``enumerate_jobs`` on the flags the YAML gives it, and reading the resulting job specs
  through ``reasondb.coordinator.axes`` - the same rule the Run tab's configuration
  panel applies in the browser.

One thing it cannot know, called out in the output rather than guessed at: **how many
sweep states a benchmark has**. ``prepare_sweep`` reads the materialized KV caches off
disk, and a machine writing a report need not have them, so the planners run against the
real models at their real ratio grid with every level assumed present, text modality only
(see ``_stubbed_disk_state``).

Every other axis is exactly what the coordinator would enumerate.

    python scripts/generate_experiment_report.py --output experiments.html
    python scripts/generate_experiment_report.py --config scripts/my-sweep.yaml
"""

import argparse
import argparse as _argparse
import html
import importlib
import logging
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Resolving a state's operator set builds a configurator, and `build_toolbox` constructs
# `GPT4o()` for PythonExtract/PythonTransform. `OpenAILLM.__init__` reads
# `os.environ["OPENAI_API_KEY"]` with a bare subscript, so an unset key is a KeyError --
# on a script that only prints a table and never calls the API (that happens in
# `prepare()`, which nothing here reaches). A placeholder keeps a report runnable on a
# laptop; a real key, if present, is left alone and still never used.
os.environ.setdefault("OPENAI_API_KEY", "unused-by-the-report-generator")

from reasondb.coordinator.axes import derive_run_config  # noqa: E402
from reasondb.coordinator.cli import DEFAULT_BENCHMARKS, build_parser  # noqa: E402
from reasondb.coordinator.cluster import DEFAULT_CONFIG_PATH, load_cluster_config  # noqa: E402
from reasondb.coordinator.producers import get_producer  # noqa: E402
from reasondb.coordinator.producers import (  # noqa: E402
    parameter_sweep as sweep_producer,
    run_benchmark as run_benchmark_producer,
)
from reasondb.evaluation import parameter_sweep as psweep  # noqa: E402
from reasondb.evaluation.benchmark_registry import ALL_BENCHMARKS  # noqa: E402
from reasondb.interface.default_operator_toolbox import (  # noqa: E402
    TEXT_MODEL_8B,
    TEXT_MODEL_70B,
)
from reasondb.utils.benchmark_args import resolve_precompute_simulate  # noqa: E402

logger = logging.getLogger(__name__)

@contextmanager
def _stubbed_disk_state():
    """Let ``enumerate_jobs`` run on a machine with none of a node's working state.

    ``prepare_sweep`` measures cache directories with ``du`` to build the storage table
    the state planners walk; without caches it returns ``None`` and every producer
    enumerates nothing at all. The substitute is as close to the real thing as it can be
    without a disk: the *real* text slots, at the *real* ratio grid their spec-table
    entries define, so the state planners run their real logic - ``plan_default_state``
    finds the baselines the default suite names, and the walk to gold reaches gold in the
    number of steps it really would.

    Phase 0 is stubbed for the same reason. ``enumerate_filter_stats_jobs`` decides
    whether a benchmark still needs its predicate-overlap matrix by reading the
    --simulate store, which may be absent or incomplete on the machine writing the
    report. A phase-0 job is a per-benchmark chore that ``describe`` drops from the cross
    anyway, so these jobs are assumed done (complementing
    ``simulate_files_must_exist=False`` in ``resolve_precompute_simulate``).

    Three things are therefore fiction, and only these: every level is assumed
    materialized (the maximal case, which is what the walk starts from anyway), only the
    text modality is modelled - so a two-modality benchmark like ecommerce really has more
    states than shown - and every dataset is assumed to have its filter stats and its
    pinned query set already. Every other axis is exact, because the rest of the cross is
    decided by the producer and the flags rather than by what is on disk.
    """
    real = psweep.prepare_sweep
    slots = [
        psweep.ModelSlot("text_small", "text", TEXT_MODEL_8B, large=False),
        psweep.ModelSlot("text_large", "text", TEXT_MODEL_70B, large=True),
    ]
    levels = {slot.key: psweep.slot_effective_ratios(slot) for slot in slots}
    # Descending bytes with the ratio, as a real storage table is: it decides which slot
    # the greedy walk prunes next, hence the order of the states it visits.
    table = {
        slot.key: {cr: 1000 - 100 * i for i, cr in enumerate(levels[slot.key])}
        for slot in slots
    }

    # One cached item per 10 bytes, so the entry table is proportional to the byte one
    # and a per-item figure read off this stub stays a constant. It is never reported
    # here; it exists so the shape matches what the worker gets.
    entries = {
        key: {cr: size // 10 for cr, size in levels_.items()}
        for key, levels_ in table.items()
    }

    def stub(benchmark, args):
        plan = psweep.STATE_PLANS[psweep.resolve_state_plan(args)]
        states = plan(slots, levels, table, args.use_indexes)
        return psweep.SweepPrep(
            states, table, entries, slots, {slot.key: slot for slot in slots}, levels
        )

    def no_phase_zero(task_id, output_root, args, producer_name):
        return []

    # Both producers bind the function by name at import, so the substitute has to be
    # installed on each of them rather than on the module that defines it.
    phase_zero_owners = [sweep_producer, run_benchmark_producer]
    real_phase_zero = [m.enumerate_filter_stats_jobs for m in phase_zero_owners]

    psweep.prepare_sweep = stub
    for module in phase_zero_owners:
        module.enumerate_filter_stats_jobs = no_phase_zero
    try:
        yield
    finally:
        psweep.prepare_sweep = real
        for module, original in zip(phase_zero_owners, real_phase_zero):
            module.enumerate_filter_stats_jobs = original


def _args_for(experiment) -> _argparse.Namespace:
    """The parsed namespace the coordinator would build for this experiment.

    Uses the coordinator's own parser (``reasondb.coordinator.cli``), so the report can
    only describe flags the coordinator really accepts, and a new flag reaches both at
    once. A token a config writes as ``$VAR`` is dropped: it is expanded on the node, and
    this machine cannot say into what.

    ``simulate_files_must_exist=False`` because a report is written wherever its author
    is sitting, while the precompute stores live on the nodes that recorded them. Every
    other check that call makes still runs - an unknown benchmark, a duplicated key, a
    mapping that disagrees with ``--benchmarks`` - and those are the ones that would
    otherwise sink a launch.
    """
    argv = [token for token in experiment.args if not token.startswith("$")]
    args = build_parser().parse_args(
        ["--task-id", experiment.task_id, "--producer", experiment.producer, *argv]
    )
    resolve_precompute_simulate(
        args, ALL_BENCHMARKS, DEFAULT_BENCHMARKS, simulate_files_must_exist=False
    )
    if args.output_dir is None:
        args.output_dir = Path("benchmark_results") / experiment.task_id
    return args


def _deferred_tokens(experiment) -> List[str]:
    """Shell variables in this experiment's argv, which are expanded on the node.

    The shipped config uses none, but the config format permits a ``$VAR``; the report
    flags such values rather than silently describing the parser's defaults as the run.
    """
    return [token for token in experiment.args if token.startswith("$")]


def describe(experiment) -> Dict[str, Any]:
    """One experiment: its purpose, its flags, and its fixed/swept axes."""
    module = importlib.import_module(f"reasondb.coordinator.producers.{experiment.producer}")
    producer = get_producer(experiment.producer)

    error = None
    jobs: List[Any] = []
    try:
        with _stubbed_disk_state():
            args = _args_for(experiment)
            jobs = producer.enumerate_jobs(experiment.task_id, args.output_dir, args)
    except Exception as exc:  # noqa: BLE001 - a bad entry must not lose the whole report
        error = f"{type(exc).__name__}: {exc}"
        logger.warning("%s: could not enumerate (%s)", experiment.task_id, error)

    # Only the jobs that carry configuration. Label, filter-stats and precompute jobs are
    # per-benchmark chores, not points of the cross, and folding them in would report
    # every axis they happen to omit as swept.
    step_kinds = {"step", "approach", "point"}
    step_specs = [j.spec for j in jobs if j.spec.get("kind") in step_kinds]
    config = derive_run_config(
        step_specs,
        benchmarks=sorted({j.benchmark for j in jobs if j.benchmark}),
        source="planned",
    )
    return {
        "task_id": experiment.task_id,
        "producer": experiment.producer,
        "host": experiment.host,
        "purpose": (module.__doc__ or "").strip(),
        "argv": " ".join(experiment.args) or "(producer defaults only)",
        "config": config,
        "n_jobs": len(jobs),
        "n_step_jobs": len(step_specs),
        "other_jobs": sorted({j.spec.get("kind") for j in jobs if j.spec.get("kind") not in step_kinds}),
        "error": error,
        "deferred": _deferred_tokens(experiment),
    }


# ── Rendering ────────────────────────────────────────────────────────────────

def _purpose_html(text: str) -> str:
    """A producer docstring as paragraphs, with its RST tables left as preformatted text.

    Deliberately not a Markdown/RST renderer: these docstrings are read as source far more
    often than as HTML, and the arm tables in `ablation`/`label_reference` are the part
    worth preserving exactly.
    """
    out: List[str] = []
    for block in text.split("\n\n"):
        block = block.rstrip()
        if not block:
            continue
        looks_tabular = any(
            line.startswith(("===", "---", "| ", "    ", "\t")) or " | " in line
            for line in block.split("\n")
        )
        if looks_tabular:
            out.append(f"<pre>{html.escape(block)}</pre>")
        else:
            out.append(f"<p>{html.escape(block)}</p>")
    return "\n".join(out)


def _operator_sets_html(sets: List[Dict[str, Any]]) -> str:
    """One collapsed block per set. A seven-state walk lists 35 operators per set, so the
    headline is how many distinct sets there are and membership is one click away."""
    blocks = []
    for i, entry in enumerate(sets):
        steps = ", ".join(str(s) for s in entry["steps"])
        title = f"state {steps}" if steps else "operators"
        members = "".join(
            f"<div class='op'>{html.escape(m)}</div>" for m in entry["members"]
        )
        blocks.append(
            f"<details><summary><b>{html.escape(title)}</b> "
            f"<span class='muted'>· {len(entry['members'])} operators</span></summary>"
            f"<div class='ops'>{members}</div></details>"
        )
    return "".join(blocks)


def _axis_rows_html(config: Dict[str, Any]) -> str:
    rows: List[str] = []
    for axis in config["axes"]:
        if axis.get("variant") == "operator-sets":
            body = _operator_sets_html(axis["display"]) or "<span class=muted>—</span>"
        else:
            body = "".join(
                f"<span class='chip'>{html.escape(str(v))}</span>" for v in axis["display"]
            ) or "<span class=muted>—</span>"
        note = (
            f"<span class='note'>{html.escape(axis['note'])}</span>"
            if axis.get("note") else ""
        )
        rows.append(
            f"<div class='k'>{html.escape(axis['label'])}</div>"
            f"<div class='v'>{body}{note}</div>"
            f"<div class='t'><span class='tag {axis['kind']}'>{axis['kind']}</span></div>"
        )
    for group in config["groups"]:
        span = len(group["tuples"])
        rows.append(
            f"<div class='k linked' style='grid-row: span {span}'>"
            f"<div class='keys'>{html.escape(' / '.join(group['keyLabels']))}</div>"
            f"<div class='note'>varies together</div>"
            f"<div class='note'>{span} combination{'' if span == 1 else 's'}</div></div>"
        )
        for i, tuple_ in enumerate(group["tuples"]):
            cells = []
            for col, value in enumerate(tuple_):
                if group["variants"][col] == "operator-sets":
                    # One column may be an operator set rather than a scalar; same
                    # disclosure it gets as a standalone row.
                    cells.append(_operator_sets_html([{**value, "steps": []}]))
                else:
                    cells.append(f"<span class='chip'>{html.escape(str(value))}</span>")
            rows.append(f"<div class='v linked-row'>{''.join(cells)}</div>")
            if i == 0:
                rows.append(
                    f"<div class='t' style='grid-row: span {span}'>"
                    f"<span class='tag {group['kind']}'>{group['kind']}</span></div>"
                )
    return "\n".join(rows)


STYLE = """
:root {
  color-scheme: light dark;
  --page:#f9f9f7; --surface:#fcfcfb; --ink:#0b0b0b; --ink-2:#52514e; --muted:#898781;
  --border:rgba(11,11,11,0.12); --accent:#2a78d6; --radius:10px;
  --sans:ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
  --mono:ui-monospace,SFMono-Regular,Menlo,monospace;
}
@media (prefers-color-scheme: dark) {
  :root { --page:#0f0f0e; --surface:#171716; --ink:#f2f2ef; --ink-2:#bdbcb6;
          --muted:#8a8983; --border:rgba(242,242,239,0.14); --accent:#6ea8f0; }
}
* { box-sizing:border-box; }
body { margin:0; padding:32px 20px 80px; background:var(--page); color:var(--ink);
       font-family:var(--sans); line-height:1.55; }
main { max-width:1040px; margin:0 auto; }
h1 { font-size:26px; margin:0 0 6px; }
h2 { font-size:19px; margin:0; }
.lede { color:var(--ink-2); margin:0 0 28px; max-width:70ch; }
.exp { background:var(--surface); border:1px solid var(--border); border-radius:var(--radius);
       padding:20px 22px; margin-bottom:22px; }
.head { display:flex; align-items:baseline; gap:10px; flex-wrap:wrap; margin-bottom:4px; }
.pill { font-size:11px; font-weight:700; letter-spacing:.04em; padding:2px 8px;
        border-radius:999px; background:var(--accent); color:var(--page); }
.host { font-size:12px; color:var(--muted); font-family:var(--mono); }
.argv { font-family:var(--mono); font-size:12px; color:var(--ink-2); background:var(--page);
        border:1px solid var(--border); border-radius:6px; padding:7px 10px; margin:10px 0 16px;
        overflow-x:auto; white-space:pre; }
.purpose p { margin:0 0 10px; color:var(--ink-2); font-size:14px; max-width:74ch; }
.purpose pre { font-family:var(--mono); font-size:12px; background:var(--page); color:var(--ink-2);
        border:1px solid var(--border); border-radius:6px; padding:9px 11px; overflow-x:auto; }
details { margin-bottom:14px; }
summary { cursor:pointer; font-size:13px; color:var(--accent); }
.rows { display:grid; grid-template-columns:minmax(140px,1fr) minmax(220px,4fr) auto;
        gap:6px 14px; align-items:start; margin-top:6px; }
.k { font-size:12.5px; color:var(--ink-2); padding-top:4px; }
.v { display:flex; flex-wrap:wrap; gap:5px; align-items:center; }
.chip { font-size:12px; padding:3px 9px; border-radius:999px; border:1px solid var(--accent);
        background:color-mix(in srgb, var(--accent) 12%, transparent); color:var(--ink);
        font-weight:600; }
.tag { font-size:10.5px; font-weight:700; letter-spacing:.04em; padding:1px 7px; border-radius:999px; }
.tag.fixed { background:transparent; color:var(--ink-2); border:1px solid var(--border); }
.tag.swept { background:var(--accent); color:var(--page); }
.k.linked { border:1px solid var(--accent); border-radius:8px; padding:8px 10px;
        background:color-mix(in srgb, var(--accent) 10%, transparent);
        display:flex; flex-direction:column; justify-content:center; gap:2px; }
.note { font-size:11px; color:var(--muted); font-style:italic; }
.keys { font-size:11px; font-family:var(--mono); color:var(--ink-2); }
.linked-row { background:color-mix(in srgb, var(--accent) 6%, transparent);
        border-radius:6px; padding:3px 6px; flex-wrap:nowrap; overflow-x:auto; min-width:0; }
.linked-row details { flex:0 1 auto; min-width:0; }
.muted { color:var(--muted); }
.v > details { width:100%; }
.v > details + details { margin-top:3px; }
summary b { font-weight:600; color:var(--ink); }
.ops { display:flex; flex-direction:column; gap:2px; padding:6px 0 2px 14px;
        max-height:320px; overflow-y:auto; }
/* The identifier verbatim -- class, backend, model, ratios, vanilla. Monospace because
   it is an identifier, wrapping rather than ellipsised because reading all of it is the
   point. */
.op { font-family:var(--mono); font-size:11.5px; color:var(--ink-2); overflow-wrap:anywhere; }
.tag.unknown { background:transparent; color:var(--muted); border:1px dashed var(--border); }
.meta { font-size:12px; color:var(--muted); margin-top:14px; }
.warn { border-left:3px solid #d03b3b; padding-left:10px; color:#d03b3b; font-size:13px; }
.defer { border-left:3px solid var(--muted); padding-left:10px; color:var(--ink-2);
        font-size:12.5px; max-width:74ch; }
.defer code { font-family:var(--mono); font-size:11.5px; }
@media (max-width:700px) { .rows { grid-template-columns:1fr; } .k.linked { grid-row:auto !important; } }
"""


def render(experiments: List[Dict[str, Any]], config_path: str) -> str:
    sections = []
    for exp in experiments:
        warn = f"<p class='warn'>Could not enumerate: {html.escape(exp['error'])}</p>" if exp["error"] else ""
        deferred = ""
        if exp["deferred"]:
            names = html.escape(", ".join(exp["deferred"]))
            deferred = (
                f"<p class='defer'>{names} is expanded on the node, so whatever flags it "
                "carries are invisible here. If it supplies <code>--simulate</code>, the "
                "<b>Benchmark</b> and <b>Simulated</b> rows below are the parser's defaults "
                "rather than what this experiment will really run - the selection is derived "
                "from that mapping when <code>--benchmarks</code> is left at its default.</p>"
            )
        others = (
            f" · plus {html.escape(', '.join(exp['other_jobs']))} jobs"
            if exp["other_jobs"] else ""
        )
        sections.append(f"""
<section class="exp">
  <div class="head">
    <h2>{html.escape(exp['task_id'])}</h2>
    <span class="pill">{html.escape(exp['producer'])}</span>
    <span class="host">{html.escape(exp['host'])}</span>
  </div>
  <div class="argv">{html.escape(exp['argv'])}</div>
  {warn}{deferred}
  <details><summary>What this experiment is for</summary>
    <div class="purpose">{_purpose_html(exp['purpose'])}</div>
  </details>
  <div class="rows">{_axis_rows_html(exp['config'])}</div>
  <p class="meta">{exp['n_step_jobs']} configured job(s) of {exp['n_jobs']} enumerated{others}.</p>
</section>""")

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Stretto experiments</title><style>{STYLE}</style></head>
<body><main>
<h1>Stretto experiments</h1>
<p class="lede">Every experiment in <code>{html.escape(config_path)}</code>: what it is for,
and which axes it holds fixed versus sweeps. Generated from the cluster config, each
producer's own documentation, and a real enumeration of its jobs — so it cannot drift from
the code. The one exception is the <em>number</em> of sweep states, which depends on which
KV caches are materialized on the machine that runs the sweep; everything else is exactly
what the coordinator would enqueue.</p>
{''.join(sections)}
</main></body></html>
"""


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output", type=Path, default=Path("experiments.html"))
    parser.add_argument(
        "--experiments", type=str, nargs="+", default=None,
        help="Only these task ids (default: every experiment in the config).",
    )
    args = parser.parse_args()

    cluster = load_cluster_config(args.config)
    chosen = [
        e for e in cluster.experiments
        if args.experiments is None or e.task_id in args.experiments
    ]
    if not chosen:
        raise SystemExit(f"No experiments matched {args.experiments} in {args.config}.")

    described = [describe(e) for e in chosen]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render(described, args.config))
    logger.info("Wrote %s (%d experiment(s))", args.output, len(described))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
