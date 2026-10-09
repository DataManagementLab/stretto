"""Which axes a set of jobs holds fixed, and which it sweeps.

The Python half of the question the Run tab's configuration panel answers in the browser
(``reasondb/monitor/static/run-config.js``). Same rule in both: an axis taking one
distinct value across the whole job set is FIXED, an axis taking several is SWEPT, and
two axes whose values only ever occur in fixed pairs are one LINKED group - which is what
a zipped ``--precision-guarantees``/``--recall-guarantees`` looks like from here.

Two implementations because they read the same data in two places - the browser has the
job specs as JSON, the report generator has them as ``Job`` objects - and neither can
call the other. They are pinned together by a golden fixture rather than by care:
``tests/test_experiment_report.py`` writes what this module produces for a known job list
into ``reasondb/monitor/static/tests/fixtures/golden-run-config.json``, and
``static/tests/run-config.test.js`` asserts the browser reproduces it. That is the same
arrangement ``golden-aggregates.json`` already uses for the collector's arithmetic.
"""

from typing import Any, Dict, List, Sequence, Tuple

from reasondb.monitor.dimensions import (
    DIMENSION_LABELS,
    DIMENSION_VALUE_LABELS,
    SPEC_KEYS_NOT_DIMENSIONS,
)

#: Spec keys that say *which job this is* rather than how the sweep was configured. They
#: vary per job by construction, so without this every run reports them as swept axes and
#: buries the ones that mean something. Mirrors NOT_CONFIGURATION in run-config.js.
NOT_CONFIGURATION = frozenset({"kind", "name", "job_id", "task_id", "producer"})

#: The order axes are presented in - the order the questions are actually asked in, not
#: alphabetical. Mirrors AXIS_ORDER in run-config.js.
AXIS_ORDER: Sequence[str] = (
    "benchmark",
    "split",
    "approach",
    "executor",
    "state_plan",
    "step_idx",
    "operator_set",
    "guarantee",
    "precision",
    "recall",
    "sample_size",
    "adaptive_sampling",
    "tune_parameters",
    "reorder",
    "labels",
    "label_set",
    "human_labels",
    "role",
    "use_indexes",
    "cost_type",
    "press_name",
    "simulate",
    "sweep_to_gold",
    "precompute_states",
    "precompute_modality",
)

SYNTHETIC_LABELS = {"operator_set": "Operator set", "guarantee": "Guarantee"}


def axis_label(key: str) -> str:
    if key in SYNTHETIC_LABELS:
        return SYNTHETIC_LABELS[key]
    return DIMENSION_LABELS.get(key, key.replace("_", " ").capitalize())


def value_label(key: str, value: Any) -> str:
    served = DIMENSION_VALUE_LABELS.get(key, {}).get(str(value))
    if served:
        return served
    if key.endswith("_model") or key == "model_name":
        return str(value).rsplit("/", 1)[-1]
    if isinstance(value, bool):
        return "yes" if value else "no"
    return str(value)


def _sort_key(value: Any):
    """Numbers before strings, and numerically among themselves - so a sample-size axis
    reads 10, 25, 50, 100 rather than 10, 100, 25, 50."""
    return (0, float(value), "") if isinstance(value, (int, float)) and not isinstance(value, bool) else (1, 0.0, str(value))


#: Keys whose ``None`` means "leave this alone" rather than "absent", and what to call
#: that. E.g. a producer that does not sweep the sample size writes ``sample_size: None``
#: (the optimizer keeps ``DEFAULT_SAMPLE_SIZE``); ``precompute_modality: None`` means one
#: pass records every modality. Mirrored by NONE_MEANS in run-config.js.
NONE_MEANS = {"sample_size": "optimizer default", "precompute_modality": "all"}


def spec_dimensions(spec: Dict[str, Any]) -> Dict[str, Any]:
    """The groupable scalars of one job spec. Mirrors ``specDimensions`` in the browser:
    flat scalars only, minus the paths and job metadata that are not knobs."""
    out: Dict[str, Any] = {}
    for key, value in (spec or {}).items():
        if key in SPEC_KEYS_NOT_DIMENSIONS or key in NOT_CONFIGURATION:
            continue
        if value is None:
            if key in NONE_MEANS:
                out[key] = NONE_MEANS[key]
            continue
        if isinstance(value, (dict, list, tuple, set)):
            continue
        out[key] = value
    return out


def job_rows(specs: Sequence[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, List[str]]]:
    """One comparable row per job, plus the operator sets keyed by their id.

    Two spec keys are lists and so are dropped by :func:`spec_dimensions`, but both are
    real axes and both have to take part in the linkage arithmetic below - which compares
    values, so each has to become one:

    - ``guarantee`` is ``[precision, recall]``, split into two scalar axes. That is what
      lets a zipped guarantee be *discovered* to vary together rather than hardcoded to.
    - ``operator_set`` is a list of identifiers, reduced to a stable key. The members are
      returned alongside, because the renderer still needs them.
    """
    rows: List[Dict[str, Any]] = []
    members_by_key: Dict[str, List[str]] = {}
    for spec in specs:
        row = spec_dimensions(spec)
        guarantee = (spec or {}).get("guarantee")
        if isinstance(guarantee, (list, tuple)) and len(guarantee) == 2:
            row["precision"], row["recall"] = guarantee[0], guarantee[1]
        members = (spec or {}).get("operator_set")
        if isinstance(members, (list, tuple)) and members:
            key = "\n".join(members)
            members_by_key[key] = list(members)
            row["operator_set"] = key
        rows.append(row)
    return rows, members_by_key


def _distinct(rows: Sequence[Dict[str, Any]], key: str) -> set:
    return {row[key] for row in rows if key in row}


def linked_groups(rows: Sequence[Dict[str, Any]], swept: Sequence[str]) -> List[List[str]]:
    """Which swept axes vary *together*, by the only rule there is.

    Two axes are tied when the combinations actually enumerated are fewer than the product
    of their value counts: a full cross means they move independently, anything less means
    the values are paired up. That covers a zipped ``--precision-guarantees``/
    ``--recall-guarantees``, a state and the operator set it decides (one set per state, so
    n pairs out of n x n), and an experiment whose arms are a chosen subset of a cross -
    the ablation's ``(state 0, optim_global)``, ``(state 1, optim_global)``,
    ``(state 1, no_optim)``.

    Closed transitively rather than reported as cliques: if A is tied to B and B to C, then
    the combinations that exist are what they are, and one group showing them is the honest
    rendering. Splitting into overlapping cliques would put an axis in two places and imply
    the two could be read separately.

    Only *swept* axes take part. A fixed axis is trivially "tied" to everything, and a run
    at one guarantee would otherwise be reported as zipped when zip and cross coincide.
    """
    edges: List[Tuple[str, str]] = []
    for i, first in enumerate(swept):
        for second in swept[i + 1 :]:
            pairs = {
                (row.get(first), row.get(second))
                for row in rows
                if first in row and second in row
            }
            if not pairs:
                continue
            if len(pairs) < len(_distinct(rows, first)) * len(_distinct(rows, second)):
                edges.append((first, second))

    groups: List[set] = []
    for first, second in edges:
        touching = [g for g in groups if first in g or second in g]
        groups = [g for g in groups if g not in touching]
        groups.append(set().union(*touching, {first, second}))
    order = {key: i for i, key in enumerate(swept)}
    return [sorted(g, key=lambda k: order.get(k, len(swept))) for g in groups]


def derive_run_config(
    specs: Sequence[Dict[str, Any]],
    benchmarks: Sequence[str] = (),
    source: str = "planned",
    complete: bool = True,
) -> Dict[str, Any]:
    """``{"source", "axes": [...], "groups": [...], "jobCount"}``.

    ``benchmarks`` names datasets the specs themselves do not. ``Job.__post_init__`` puts
    the benchmark *on* the spec, so this is only a top-up for a dataset whose jobs were
    all filtered out before we got here (e.g. a benchmark with nothing but a label job).
    It adds values to the axis, never rows, so a name supplied this way does not take
    part in linkage.

    ``complete`` says whether these specs are the whole enumerated grid. Linkage is
    inferred from *missing* combinations, so a half-finished run makes every axis look tied
    to every other; detection is skipped when the grid cannot be trusted to be whole. The
    report always enumerates fully; the panel sets it from whether it read the coordinator's
    job list or a finished run's recorded specs.
    """
    rows, members_by_key = job_rows(specs)
    values: Dict[str, set] = {}
    for row in rows:
        for key, value in row.items():
            if value is None or value == "":
                continue
            values.setdefault(key, set()).add(value)
    for name in benchmarks:
        values.setdefault("benchmark", set()).add(name)

    order = {key: i for i, key in enumerate(AXIS_ORDER)}

    def in_order(keys: Sequence[str]) -> List[str]:
        return sorted(keys, key=lambda k: (order.get(k, len(AXIS_ORDER)), k))

    swept = in_order([key for key, found in values.items() if len(found) > 1])
    groups_of = linked_groups(rows, swept) if complete else []

    def build_group(keys: Sequence[str]) -> Dict[str, Any]:
        """One group, with the combinations it actually holds. Built before the axes,
        because a group holding none of them releases its axes back to the table."""
        tuples = sorted(
            {tuple(row.get(k) for k in keys) for row in rows if all(k in row for k in keys)},
            key=lambda t: tuple(_sort_key(v) for v in t),
        )
        return {
            "id": "+".join(keys),
            "label": " / ".join(axis_label(k) for k in keys),
            "keys": list(keys),
            "keyLabels": [axis_label(k) for k in keys],
            "variants": ["operator-sets" if k == "operator_set" else None for k in keys],
            # Operator sets are keys in the tuples; the renderer resolves them here.
            "operatorSets": dict(members_by_key) if "operator_set" in keys else {},
            "tuples": [
                [
                    {"members": members_by_key.get(v, []), "steps": []}
                    if k == "operator_set"
                    else value_label(k, v)
                    for k, v in zip(keys, values_tuple)
                ]
                for values_tuple in tuples
            ],
            "kind": "fixed" if len(tuples) == 1 else "swept",
        }

    # Drop groups with no row holding all of their keys (an artifact of the transitive
    # closure, e.g. across label jobs that carry no state or operator set); their axes
    # are then rendered individually. Mirrors the same filter in run-config.js.
    groups = [g for g in map(build_group, groups_of) if g["tuples"]]
    grouped = {key for group in groups for key in group["keys"]}

    def render_values(key: str, raw: Sequence[Any]) -> Dict[str, Any]:
        """One axis's values, as the table shows them. Operator sets are lists of
        identifiers rather than scalars, so they carry a variant the renderer dispatches
        on; everything else is a chip."""
        if key == "operator_set":
            sets = [
                {
                    "steps": sorted(
                        {
                            row.get("step_idx")
                            for row in rows
                            if row.get("operator_set") == value
                            and row.get("step_idx") is not None
                        },
                        key=_sort_key,
                    ),
                    "members": members_by_key.get(value, []),
                }
                for value in raw
            ]
            sets.sort(key=lambda s: (s["steps"][:1] or [0], -len(s["members"])))
            return {"variant": "operator-sets", "values": sets, "display": sets}
        ordered = sorted(raw, key=_sort_key)
        return {
            "values": ordered,
            "display": [value_label(key, v) for v in ordered],
        }

    axes: List[Dict[str, Any]] = []
    for key in in_order(values):
        if key in grouped:
            continue
        rendered = render_values(key, values[key])
        axis = {
            "key": key,
            "label": axis_label(key),
            "kind": "fixed" if len(values[key]) == 1 else "swept",
            **rendered,
        }
        if key == "operator_set":
            n = len(axis["values"])
            axis["note"] = f"{n} set{'s' if n != 1 else ''}"
        axes.append(axis)

    # No spec carried an operator set: show it as unknown rather than omit the row.
    if "operator_set" not in values:
        axes.append(
            {
                "key": "operator_set",
                "label": axis_label("operator_set"),
                "kind": "unknown",
                "variant": "operator-sets",
                "values": [],
                "display": [],
                "note": "not recorded - enumerated before operator sets were captured",
            }
        )

    axes.sort(key=lambda a: (order.get(a["key"], len(AXIS_ORDER)), a["label"]))
    return {"source": source, "axes": axes, "groups": groups, "jobCount": len(specs)}
