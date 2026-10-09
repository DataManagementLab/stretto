"""The YAML launch plan behind scripts/start-{configurators,workers}.sh.

Both scripts ssh what this module renders, so an error here launches the wrong thing (or
nothing) on every node. The three properties worth pinning: the
coordinator and the worker plan agree about where each task lives, no rendered argument
can break out of the single-quoted remote command the scripts build, and what the config
renders is something the coordinator's own parser accepts.
"""

import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from reasondb.coordinator.cluster import (
    emit_coordinators,
    emit_settings,
    emit_workers,
    format_node_selection,
    load_cluster_config,
    parse_cluster_config,
    parse_node_selection,
    render_args,
    shell_join,
)

BASE = {
    "cluster": {"prefix": "sweep-", "port": 5099, "repo": "/repo", "bashrc": "/rc", "home": "/home"},
    "workers": {"nodes": 3},
    "experiments": [
        {"task_id": "samp01", "producer": "sample_size"},
        {"task_id": "ops01", "producer": "operator_count"},
        {"task_id": "tune01", "producer": "tuning"},
    ],
}


def config(**changes):
    raw = yaml.safe_load(yaml.safe_dump(BASE))
    for section, value in changes.items():
        raw[section] = value
    return parse_cluster_config(raw)


#: The precompute stores every random-benchmark experiment replays, in the order the
#: config writes them. The config repeats the mapping per experiment; spelling it once
#: here and asserting it against each entry keeps the copies in sync.
RANDOM_STORES = [
    "artwork_random_medium=artwork_precompute_kv.json",
    "rotowire_random=rotowire_precompute_kv.json",
    "movie_random_huge=movie_huge_precompute_kv.json",
    "email_random=email_precompute_kv.json",
    "ecommerce_random_large=ecomm_precompute_kv.json",
]


def _mapping(args, flag):
    """The ``NAME=VALUE`` pairs following ``flag``, as a dict."""
    return dict(
        token.split("=", 1) for token in args[args.index(flag) + 1:] if "=" in token
    )


# ── The shipped config ───────────────────────────────────────────────────────────


def test_shipped_config_puts_each_experiment_on_its_own_node():
    cfg = load_cluster_config("scripts/cluster.yaml")
    assert [(e.task_id, e.producer, e.host) for e in cfg.experiments] == [
        ("base01", "baselines", "sweep-1"),
        ("samp01", "sample_size", "sweep-2"),
        ("ops01", "operator_count", "sweep-3"),
        ("ref01", "label_reference", "sweep-4"),
        ("abl01", "ablation", "sweep-5"),
        ("mode01", "baselines", "sweep-6"),
        ("kvop01", "kv_operator", "sweep-7"),
        # No recording passes: every entry replays a store that already exists on its
        # node, so the fleet's `capability: simulate` drains the whole file.
    ]
    # base01 carries no flags but the stores it replays: the whole experiment is the
    # producer's own pins - the default state, tuning on, DEFAULT_SAMPLE_SIZE, and the
    # three approaches.
    assert cfg.experiments[0].args == ["--simulate", *RANDOM_STORES]
    # samp01 extends the producer's [10, 25, 50, 100] by one point past
    # DEFAULT_SAMPLE_SIZE.
    assert cfg.experiments[1].args == [
        "--sample-sizes", "10", "25", "50", "100", "150",
        "--simulate", *RANDOM_STORES,
    ]
    # ops01 carries no --sample-sizes at all: operator_count's own default is
    # DEFAULT_SAMPLE_SIZE, so the curve is measured at the budget a deployment draws.
    #
    # Its dataset list is the one that is not a free choice: its greedy walk visits every
    # materialized level, so it cannot replay a store narrowed by --precompute-states.
    # That includes `movie_random_huge`, whose store must be at full coverage before this
    # task is launched (the yaml spells out the pass).
    assert cfg.experiments[2].args == ["--simulate", *RANDOM_STORES]
    # ref01 is the one replaying entry whose mapping names the *curated* datasets, and
    # those five files have to be on the node: --simulate checks each one exists before
    # the coordinator enumerates anything.
    assert cfg.experiments[3].args == [
        "--precision-guarantees", "0.5", "0.7", "0.9",
        "--recall-guarantees", "0.5", "0.7", "0.9",
        "--simulate",
        "artwork_curated=artwork_curated_kv.json",
        "email_curated=email_curated_kv.json",
        "rotowire_curated=rotowire_curated_kv.json",
        "ecommerce_curated=ecommerce_curated_kv.json",
        "movie_huge_curated=movie_huge_curated_kv.json",
    ]
    # abl01 wraps the *sweep* engine, so it runs the random benchmarks - the same
    # recordings every other entry replays.
    assert cfg.experiments[4].args == [
        "--precision-guarantees", "0.5", "0.7", "0.9",
        "--recall-guarantees", "0.5", "0.7", "0.9",
        "--simulate", *RANDOM_STORES,
    ]
    # mode01 runs the three optimization modes over the whole search space at a larger
    # profiling budget. `full` is a single-state plan, which is the only kind `baselines`
    # accepts - a greedy plan there would be ops01's experiment under this producer's name.
    assert cfg.experiments[5].args == [
        "--approaches", "optim_global", "optim_local", "optim_shift_budget",
        "--state-plan", "full",
        "--sample-sizes", "150",
        "--simulate", *RANDOM_STORES,
    ]
    # kvop01 adds one KV operator per state to the uncompressed reference suite, over
    # every dataset.
    assert cfg.experiments[6].args == [
        "--state-plan", "kv_operator_marginal",
        "--precision-guarantees", "0.5", "0.7", "0.9",
        "--recall-guarantees", "0.5", "0.7", "0.9",
        "--simulate", *RANDOM_STORES,
    ]
    # Which entries replay is readable off the config itself, so the experiment report and
    # --dry-run can both say which benchmarks each experiment runs.
    replaying = {e.task_id for e in cfg.experiments if "--simulate" in e.args}
    assert replaying == {"base01", "samp01", "ops01", "ref01", "abl01", "mode01", "kvop01"}
    # And nothing defers to the node's shell: a config using a `$VAR` could not be read
    # here or reported.
    assert not any(token.startswith("$") for e in cfg.experiments for token in e.argv())
    # Every entry replays and none records. A run may not do both, so this is also what
    # makes the whole file drainable by one `capability: simulate` fleet.
    assert replaying == {e.task_id for e in cfg.experiments}
    assert not [e for e in cfg.experiments if "--precompute" in e.args]

    # The fleet has to be able to host what the config asks for, and one worker per node
    # is what makes "its own node" true. The exact fleet size is an operational knob, so
    # only the lower bound is pinned.
    assert cfg.worker_nodes >= len(cfg.experiments)
    assert cfg.workers_per_node == 1
    assert cfg.capability == "simulate"


def test_shipped_config_renders_arguments_the_coordinator_accepts():
    """Every experiment's argv parses under the parser that will receive it.

    The assertions above pin what the *config* says; this pins that what it says is
    runnable. The two come apart where a YAML type and a flag's arity disagree -
    `adaptive-sampling: true` renders a valueless `--adaptive-sampling`, which argparse
    rejects because the flag takes `true`/`false`. Rendering is not enough; only the
    real parser knows arity.

    Producer-level pinning (`reject_pinned`) is deliberately not exercised here: it runs
    at enumeration, needs the benchmark data on disk, and has its own tests.
    """
    coordinator = pytest.importorskip("scripts.run_coordinator")
    cfg = load_cluster_config("scripts/cluster.yaml")
    for experiment in cfg.experiments:
        # Any `$VAR` token would be expanded by the node's shell; drop it before parsing.
        argv = [token for token in experiment.argv() if not token.startswith("$")]
        parsed = coordinator.build_parser().parse_args(argv)
        assert parsed.task_id == experiment.task_id
        assert parsed.producer == experiment.producer
    # And a multi-value axis flag arrives as the list the producer sweeps, not as one
    # token or a bare switch.
    samp01 = [e for e in cfg.experiments if e.task_id == "samp01"][0]
    argv = [token for token in samp01.argv() if not token.startswith("$")]
    assert coordinator.build_parser().parse_args(argv).sample_sizes == [10, 25, 50, 100, 150]


def test_shipped_config_renders_arguments_the_worker_accepts():
    """The worker half of the check above: a rendered worker command parses too.

    The pin budget is the one worker flag the config *always* renders, so a rename on
    either side would make every worker exit on "unrecognized arguments" at launch.
    """
    worker_module = pytest.importorskip("scripts.run_worker")
    cfg = load_cluster_config("scripts/cluster.yaml")
    for worker in cfg.workers():
        parsed = worker_module.build_parser().parse_args(worker.argv())
        assert parsed.worker_id == worker.worker_id
        assert parsed.kv_cache_pin_gb == cfg.kv_cache_pin_gb
    # And it is sized rather than left at the start scripts' own 0, which would refuse
    # every -in-memory operator in the default suite at setup().
    assert cfg.kv_cache_pin_gb > 0


# ── Node assignment: the thing both scripts must agree on ────────────────────────


def test_worker_reaches_its_own_coordinator_as_localhost():
    workers = config().workers()
    # Worker on node 2 is the only one that may say localhost for ops01.
    by_id = {worker.worker_id: dict(worker.targets) for worker in workers}
    assert by_id["sweep-2"]["ops01"] == "http://localhost:5099"
    assert by_id["sweep-1"]["ops01"] == "http://sweep-2:5099"


def test_rotation_starts_each_worker_on_a_different_task():
    cfg = config(workers={"nodes": 3, "rotate": True})
    firsts = [worker.targets[0][0] for worker in cfg.workers()]
    assert firsts == ["samp01", "ops01", "tune01"]


def test_the_whole_fleet_drains_in_config_order_by_default():
    # The default, and the point of it: every worker queues on the first experiment, so
    # the fleet finishes it before any of it starts the second. Rotation is opt-in.
    workers = config().workers()
    assert not any(w.targets[0][0] != "samp01" for w in workers)
    assert [task for task, _ in workers[0].targets] == ["samp01", "ops01", "tune01"]


def test_no_rotation_drains_in_config_order():
    cfg = config(workers={"nodes": 2, "rotate": False})
    assert [w.targets[0][0] for w in cfg.workers()] == ["samp01", "samp01"]


def test_workers_per_node_names_each_process_and_spreads_them():
    cfg = config(workers={"nodes": 2, "per_node": 2, "rotate": True})
    workers = cfg.workers()
    assert [w.worker_id for w in workers] == ["sweep-1-w1", "sweep-1-w2", "sweep-2-w1", "sweep-2-w2"]
    assert [w.host for w in workers] == ["sweep-1", "sweep-1", "sweep-2", "sweep-2"]
    # Rotation counts across the fleet, not per node, so two workers on one node do not
    # both pile onto the same coordinator.
    assert [w.targets[0][0] for w in workers] == ["samp01", "ops01", "tune01", "samp01"]


def test_explicit_node_and_host_win_over_the_position():
    cfg = config(
        experiments=[
            {"task_id": "a", "producer": "tuning", "node": 3},
            {"task_id": "b", "producer": "tuning", "host": "gpu-box", "port": 6000},
        ]
    )
    assert [(e.host, e.port) for e in cfg.experiments] == [("sweep-3", 5099), ("gpu-box", 6000)]
    assert dict(cfg.workers()[0].targets)["b"] == "http://gpu-box:6000"


def test_two_experiments_may_share_a_node_on_different_ports():
    cfg = config(
        experiments=[
            {"task_id": "a", "producer": "tuning", "node": 1},
            {"task_id": "b", "producer": "tuning", "node": 1, "port": 5100},
        ]
    )
    targets = dict(cfg.workers()[0].targets)
    assert targets == {"a": "http://localhost:5099", "b": "http://localhost:5100"}


def test_same_node_and_port_twice_is_rejected():
    with pytest.raises(ValueError, match="both bind"):
        config(
            experiments=[
                {"task_id": "a", "producer": "tuning", "node": 1},
                {"task_id": "b", "producer": "tuning", "node": 1},
            ]
        )


def test_selection_keeps_the_node_assignment_of_the_full_config():
    raw = yaml.safe_load(yaml.safe_dump(BASE))
    cfg = parse_cluster_config(raw, select=["tune01"])
    (experiment,) = cfg.experiments
    # Still node 3, not renumbered to 1 - the rest of the fleet is pointed there already.
    assert (experiment.node, experiment.host) == (3, "sweep-3")
    targets = {w.worker_id: dict(w.targets) for w in cfg.workers()}
    assert targets["sweep-3"] == {"tune01": "http://localhost:5099"}
    assert targets["sweep-1"] == {"tune01": "http://sweep-3:5099"}


def test_unknown_selection_is_rejected():
    with pytest.raises(ValueError, match="not in the config"):
        parse_cluster_config(yaml.safe_load(yaml.safe_dump(BASE)), select=["nope"])


# ── Arguments ────────────────────────────────────────────────────────────────────


def test_mapping_args_render_as_flags():
    rendered = render_args(
        {
            "benchmarks": ["movie_random", "artwork_random_medium"],
            "sweep_to_gold": True,
            "use-indexes": False,
            "sample-sizes": 50,
            "tune-parameters": [True, False],
            "simulate": {"movie_random": "movie.json"},
        },
        "args",
    )
    assert rendered == [
        "--benchmarks", "movie_random", "artwork_random_medium",
        "--sweep-to-gold",
        "--sample-sizes", "50",
        "--tune-parameters", "true", "false",
        "--simulate", "movie_random=movie.json",
    ]


def test_list_and_string_args_are_split_like_a_shell():
    assert render_args(["--benchmarks movie_random", "--device cpu"], "args") == [
        "--benchmarks", "movie_random", "--device", "cpu",
    ]
    assert render_args("--local", "args") == ["--local"]


def test_experiment_args_follow_the_shared_ones():
    cfg = config(
        coordinator={"args": {"benchmarks": ["movie_random"]}},
        experiments=[{"task_id": "a", "producer": "tuning", "args": {"benchmarks": ["ecommerce_random_large"]}}],
    )
    argv = cfg.experiments[0].argv()
    assert argv == [
        "--task-id", "a", "--producer", "tuning", "--port", "5099",
        "--benchmarks", "movie_random",
        "--benchmarks", "ecommerce_random_large",
    ]
    # argparse keeps the last one, so the experiment's own value is what runs.
    assert argv.index("ecommerce_random_large") > argv.index("movie_random")


def test_an_argv_holds_only_what_the_config_says():
    """Nothing is appended from the node's environment.

    The --simulate mapping lives in each experiment, so the config states its own
    datasets. An unknown key such as `sim_var` is an error rather than silently ignored,
    which is what `EXPERIMENT_KEYS` is for.
    """
    cfg = config(experiments=[{"task_id": "a", "producer": "tuning"}])
    assert cfg.experiments[0].argv() == [
        "--task-id", "a", "--producer", "tuning", "--port", "5099",
    ]
    with pytest.raises(ValueError, match="unknown key"):
        config(experiments=[{"task_id": "a", "producer": "tuning", "sim_var": "SIM"}])


def test_worker_args_reach_the_worker_command():
    cfg = config(workers={"nodes": 1, "args": {"use_indexes": True, "server-ready-timeout-s": 3600}})
    assert cfg.workers()[0].argv()[-3:] == ["--use-indexes", "--server-ready-timeout-s", "3600"]


def test_pin_budget_is_rendered_per_worker_and_defaults_to_50():
    argv = config(workers={"nodes": 1}).workers()[0].argv()
    assert argv[argv.index("--kv-cache-pin-gb") + 1] == "50"


@pytest.mark.parametrize("value, token", [(0, "0"), (12.5, "12.5"), ("120", "120")])
def test_pin_budget_is_a_number_of_gb(value, token):
    cfg = config(workers={"nodes": 1, "kv_cache_pin_gb": value})
    assert cfg.workers()[0].argv()[-1] == token


def test_a_workers_args_copy_of_the_pin_budget_wins():
    """`workers.args` is the escape hatch for anything run_worker.py takes, and it may
    not lose to the key the config renders for everyone - argparse keeps the last
    occurrence, so the built-in flag has to come first."""
    worker_module = pytest.importorskip("scripts.run_worker")
    cfg = config(workers={"nodes": 1, "kv_cache_pin_gb": 50, "args": {"kv-cache-pin-gb": 200}})
    assert worker_module.build_parser().parse_args(cfg.workers()[0].argv()).kv_cache_pin_gb == 200


@pytest.mark.parametrize("value", ["a'b", 'a"b', "a\\b", "a;b", "a|b", "a`b`", "a\tb", "a\nb"])
def test_arguments_that_would_escape_the_remote_quoting_are_rejected(value):
    # The scripts build ssh host "bash -ic '<command>'": a quote of either kind, a
    # backslash or a metacharacter here is a command the node runs, or a mangled TSV line.
    with pytest.raises(ValueError):
        render_args({"debug-query": value}, "args")


def test_spaces_are_backslash_escaped_rather_than_quoted():
    assert shell_join(["--debug-query", "two words"]) == r"--debug-query two\ words"


def test_dollar_signs_survive_for_the_node_to_expand():
    assert shell_join(["$SIM"]) == "$SIM"


# ── Config validation ────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"cluster": {"prefix": "sweep-"}}, "cluster.repo is required"),
        ({"experiments": []}, "at least one entry"),
        ({"workers": {"nodes": 0}}, "at least 1"),
        ({"workers": {"nodes": 1, "capability": "gpu"}}, "capability"),
        ({"workers": {"nodes": 1, "capabilty": "text"}}, "unknown key"),
        ({"workers": {"nodes": 1, "kv_cache_pin_gb": -1}}, "kv_cache_pin_gb must be >= 0"),
        ({"workers": {"nodes": 1, "kv_cache_pin_gb": "lots"}}, "kv_cache_pin_gb must be a number"),
        ({"experiments": [{"task_id": "a", "producer": "tuning", "benchmarks": []}]}, "unknown key"),
        ({"experiments": [{"task_id": "a"}]}, "task_id and producer"),
        ({"experiments": [{"task_id": "a", "producer": "no_such_producer"}]}, "unknown producer"),
        (
            {"experiments": [{"task_id": "a", "producer": "tuning"}, {"task_id": "a", "producer": "tuning", "node": 2}]},
            "used twice",
        ),
    ],
)
def test_rejected_configs(changes, message):
    with pytest.raises(ValueError, match=message):
        config(**changes)


def test_hosts_list_replaces_the_prefix():
    cfg = config(cluster={"hosts": ["alpha", "beta", "gamma"], "repo": "/repo"}, workers={"nodes": 2})
    assert [e.host for e in cfg.experiments] == ["alpha", "beta", "gamma"]
    assert [w.host for w in cfg.workers()] == ["alpha", "beta"]


def test_more_worker_nodes_than_hosts_is_rejected():
    with pytest.raises(ValueError, match="lists only"):
        config(cluster={"hosts": ["alpha"], "repo": "/repo"}, workers={"nodes": 2})


# ── Which nodes the fleet runs on ────────────────────────────────────────────────


@pytest.mark.parametrize(
    "spec, expected",
    [
        # A lone integer is a *count*, in both the YAML and the string form a --set
        # override arrives as.
        (3, (1, 2, 3)),
        ("3", (1, 2, 3)),
        ("1-3", (1, 2, 3)),
        ("8-12", (8, 9, 10, 11, 12)),
        ("9-9", (9,)),
        ("1,3,5-7", (1, 3, 5, 6, 7)),
        (" 8 - 12 ", (8, 9, 10, 11, 12)),
    ],
)
def test_node_selection_spellings(spec, expected):
    assert parse_node_selection(spec) == expected


@pytest.mark.parametrize(
    "spec, message",
    [
        (0, "at least 1"),
        ("0-3", "1-based"),
        ("12-8", "counts down"),
        ("1-3,2", "named twice"),
        ("2,2", "named twice"),
        ("a-b", "not a node id"),
        ("", "empty"),
        ("1,,3", "empty entry"),
    ],
)
def test_rejected_node_selections(spec, message):
    with pytest.raises(ValueError, match=message):
        parse_node_selection(spec)


def test_a_range_of_nodes_launches_on_those_nodes():
    """`--nodes 8-12` is five workers on sweep-8..12, not on the first five nodes."""
    cfg = parse_cluster_config(yaml.safe_load(yaml.safe_dump(BASE)), overrides={"workers.nodes": "8-12"})
    assert cfg.worker_node_ids == (8, 9, 10, 11, 12)
    # The count is how many nodes, not the highest id - it is what the launcher reports
    # and what NUM_WORKERS multiplies.
    assert cfg.worker_nodes == 5
    assert [w.host for w in cfg.workers()] == [f"sweep-{i}" for i in range(8, 13)]


def test_a_range_keeps_the_localhost_hop_right():
    """The one thing node ids decide: which worker shares a node with which coordinator."""
    cfg = parse_cluster_config(yaml.safe_load(yaml.safe_dump(BASE)), overrides={"workers.nodes": "2-3"})
    urls = {w.host: dict(w.targets) for w in cfg.workers()}
    # samp01 is on node 1, ops01 on node 2, tune01 on node 3.
    assert urls["sweep-2"]["ops01"] == "http://localhost:5099"
    assert urls["sweep-2"]["samp01"] == "http://sweep-1:5099"
    assert urls["sweep-3"]["ops01"] == "http://sweep-2:5099"


def test_node_ids_index_the_hosts_list():
    cfg = config(cluster={"hosts": ["alpha", "beta", "gamma"], "repo": "/repo"}, workers={"nodes": "2-3"})
    assert [w.host for w in cfg.workers()] == ["beta", "gamma"]
    with pytest.raises(ValueError, match="lists only"):
        config(cluster={"hosts": ["alpha", "beta"], "repo": "/repo"}, workers={"nodes": "2-3"})


def test_settings_name_the_nodes_as_well_as_counting_them():
    settings = dict(line.split("=", 1) for line in emit_settings(config(workers={"nodes": "8-12"})).splitlines())
    assert settings["WORKER_NODES"] == "5"
    assert settings["WORKER_NODE_IDS"] == "8-12"
    assert settings["NUM_WORKERS"] == "5"


@pytest.mark.parametrize("node_ids", [(1, 2, 3), (8, 9, 10, 11, 12), (1, 3, 5, 6, 7), (9,)])
def test_node_selection_formats_back_to_what_it_parses(node_ids):
    assert parse_node_selection(format_node_selection(node_ids)) == node_ids


def test_overrides_apply_and_empty_ones_do_not():
    raw = yaml.safe_load(yaml.safe_dump(BASE))
    cfg = parse_cluster_config(raw, overrides={"workers.nodes": "5", "workers.rotate": "false", "cluster.prefix": ""})
    assert (cfg.worker_nodes, cfg.rotate, cfg.prefix) == (5, False, "sweep-")


def test_unknown_override_is_rejected():
    with pytest.raises(ValueError, match="unknown key"):
        parse_cluster_config(yaml.safe_load(yaml.safe_dump(BASE)), overrides={"workers.node": "5"})


# ── What the shell reads back ────────────────────────────────────────────────────


def test_emitted_lines_are_tab_separated_and_one_per_launch():
    cfg = config(workers={"nodes": 2})
    coordinators = emit_coordinators(cfg).splitlines()
    assert len(coordinators) == 3
    node, task, producer, port, args = coordinators[0].split("\t")
    assert (node, task, producer, port) == ("sweep-1", "samp01", "sample_size", "5099")
    assert args.startswith("--task-id samp01 ")

    workers = emit_workers(cfg).splitlines()
    assert len(workers) == 2
    assert all(len(line.split("\t")) == 3 for line in workers)


def test_settings_are_shell_quoted_key_value_lines():
    settings = dict(line.split("=", 1) for line in emit_settings(config()).splitlines())
    assert settings["REPO"] == "/repo"
    assert settings["WORKER_NODES"] == "3"
    assert settings["TASK_IDS"] == "'samp01 ops01 tune01'"
    assert settings["CACHE_DIR"] == "''"
