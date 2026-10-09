"""What the next profiling round should do: how many rows, and which operators.

Pure functions over hand-built inputs: no pipeline, no database, no GPU.

The budget half is the decision the whole feature exists for -- "would a larger sample
pay for itself?" -- priced end to end rather than guessed. The pruning half is
deliberately *conservative*: it removes a candidate only on positive evidence that no
feasible restart, at any budget slot, wanted it.
"""

import torch

from reasondb.optimizer.base_optimizer import OperatorChoiceKey
from reasondb.optimizer.adaptive import (
    apply_to_pruning_mask,
    choose_sampling_budget,
    count_operators,
    derive_keep_set,
    describe_candidates,
    is_strict_shrink,
    mask_columns_for,
    protected_operator_ids,
)


# ── how many rows: is a larger sample worth it? ─────────────────────────────────


def _budget(costs, feasible, rounds_remaining=3, rows_remaining=100_000):
    return choose_sampling_budget(
        total_cost=torch.tensor(costs, dtype=torch.float32),
        meets_targets=torch.tensor(feasible, dtype=torch.bool),
        rounds_remaining=rounds_remaining,
        rows_remaining=rows_remaining,
    )


def test_it_keeps_sampling_when_more_rows_are_predicted_cheaper():
    """Slot 2's extra profiling and solve are outweighed by the execution it saves."""
    argmin, ended = _budget([100.0, 90.0, 80.0], [True, True, True])
    assert argmin == 2
    assert ended is False


def test_it_stops_when_the_extra_profiling_outweighs_the_saving():
    """Same plan quality either way, so paying to re-measure it is pure loss."""
    argmin, ended = _budget([80.0, 90.0, 100.0], [True, True, True])
    assert argmin == 0
    assert ended is True


def test_the_extra_solve_can_be_what_tips_it():
    # Execution saving of 5 against extra profiling of 3: worth it on its own...
    assert _budget([100.0, 97.0], [True, True])[1] is False
    # ...but not once the further GD solve's 4 seconds are charged too.
    assert _budget([100.0, 101.0], [True, True])[1] is True


def test_an_infeasible_slot_is_never_chased_however_cheap():
    """An infeasible plan is often one that runs almost nothing, so it looks free."""
    argmin, ended = _budget([100.0, 0.1], [True, False])
    assert argmin == 0
    assert ended is True


def test_a_feasible_expensive_slot_beats_an_infeasible_cheap_one():
    argmin, ended = _budget([100.0, 0.1, 90.0], [True, False, True])
    assert argmin == 2
    assert ended is False


def test_it_keeps_optimizing_while_the_current_sample_misses_the_targets():
    argmin, ended = _budget([100.0, 90.0], [False, True])
    assert ended is False
    del argmin


def test_the_last_round_takes_the_feasible_plan_it_has():
    """Deferring with no round left means falling through to the fallback plan."""
    _, ended = _budget([100.0, 80.0], [True, True], rounds_remaining=0)
    assert ended is True


def test_an_exhausted_table_also_ends_it():
    _, ended = _budget([100.0, 80.0], [True, True], rows_remaining=0)
    assert ended is True


def test_a_single_budget_slot_always_stops():
    """The non-adaptive shape: one slot, so there is nothing to defer to."""
    _, ended = _budget([42.0], [True])
    assert ended is True


def test_a_single_infeasible_slot_does_not_end_optimization():
    _, ended = _budget([42.0], [False])
    assert ended is False


# ── which operators ─────────────────────────────────────────────────────────────


class _StubConfig:
    """The five attributes `derive_keep_set` reads off a `DifferentiableConfig`.

    Layout mirrors `build_lookup_structures`: one lookup entry per (cascade, level),
    `gold_index_lookup` holding the count of *non-gold* candidates (which is also gold's
    own operator id), and columns allocated per level rather than per cascade.
    """

    def __init__(self, steps, pick_scores, num_initializations=2, num_budgets=1):
        self.operator_lookup = {}
        self.gold_index_lookup = {}
        column = 0
        for (cascade_id, level), num_operators in steps.items():
            key = OperatorChoiceKey(cascade_id=cascade_id, level=level)
            self.operator_lookup[key] = column
            self.gold_index_lookup[key] = num_operators - 1
            column += num_operators - 1
        self.num_pick_params = column
        self._plans = torch.tensor(pick_scores, dtype=torch.bool)
        self.num_initializations = num_initializations
        self.num_budgets = num_budgets
        self.num_jobs_single_method = num_initializations * num_budgets
        self.pruning_mask = torch.zeros(column, dtype=torch.bool)

    def discrete_plans(self):
        return self._plans


def _feasible(*rows):
    return torch.tensor(rows, dtype=torch.bool)


# One step, cascade 0 level 0, with 4 candidates: ids 0,1,2 choosable and 3 = gold.
ONE_STEP = {(0, 0): 4}


def test_gold_survives_even_when_no_restart_picks_anything():
    config = _StubConfig(ONE_STEP, pick_scores=[[False] * 3, [False] * 3])
    keep = derive_keep_set(
        config=config,
        used_method=0,
        feasible_by_budget=_feasible([True, True]),
        protected={},
        min_operators_per_step=1,
    )
    # Gold is where labels come from; prune it and the next round has nothing to score
    # the other candidates against.
    assert keep == {(0, 0): {3}}


def test_a_candidate_one_feasible_restart_picked_is_kept():
    config = _StubConfig(
        ONE_STEP,
        pick_scores=[
            [True, False, False],  # feasible, picks candidate 0
            [False, False, True],  # infeasible, picks candidate 2
        ],
    )
    keep = derive_keep_set(
        config=config,
        used_method=0,
        feasible_by_budget=_feasible([True, False]),
        protected={},
        min_operators_per_step=1,
    )
    # 2 is dropped: the only restart that wanted it did not meet the guarantees.
    assert keep == {(0, 0): {0, 3}}


def test_a_candidate_only_a_larger_sample_makes_feasible_survives():
    """The whole reason another round exists must not be pruned before it happens."""
    config = _StubConfig(
        ONE_STEP,
        pick_scores=[
            [True, False, False],  # budget slot 0, feasible now
            [False, False, False],
            [False, False, True],  # budget slot 1 (+more rows), feasible there
            [False, False, False],
        ],
        num_initializations=2,
        num_budgets=2,
    )
    keep = derive_keep_set(
        config=config,
        used_method=0,
        feasible_by_budget=_feasible([True, False], [True, False]),
        protected={},
        min_operators_per_step=1,
    )
    assert keep == {(0, 0): {0, 2, 3}}


def test_nothing_is_pruned_when_nothing_is_feasible():
    """No feasible restart is evidence of nothing, not evidence against everything."""
    config = _StubConfig(ONE_STEP, pick_scores=[[True, False, False], [False] * 3])
    assert (
        derive_keep_set(
            config=config,
            used_method=0,
            feasible_by_budget=_feasible([False, False]),
            protected={},
            min_operators_per_step=1,
        )
        is None
    )


def test_nothing_is_pruned_without_a_feasibility_mask():
    config = _StubConfig(ONE_STEP, pick_scores=[[True, False, False], [False] * 3])
    assert (
        derive_keep_set(
            config=config,
            used_method=0,
            feasible_by_budget=None,
            protected={},
            min_operators_per_step=1,
        )
        is None
    )


def test_a_misaligned_mask_declines_rather_than_pruning_arbitrarily():
    config = _StubConfig(ONE_STEP, pick_scores=[[True, False, False], [False] * 3])
    assert (
        derive_keep_set(
            config=config,
            used_method=0,
            feasible_by_budget=_feasible([True, True, True]),  # 3 vs 2 plans
            protected={},
            min_operators_per_step=1,
        )
        is None
    )


def test_protected_operators_survive():
    config = _StubConfig(ONE_STEP, pick_scores=[[False] * 3, [False] * 3])
    keep = derive_keep_set(
        config=config,
        used_method=0,
        feasible_by_budget=_feasible([True, True]),
        protected={(0, 0): {1}},  # e.g. the step's last executable operator
        min_operators_per_step=1,
    )
    assert keep == {(0, 0): {1, 3}}


def test_the_floor_keeps_a_cheap_alternative():
    config = _StubConfig(ONE_STEP, pick_scores=[[False] * 3, [False] * 3])
    keep = derive_keep_set(
        config=config,
        used_method=0,
        feasible_by_budget=_feasible([True, True]),
        protected={},
        min_operators_per_step=2,
    )
    # Gold plus the highest-quality proxy, so the step is never "gold or nothing".
    assert keep == {(0, 0): {2, 3}}


def test_columns_are_read_per_level_not_by_arithmetic():
    """A multi-level cascade allocates columns per level; `num_pick_params` over-counts.

    Deriving ranges from `operator_lookup`/`gold_index_lookup` is what keeps a level's
    candidates from being read out of the next level's columns.
    """
    config = _StubConfig(
        {(0, 0): 3, (0, 1): 4},  # columns 0,1 then columns 2,3,4
        pick_scores=[[False, True, False, False, True]],
        num_initializations=1,
    )
    keep = derive_keep_set(
        config=config,
        used_method=0,
        feasible_by_budget=_feasible([True]),
        protected={},
        min_operators_per_step=1,
    )
    assert keep == {(0, 0): {1, 2}, (0, 1): {2, 3}}


def test_the_used_method_selects_its_own_block_of_restarts():
    config = _StubConfig(
        ONE_STEP,
        pick_scores=[
            [True, False, False],  # method 0
            [False, False, False],
            [False, True, False],  # method 1
            [False, False, False],
        ],
        num_initializations=2,
    )
    keep_first = derive_keep_set(
        config=config,
        used_method=0,
        feasible_by_budget=_feasible([True, True]),
        protected={},
        min_operators_per_step=1,
    )
    keep_second = derive_keep_set(
        config=config,
        used_method=1,
        feasible_by_budget=_feasible([True, True]),
        protected={},
        min_operators_per_step=1,
    )
    assert keep_first == {(0, 0): {0, 3}}
    assert keep_second == {(0, 0): {1, 3}}


# ── monotonicity ────────────────────────────────────────────────────────────────


def test_the_first_filter_counts_as_a_shrink():
    assert is_strict_shrink({(0, 0): {0, 3}}, None)


def test_removing_a_candidate_is_a_shrink():
    assert is_strict_shrink({(0, 0): {3}}, {(0, 0): {0, 3}})


def test_keeping_the_same_set_is_not():
    assert not is_strict_shrink({(0, 0): {0, 3}}, {(0, 0): {0, 3}})


def test_widening_is_rejected():
    """A widened filter would ask for an operator whose earlier rows were discarded."""
    assert not is_strict_shrink({(0, 0): {0, 1, 3}}, {(0, 0): {0, 3}})


def test_a_new_step_appearing_is_rejected():
    assert not is_strict_shrink({(1, 0): {3}}, {(0, 0): {0, 3}})


# ── mask columns ────────────────────────────────────────────────────────────────


def test_masked_columns_are_exactly_the_dropped_candidates():
    config = _StubConfig(ONE_STEP, pick_scores=[[True, False, False]])
    assert sorted(mask_columns_for(config, {(0, 0): {0, 3}})) == [1, 2]


def test_gold_is_never_masked():
    """Gold carries no pick parameter at all -- it is forced on as the resolver."""
    config = _StubConfig(ONE_STEP, pick_scores=[[False, False, False]])
    columns = sorted(mask_columns_for(config, {(0, 0): {3}}))
    assert columns == [0, 1, 2]
    assert config.gold_index_lookup[OperatorChoiceKey(0, 0)] == 3


def test_a_step_absent_from_the_filter_is_left_alone():
    config = _StubConfig({(0, 0): 3, (0, 1): 4}, pick_scores=[[False] * 5])
    assert sorted(mask_columns_for(config, {(0, 0): {2}})) == [0, 1]


# ── telemetry counts ────────────────────────────────────────────────────────────


class _FakeStep:
    def __init__(self, num_operators):
        self.operators = list(range(num_operators))


class _FakePipeline:
    def __init__(self, cascades):
        self.steps_in_parallel = cascades


def test_counting_pruned_and_kept():
    pipeline = _FakePipeline([[_FakeStep(4)], [_FakeStep(3)]])
    assert count_operators(pipeline, None) == (0, 7)
    assert count_operators(pipeline, {(0, 0): {0, 3}, (1, 0): {2}}) == (4, 3)


# ── the protected set, and writing the mask ─────────────────────────────────────


class _FakeStepWithLabel(_FakeStep):
    """A step whose last candidate is a label source, so gold != last executable."""

    def __init__(self, num_operators, last_executable):
        super().__init__(num_operators)
        self._last_executable = last_executable

    def get_last_executable_operator_index(self):
        return self._last_executable


def test_protected_keeps_gold_and_the_last_executable_operator():
    """Gold is where labels come from; the last executable one is forced on as the
    resolver whatever its pick score says, so planning it without an observation would
    plan a tier nothing measured."""
    step = _FakeStepWithLabel(4, last_executable=3)
    protected = protected_operator_ids(_FakePipeline([[step]]))
    assert protected == {(0, 0): {3}}


def test_protected_spares_both_when_a_label_operator_ends_the_step():
    """With a human label source last, gold (3) is profiled but never executed, and the
    resolver is the candidate before it (2). Both have to survive."""
    step = _FakeStepWithLabel(4, last_executable=2)
    protected = protected_operator_ids(_FakePipeline([[step]]))
    assert protected == {(0, 0): {2, 3}}


def test_protected_covers_every_step_of_every_cascade():
    pipeline = _FakePipeline(
        [[_FakeStepWithLabel(3, 2), _FakeStepWithLabel(4, 3)], [_FakeStepWithLabel(2, 1)]]
    )
    assert set(protected_operator_ids(pipeline)) == {(0, 0), (0, 1), (1, 0)}


def test_applying_the_mask_forces_exactly_the_dropped_candidates_off():
    config = _StubConfig(ONE_STEP, pick_scores=[[True, True, True]])
    apply_to_pruning_mask(config, {(0, 0): {0, 3}})
    # Columns 1 and 2 are the candidates the keep set excludes; gold has no column.
    assert config.pruning_mask.tolist() == [False, True, True]


def test_applying_an_empty_narrowing_leaves_the_mask_alone():
    config = _StubConfig(ONE_STEP, pick_scores=[[True, True, True]])
    apply_to_pruning_mask(config, {(0, 0): {0, 1, 2, 3}})
    assert config.pruning_mask.tolist() == [False, False, False]


def test_the_mask_is_cumulative_across_rounds():
    """Pruning is irreversible, so a second narrowing adds to the first rather than
    replacing it."""
    config = _StubConfig(ONE_STEP, pick_scores=[[True, True, True]])
    apply_to_pruning_mask(config, {(0, 0): {0, 1, 3}})
    apply_to_pruning_mask(config, {(0, 0): {0, 3}})
    assert config.pruning_mask.tolist() == [False, True, True]


def test_the_mask_index_matches_the_mask_s_own_device():
    """The mask is deliberately host-resident; a writer that assumed `config.device`
    would silently add a copy per pruning pass rather than raising."""
    config = _StubConfig(ONE_STEP, pick_scores=[[True, True, True]])
    apply_to_pruning_mask(config, {(0, 0): {0, 3}})
    assert config.pruning_mask.device == torch.device("cpu")


# ── describing candidates for the dashboard ─────────────────────────────────────


class _FakeOperator:
    """Enough of a `PhysicalOperator` for the description to read."""

    def __init__(self, name, quality=1.0, fake_cost=1.0, label_only=False):
        self._name = name
        self.quality = quality
        self._fake_cost = fake_cost
        self.is_label_only = label_only

    def get_operation_identifier(self):
        return self._name


class _NamedStep:
    def __init__(self, operators, last_executable=None):
        self.operators = operators
        self._last_executable = (
            len(operators) - 1 if last_executable is None else last_executable
        )

    def get_last_executable_operator_index(self):
        return self._last_executable


def _named_pipeline(*steps):
    return _FakePipeline([[s] for s in steps])


def _config_for(steps, pruned_columns=()):
    """A stub config whose lookups match `steps`, with `pruned_columns` masked."""
    config = _StubConfig(
        {(i, 0): len(s.operators) for i, s in enumerate(steps)},
        pick_scores=[[False] * sum(len(s.operators) - 1 for s in steps)],
    )
    for column in pruned_columns:
        config.pruning_mask[column] = True
    return config


def test_every_candidate_is_described_not_only_the_pruned_ones():
    """Filtering answers "which were dropped"; grouping by the flag answers "were the
    right ones dropped", which is the question that says whether pruning is safe."""
    step = _NamedStep([_FakeOperator("A"), _FakeOperator("B"), _FakeOperator("gold")])
    rows = describe_candidates(_named_pipeline(step), _config_for([step], pruned_columns=[1]))
    assert [r["operator"] for r in rows] == ["A", "B", "gold"]
    assert [r["pruned"] for r in rows] == [False, True, False]


def test_gold_is_never_marked_pruned():
    """It carries no pick column at all, so there is nothing to mask -- and pruning it
    would leave the next round with no labels to score anything against."""
    step = _NamedStep([_FakeOperator("A"), _FakeOperator("gold")])
    config = _config_for([step])
    config.pruning_mask[:] = True  # prune everything that *can* be pruned
    rows = describe_candidates(_named_pipeline(step), config)
    gold = [r for r in rows if r["gold"]]
    assert len(gold) == 1
    assert gold[0]["pruned"] is False


def test_the_protected_flag_marks_what_pruning_may_never_drop():
    step = _NamedStep([_FakeOperator("A"), _FakeOperator("B"), _FakeOperator("gold")])
    rows = describe_candidates(_named_pipeline(step), _config_for([step]))
    by_name = {r["operator"]: r for r in rows}
    assert by_name["gold"]["protected"] is True
    assert by_name["A"]["protected"] is False


def test_a_label_source_leaves_the_resolver_protected_too():
    """Gold is a human label source here, so the resolver is the candidate before it and
    both have to survive."""
    step = _NamedStep(
        [_FakeOperator("A"), _FakeOperator("B"), _FakeOperator("labels", label_only=True)],
        last_executable=1,
    )
    rows = describe_candidates(_named_pipeline(step), _config_for([step]))
    by_name = {r["operator"]: r for r in rows}
    assert by_name["labels"]["protected"] is True
    assert by_name["labels"]["label_only"] is True
    assert by_name["B"]["protected"] is True
    assert by_name["A"]["protected"] is False


def test_columns_are_read_per_step_across_a_multi_step_pipeline():
    """The column offsets come from `operator_lookup`, so a second step's candidates must
    not be read out of the first step's columns."""
    first = _NamedStep([_FakeOperator("A"), _FakeOperator("gold1")])       # column 0
    second = _NamedStep([_FakeOperator("B"), _FakeOperator("C"), _FakeOperator("gold2")])
    steps = [first, second]                                                # columns 1, 2
    rows = describe_candidates(_named_pipeline(*steps), _config_for(steps, pruned_columns=[2]))
    by_name = {r["operator"]: r for r in rows}
    assert by_name["C"]["pruned"] is True
    assert by_name["A"]["pruned"] is False and by_name["B"]["pruned"] is False
    assert by_name["C"]["cascade_id"] == 1 and by_name["C"]["operator_id"] == 1


def test_the_rows_are_flat_scalars_the_facet_engine_can_group_by():
    """The dashboard derives dimensions from scalar fields; a nested value would be
    silently unusable as a group-by."""
    step = _NamedStep([_FakeOperator("A"), _FakeOperator("gold")])
    rows = describe_candidates(_named_pipeline(step), _config_for([step]))
    for row in rows:
        for key, value in row.items():
            assert isinstance(value, (str, int, float, bool, type(None))), (key, value)


def test_an_operator_missing_its_metadata_still_yields_a_row():
    """Non-KV operators carry no compression fields at all, and the description must
    degrade rather than raise -- it runs inside a solve."""

    class _Bare:
        def get_operation_identifier(self):
            return "TraditionalFilter"

    step = _NamedStep([_Bare(), _FakeOperator("gold")])
    rows = describe_candidates(_named_pipeline(step), _config_for([step]))
    bare = rows[0]
    assert bare["operator"] == "TraditionalFilter"
    assert bare["quality"] is None and bare["fake_cost"] is None
    assert bare["cr_label"] == "n/a"
