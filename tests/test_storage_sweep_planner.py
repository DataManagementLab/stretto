"""Golden sequences for the two greedy storage planners.

``plan_greedy_states_indexed`` / ``plan_greedy_states_direct`` are pure functions
over ``(available, valid_levels, storage_table)`` — no disk, no models — and they
decide the entire shape of a storage sweep: how many steps it has, which slot
gives up storage next, and where it stops.

The fixture is fully synthetic (the planners never consult the spec tables, only
the ``valid_levels``/``storage_table`` they are handed), with byte counts chosen so
every greedy choice has a unique winner and the whole sequence is forced.

The terminal-state assertions are deliberate: by default both planners stop with every
slot still holding exactly one materialized cache, never an empty one. ``--sweep-to-gold``
extends the walk by one further step per slot, which is tested separately below.
"""

import pytest

try:
    from reasondb.evaluation.parameter_sweep import (
        ModelSlot,
        plan_greedy_states_direct,
        plan_greedy_states_indexed,
    )
except ImportError:
    pytest.skip("parameter_sweep deps not installed", allow_module_level=True)

SMALL = ModelSlot("text_small", "text", "fake/small", large=False)
LARGE = ModelSlot("text_large", "text", "fake/large", large=True)

AVAILABLE = [SMALL, LARGE]

#: Deliberately uneven, and not the same ratio grid per slot: the planners must
#: read each slot's own levels rather than a shared grid.
VALID_LEVELS = {
    "text_small": [0.0, 0.3, 0.5],
    "text_large": [0.0, 0.5],
}

#: bytes[slot][cr]. Chosen so each greedy comparison has a strict winner.
STORAGE_TABLE = {
    "text_small": {0.0: 1000, 0.3: 400, 0.5: 100},
    "text_large": {0.0: 800, 0.5: 300},
}


def test_indexed_planner_golden_sequence():
    """One cache per slot; each step raises the ratio that frees the most bytes."""
    states = plan_greedy_states_indexed(AVAILABLE, VALID_LEVELS, STORAGE_TABLE)

    assert states == [
        # step 0: every slot at its lowest ratio = maximum storage.
        ({"text_small": [0.0], "text_large": [0.0]}, 1800),
        # small 0.0->0.3 frees 600, larger than large's 0.0->0.5 (500).
        ({"text_small": [0.3], "text_large": [0.0]}, 1200),
        # now large's 500 beats small's 0.3->0.5 (300).
        ({"text_small": [0.3], "text_large": [0.5]}, 700),
        # only small can still advance.
        ({"text_small": [0.5], "text_large": [0.5]}, 400),
    ]


def test_direct_planner_golden_sequence():
    """Every level is its own cache; each step drops the lowest retained one."""
    states = plan_greedy_states_direct(AVAILABLE, VALID_LEVELS, STORAGE_TABLE)

    assert states == [
        # step 0 retains ALL levels of every slot: maximum choice, maximum storage.
        ({"text_small": [0.0, 0.3, 0.5], "text_large": [0.0, 0.5]}, 2600),
        # dropping small's 0.0 frees 1000, more than large's 0.0 (800).
        ({"text_small": [0.3, 0.5], "text_large": [0.0, 0.5]}, 1600),
        # now large's 0.0 (800) beats small's 0.3 (400).
        ({"text_small": [0.3, 0.5], "text_large": [0.5]}, 800),
        ({"text_small": [0.5], "text_large": [0.5]}, 400),
    ]


@pytest.mark.parametrize(
    "planner", [plan_greedy_states_indexed, plan_greedy_states_direct]
)
def test_sweep_stops_before_vanilla(planner):
    """Both planners terminate with one materialized cache per slot, never zero.

    This is deliberate: ``valid_levels`` is built from the non-vanilla spec entries, so
    the cheapest reachable state still pays for one compressed cache per slot.
    ``--sweep-to-gold`` is what lifts it.
    """
    states = planner(AVAILABLE, VALID_LEVELS, STORAGE_TABLE)
    final_state, final_footprint = states[-1]

    assert all(len(crs) == 1 for crs in final_state.values())
    assert final_state == {"text_small": [0.5], "text_large": [0.5]}
    assert final_footprint == 400, "cheapest state still stores one cache per slot"


@pytest.mark.parametrize(
    "planner", [plan_greedy_states_indexed, plan_greedy_states_direct]
)
def test_footprints_decrease_monotonically(planner):
    """The sweep walks expensive -> cheap; the monitor and plots assume that order."""
    footprints = [f for _state, f in planner(AVAILABLE, VALID_LEVELS, STORAGE_TABLE)]
    assert footprints == sorted(footprints, reverse=True)
    assert len(set(footprints)) == len(footprints), "no step may be a no-op"


@pytest.mark.parametrize(
    "planner", [plan_greedy_states_indexed, plan_greedy_states_direct]
)
def test_single_level_slot_stays_fixed(planner):
    """A slot with one materialized level never advances and never blocks the loop.

    ``prepare_sweep`` warns about exactly this case; the planner must still
    terminate rather than spin looking for a slot it can advance.
    """
    valid_levels = {"text_small": [0.0, 0.5], "text_large": [0.5]}
    storage_table = {
        "text_small": {0.0: 1000, 0.5: 100},
        "text_large": {0.5: 300},
    }

    states = planner(AVAILABLE, valid_levels, storage_table)

    assert [state["text_large"] for state, _f in states] == [[0.5]] * len(states)
    assert len(states) == 2  # only text_small has a move to make


# ── --sweep-to-gold ───────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "planner", [plan_greedy_states_indexed, plan_greedy_states_direct]
)
def test_sweep_to_gold_reaches_an_empty_state(planner):
    """One step further per slot: nothing materialized, only vanilla operators left."""
    states = planner(AVAILABLE, VALID_LEVELS, STORAGE_TABLE, sweep_to_gold=True)
    final_state, final_footprint = states[-1]

    assert final_state == {"text_small": [], "text_large": []}
    assert final_footprint == 0


@pytest.mark.parametrize(
    "planner", [plan_greedy_states_indexed, plan_greedy_states_direct]
)
def test_sweep_to_gold_extends_rather_than_replaces_the_walk(planner):
    """The default walk must be a strict prefix of the extended one.

    Otherwise enabling the flag would silently renumber the steps that already
    existed, and step index is what the merged CSV and the job priorities key on.
    """
    default = planner(AVAILABLE, VALID_LEVELS, STORAGE_TABLE)
    extended = planner(AVAILABLE, VALID_LEVELS, STORAGE_TABLE, sweep_to_gold=True)

    assert extended[: len(default)] == default
    assert len(extended) == len(default) + len(AVAILABLE)  # one extra drop per slot


@pytest.mark.parametrize(
    "planner", [plan_greedy_states_indexed, plan_greedy_states_direct]
)
def test_sweep_to_gold_still_decreases_monotonically(planner):
    footprints = [
        f
        for _s, f in planner(
            AVAILABLE, VALID_LEVELS, STORAGE_TABLE, sweep_to_gold=True
        )
    ]
    assert footprints == sorted(footprints, reverse=True)
    assert len(set(footprints)) == len(footprints)


def test_empty_state_still_builds_a_usable_search_space():
    """The vanilla-only terminal state must yield a non-empty toolbox whose gold is vanilla.

    ``build_toolbox`` substitutes its defaults only for ``None``, so an explicit empty
    active list means "no compressed operators" rather than "all of them"; the
    ``include_vanilla`` families remain. And the profiler picks gold positionally as
    the highest-quality operator, so vanilla must still top the list.
    """
    from reasondb.evaluation.parameter_sweep import build_storage_configurator

    toolbox = build_storage_configurator([], use_indexes=False).physical_operators

    text_filters = [
        op for op in toolbox.filter_operators if type(op).__name__ == "TextQaFilter"
    ]
    assert text_filters, "vanilla-only state produced no QA filters at all"
    assert all(op.text_qa_backend.vanilla for op in text_filters)
    assert max(text_filters, key=lambda o: o.quality).text_qa_backend.vanilla


def _vanilla_filter_identifiers(cls_name, **kwargs):
    """QA-filter identifiers of the empty (no materialized baselines) search space.

    The identifier rather than a backend attribute: the text and image backends nest
    their model differently, and the identifier is what the rest of the system treats as
    an operator's identity (it is the precompute key and the simulate lookup key).
    """
    from reasondb.evaluation.parameter_sweep import build_storage_configurator

    toolbox = build_storage_configurator([], use_indexes=False, **kwargs).physical_operators
    return [
        op.get_operation_identifier()
        for op in toolbox.filter_operators
        if type(op).__name__ == cls_name
    ]


def test_small_model_vanilla_can_be_switched_off():
    """The greedy walk's empty search space is the *large* model alone.

    The small models' ``vanilla=True`` spec row is in the default suite, but the walk
    switches it off (``state_includes_small_model_vanilla``), so its terminal state holds
    one operator per modality. That makes its gold gold by construction and fixes the
    right-hand end of every operator-count curve.
    """
    from reasondb.interface.default_operator_toolbox import IMAGE_MODEL_70B, TEXT_MODEL_70B

    for cls_name, large in (
        ("TextQaFilter", TEXT_MODEL_70B),
        ("ImageQaFilter", IMAGE_MODEL_70B),
    ):
        found = _vanilla_filter_identifiers(cls_name, include_small_model_vanilla=False)
        assert len(found) == 1, f"{cls_name}: expected the large model alone, got {found}"
        assert large in found[0], found


def test_small_model_vanilla_makes_the_empty_state_a_choice_by_default():
    """What the ablation's vanilla-only arm needs, and gets from the default.

    With both models the arm's search space has no KV caches in it and is still a choice,
    so the step down from the default suite removes compression and nothing else. With
    only the large model it holds a single LLM operator per modality: the optimizer has
    nothing to cascade from, and "no KV compression" cannot be told apart from "no cheap
    proxy at all".
    """
    from reasondb.interface.default_operator_toolbox import (
        IMAGE_MODEL_8B,
        IMAGE_MODEL_70B,
        TEXT_MODEL_8B,
        TEXT_MODEL_70B,
    )

    for cls_name, small, large in (
        ("TextQaFilter", TEXT_MODEL_8B, TEXT_MODEL_70B),
        ("ImageQaFilter", IMAGE_MODEL_8B, IMAGE_MODEL_70B),
    ):
        found = _vanilla_filter_identifiers(cls_name)
        assert len(found) == 2, f"{cls_name}: expected one per model, got {found}"
        assert any(small in ident for ident in found), f"{cls_name} lost the small model: {found}"
        assert any(large in ident for ident in found), f"{cls_name} lost the large model: {found}"
        assert all(ident.endswith("-vanilla") for ident in found), found


def test_only_the_greedy_walks_switch_the_small_model_vanilla_off():
    """The predicate that scopes it, in both directions.

    ``default`` must answer True or ``plan_default_state`` stops reproducing
    ``get_default_configurator``, which is the invariant the sample_size, tuning,
    baselines and adaptive_sampling experiments rest on. Both ablation steps must too:
    step 0 *is* the default state, and step 1 is the arm that needs a choice of model.
    The greedy walks must answer False at every step - including ``greedy_to_gold``'s
    terminal step, the same ``{slot: []}`` assignment as the ablation's second arm, which
    must keep holding one operator per modality.
    """
    from reasondb.evaluation.parameter_sweep import (
        ABLATION_VANILLA_STEP,
        state_includes_small_model_vanilla,
    )

    assert state_includes_small_model_vanilla("ablation", ABLATION_VANILLA_STEP)
    assert state_includes_small_model_vanilla("ablation", 0)
    assert state_includes_small_model_vanilla("default", 0)
    # `full` too, or the widest state would be missing an operator the default suite has
    # and a comparison between the two would move in both directions at once.
    assert state_includes_small_model_vanilla("full", 0)
    # `gold` answers False with the walks, and that is the whole point of it being a
    # separate plan from the ablation's second state: `reorder_only` needs one operator
    # per modality, or its arms could cascade and the gap would pick up operator
    # selection again.
    for plan in ("greedy", "greedy_to_gold", "gold", None):
        for step in range(14):
            assert not state_includes_small_model_vanilla(plan, step), (plan, step)


def test_the_in_memory_gate_answers_for_the_same_plans():
    """Both predicates scope the *deployed* suite, so they must agree on which plans
    reproduce it. They differ from each other only in what they gate, never in where.

    ``full`` matters here even though the answer is inert while
    ``DEFAULT_*_IN_MEMORY_DIRECT`` are empty: the tables substitute an ``-in-memory``
    spelling for the disk one, so a False answer would have ``full`` hold ``...-cr0.8``
    where ``default`` holds ``...-cr0.8-in-memory`` once they are populated, and ``full``
    would silently stop containing ``default``.
    """
    from reasondb.evaluation.parameter_sweep import (
        state_includes_in_memory,
        state_includes_small_model_vanilla,
    )

    for plan in ("default", "ablation", "full", "greedy", "greedy_to_gold", None):
        for step in range(3):
            assert state_includes_in_memory(plan, step) == (
                state_includes_small_model_vanilla(plan, step)
            ), (plan, step)


# ── plan_default_state ───────────────────────────────────────────────────────
#
# The state plan the sample_size and tuning experiments pin. Unlike the greedy walk it
# is not a function of the storage table's ordering at all - it names the baselines the
# *default configurator* activates - so it needs the real spec-table models rather than
# the synthetic fixture above.


def _real_text_slots():
    from reasondb.evaluation.parameter_sweep import ModelSlot
    from reasondb.interface.default_operator_toolbox import (
        TEXT_MODEL_8B,
        TEXT_MODEL_70B,
    )

    return [
        ModelSlot("text_small", "text", TEXT_MODEL_8B, large=False),
        ModelSlot("text_large", "text", TEXT_MODEL_70B, large=True),
    ]


def _real_text_fixture():
    """Both text slots with their whole ratio grid materialized, and a storage table.

    The maximal case, which is what every plan below is read against: the default suite
    is a subset of it, the greedy walk starts from it, and ``full`` *is* it.
    """
    valid = {"text_small": [0.0, 0.5, 0.8], "text_large": [0.3, 0.6, 0.8]}
    table = {
        "text_small": {0.0: 1000, 0.5: 400, 0.8: 100},
        "text_large": {0.3: 900, 0.6: 500, 0.8: 200},
    }
    return _real_text_slots(), valid, table


def test_default_state_is_one_state_naming_the_default_baselines():
    from reasondb.evaluation.parameter_sweep import plan_default_state
    from reasondb.interface.default_operator_toolbox import DEFAULT_TEXT_ACTIVE_DIRECT

    slots = _real_text_slots()
    valid = {"text_small": [0.0, 0.5, 0.8], "text_large": [0.3, 0.6, 0.8]}
    table = {
        "text_small": {0.0: 1000, 0.5: 400, 0.8: 100},
        "text_large": {0.3: 900, 0.6: 500, 0.8: 200},
    }

    states = plan_default_state(slots, valid, table, use_indexes=False)
    assert len(states) == 1, "this plan is a single point, not a walk"
    state, footprint = states[0]

    expected = {"text_small": [], "text_large": []}
    for model, cr in DEFAULT_TEXT_ACTIVE_DIRECT:
        key = "text_small" if model == slots[0].model else "text_large"
        expected[key].append(cr)
    assert state == {k: sorted(v) for k, v in expected.items()}
    assert footprint == sum(table[k][cr] for k, crs in state.items() for cr in crs)


def test_the_default_state_is_not_a_greedy_step_even_where_its_levels_line_up():
    """Why this is a state *plan* rather than "the greedy step with N operators".

    The default names each text model's most compressed level alone - 8B cr 0.8, 70B
    cr 0.8 - which is exactly where the walk halts, so the *levels* of the
    default state and of the walk's terminal state agree, whatever the storage table says.
    Both tables below differ only in the walk's order, and both end there.

    That agreement is not what makes the plan a plan, and this test exists to keep the two
    apart. The default's *operators* are on no greedy step: it carries the small models'
    vanilla operators and no greedy state does, so the walk's terminal state profiles one
    operator per modality fewer than the suite whose levels it shares.

    The levels need not agree in general: a default naming one *interior* level per model
    (e.g. 8B cr 0.5, 70B cr 0.6) is a state no greedy step can hold - the walk retains a
    suffix of each slot's level list, so a state containing cr 0.5 also contains cr 0.8.
    Hence a plan named after what it reproduces rather than after where it lands.
    """
    from reasondb.evaluation.parameter_sweep import (
        build_storage_configurator,
        plan_default_state,
        plan_greedy_states_direct,
        state_includes_in_memory,
        state_includes_small_model_vanilla,
    )

    slots = _real_text_slots()
    valid = {"text_small": [0.0, 0.5, 0.8], "text_large": [0.3, 0.6, 0.8]}

    # The small model's middling cache is cheap, so the walk spends its early steps on
    # the large slot; below, it is nearly as big as the uncompressed one and goes first.
    # The walk ends in the same place either way, which is where the default is.
    cheap_middle = {
        "text_small": {0.0: 1000, 0.5: 400, 0.8: 100},
        "text_large": {0.3: 900, 0.6: 500, 0.8: 200},
    }
    dear_middle = {
        "text_small": {0.0: 1000, 0.5: 900, 0.8: 100},
        "text_large": {0.3: 800, 0.6: 500, 0.8: 200},
    }

    for table in (cheap_middle, dear_middle):
        (default, _f) = plan_default_state(slots, valid, table, use_indexes=False)[0]
        assert default == {"text_small": [0.8], "text_large": [0.8]}
        greedy = [s for s, _ in plan_greedy_states_direct(slots, valid, table)]
        # The levels agree with the walk's last state and with no earlier one: a state
        # holding cr 0.5 holds cr 0.8 too, so nothing before the end has one level a slot.
        assert greedy[-1] == default
        assert [s for s in greedy if s == default] == [default]

    active = [(s, cr) for s in slots for cr in default[s.key]]

    def text_filters(**kwargs):
        toolbox = build_storage_configurator(
            active, use_indexes=False, **kwargs
        ).physical_operators
        return {
            op.get_operation_identifier()
            for op in toolbox.filter_operators
            if type(op).__name__ == "TextQaFilter"
        }

    def filters_for(plan, step_idx):
        # Both gates, resolved the way run_state and the coordinator resolve them — from
        # the (plan name, step index) a job spec carries. Passing only one of them would
        # leave the other at its default and quietly compare two default-suite states.
        return text_filters(
            include_small_model_vanilla=state_includes_small_model_vanilla(
                plan, step_idx
            ),
            include_in_memory=state_includes_in_memory(plan, step_idx),
        )

    # Held at the same levels, the two gates differ in the small model's vanilla operator
    # alone. The in-memory gate contributes nothing while the default suite serves
    # everything from disk, since a True answer then resolves to an empty list.
    small_8b = "TextQaFilter-LLMTextQABackend-meta-llama/Llama-3.1-8B-Instruct"
    as_default = filters_for("default", 0)
    as_greedy_step = filters_for("greedy", 6)
    assert as_default - as_greedy_step == {f"{small_8b}-cr0.0-vanilla"}
    assert as_greedy_step - as_default == set()
    assert not [i for i in as_default if "-in-memory" in i]


def test_default_state_drops_a_baseline_that_is_not_materialized(caplog):
    """Degrade to what is on disk, rather than asserting deep inside a worker."""
    import logging

    from reasondb.evaluation.parameter_sweep import plan_default_state

    slots = _real_text_slots()
    # Only the 8B's default cache exists; nothing for the 70B at all.
    valid = {"text_small": [0.8], "text_large": []}
    table = {"text_small": {0.8: 100}, "text_large": {}}

    with caplog.at_level(logging.WARNING):
        (state, footprint) = plan_default_state(slots, valid, table, use_indexes=False)[0]

    assert state == {"text_small": [0.8], "text_large": []}
    assert footprint == 100
    assert "not materialized" in caplog.text
    # A slot left with nothing keeps only its vanilla operators, which is a valid state -
    # the same one the gold-only terminal step reaches.
    assert "only its vanilla operators" in caplog.text


# ── plan_full_state ──────────────────────────────────────────────────────────
#
# The widest state the benchmark has, as a single point: the greedy walk's left edge
# visited directly, which is what `mode01` compares the optimizer modes over.


def test_full_state_is_the_greedy_walks_first_state():
    """Derived from the walk rather than rebuilt, in both serving modes.

    Rebuilding it by hand would be wrong under --use-indexes: activating every level as
    its own baseline is a state the indexed walk never visits, and its footprint would
    count index bytes several baselines share.
    """
    from reasondb.evaluation.parameter_sweep import plan_full_state, plan_greedy_states

    slots, valid, table = _real_text_fixture()
    for use_indexes in (False, True):
        states = plan_full_state(slots, valid, table, use_indexes)
        assert len(states) == 1, "this plan is a single point, not a walk"
        assert states[0] == plan_greedy_states(slots, valid, table, use_indexes)[0], (
            use_indexes
        )


def test_full_state_holds_every_materialized_level_in_direct_mode():
    from reasondb.evaluation.parameter_sweep import plan_full_state

    slots, valid, table = _real_text_fixture()
    (state, footprint) = plan_full_state(slots, valid, table, use_indexes=False)[0]

    assert state == valid
    assert footprint == sum(table[k][cr] for k, crs in valid.items() for cr in crs)


def test_full_state_contains_the_default_state():
    """The property `mode01` rests on: it is the default suite plus the operators the
    default configurator does not activate, never a differently-shaped set."""
    from reasondb.evaluation.parameter_sweep import plan_default_state, plan_full_state

    slots, valid, table = _real_text_fixture()
    (full, _) = plan_full_state(slots, valid, table, use_indexes=False)[0]
    (default, _) = plan_default_state(slots, valid, table, use_indexes=False)[0]

    for key, crs in default.items():
        assert set(crs) <= set(full[key]), key
    assert full != default, "the fixture materializes more than the default suite"


def test_full_and_default_coincide_under_use_indexes():
    """Not a bug and not an accident: the indexed defaults already name each model's
    lowest ratio, which is the one baseline the whole effective menu is indexed out of,
    so there is nothing wider to ask for. An experiment comparing the two plans is a
    direct-mode experiment - under indexes it would re-run its own control."""
    from reasondb.evaluation.parameter_sweep import plan_default_state, plan_full_state

    slots, valid, table = _real_text_fixture()
    assert (
        plan_full_state(slots, valid, table, use_indexes=True)
        == plan_default_state(slots, valid, table, use_indexes=True)
    )


def test_single_state_plans_names_exactly_the_plans_that_are_one_point():
    """The constant `baselines` refuses a walk with, pinned against the plans it
    describes - so a plan that gains or loses states cannot leave it behind."""
    from reasondb.evaluation.parameter_sweep import SINGLE_STATE_PLANS, STATE_PLANS

    slots, valid, table = _real_text_fixture()
    for name, plan in STATE_PLANS.items():
        states = plan(slots, valid, table, use_indexes=False)
        assert (len(states) == 1) == (name in SINGLE_STATE_PLANS), (name, len(states))


# ── plan_gold_state ──────────────────────────────────────────────────────────
#
# `reorder_only`'s state axis, which is not an axis: one state, holding the gold operator
# alone, so that ordering is the only degree of freedom left to either of its arms.


def test_gold_plan_is_one_state_holding_no_materialized_level():
    from reasondb.evaluation.parameter_sweep import plan_gold_state

    slots, valid, table = _real_text_fixture()
    states = plan_gold_state(slots, valid, table, use_indexes=False)

    assert states == [({"text_small": [], "text_large": []}, 0)]


def test_gold_state_is_the_greedy_walks_terminal_state():
    """Same assignment, reached directly instead of after the walk - so "no compression at
    all" means one thing across the operator-count curve and this experiment."""
    from reasondb.evaluation.parameter_sweep import (
        plan_gold_state,
        plan_greedy_states_direct,
    )

    slots, valid, table = _real_text_fixture()
    to_gold = plan_greedy_states_direct(slots, valid, table, sweep_to_gold=True)

    assert plan_gold_state(slots, valid, table, use_indexes=False)[0] == to_gold[-1]


def test_gold_state_holds_one_operator_per_modality():
    """The ablation's vanilla-only state is the same *assignment* and a wider search
    space, because it takes the small model's vanilla operator too. That difference is
    what `reorder_only` cannot have: with two operators per step the optimizer could
    cascade, and its arms would differ in more than the ordering."""
    from reasondb.evaluation.parameter_sweep import (
        ABLATION_VANILLA_STEP,
        state_includes_small_model_vanilla,
    )

    assert not state_includes_small_model_vanilla("gold", 0)
    assert state_includes_small_model_vanilla("ablation", ABLATION_VANILLA_STEP)


# ── plan_ablation_states ─────────────────────────────────────────────────────
#
# The ablation's state axis: the default suite, then the vanilla-only state the greedy
# walk terminates on. Two states rather than one is what keeps the experiment's two
# `optim_global` arms from sharing a job id, an output directory and a results cache -
# see `coordinator/producers/ablation.py`.


def test_ablation_plan_is_the_default_state_then_the_vanilla_only_one():
    from reasondb.evaluation.parameter_sweep import (
        ABLATION_VANILLA_STEP,
        plan_ablation_states,
        plan_default_state,
    )

    slots, valid, table = _real_text_fixture()
    states = plan_ablation_states(slots, valid, table, use_indexes=False)

    assert len(states) == 2
    assert states[0] == plan_default_state(slots, valid, table, use_indexes=False)[0]
    # Slot keys retained with empty level lists, not dropped: the producer derives a
    # job's required capabilities from these keys, and a vanilla operator still needs
    # its model served.
    assert states[ABLATION_VANILLA_STEP] == ({"text_small": [], "text_large": []}, 0)


def test_ablation_vanilla_state_matches_the_greedy_walks_terminal_state():
    """The two plans must reach the same search space, or "no KV operators" would mean
    one thing in the ablation and another in the operator-count curve."""
    from reasondb.evaluation.parameter_sweep import (
        ABLATION_VANILLA_STEP,
        plan_ablation_states,
        plan_greedy_states_direct,
    )

    slots, valid, table = _real_text_fixture()
    ablation = plan_ablation_states(slots, valid, table, use_indexes=False)
    to_gold = plan_greedy_states_direct(slots, valid, table, sweep_to_gold=True)

    assert ablation[ABLATION_VANILLA_STEP] == to_gold[-1]


def test_ablation_plan_refuses_a_benchmark_whose_two_states_would_coincide():
    """``plan_default_state`` degrades to what is on disk with only a warning. With
    none of the default baselines materialized it degrades all the way to the
    vanilla-only state, and the ablation's first arm would silently be a second copy of
    its second - two full runs of the most expensive plan, reported as a comparison."""
    from reasondb.evaluation.parameter_sweep import plan_ablation_states

    slots = _real_text_slots()
    valid = {"text_small": [], "text_large": []}
    table = {"text_small": {}, "text_large": {}}

    with pytest.raises(AssertionError, match="identical"):
        plan_ablation_states(slots, valid, table, use_indexes=False)


# ── precompute coverage ──────────────────────────────────────────────────────
#
# What a --precompute pass records. "all" is every level on disk; a state-plan name
# records that plan's states alone, for a dataset deliberately replayed by only some of
# the experiments.


def test_precompute_coverage_all_is_every_materialized_level():
    from reasondb.evaluation.parameter_sweep import precompute_levels

    slots, valid, table = _real_text_fixture()
    assert (
        precompute_levels("all", slots, valid, table, use_indexes=False) == valid
    )


def test_precompute_coverage_greedy_walks_equal_all():
    """A greedy walk starts from every level on disk, so its union is that whole set.
    Spellable anyway, so the flag's vocabulary is the state plans' own."""
    from reasondb.evaluation.parameter_sweep import precompute_levels

    slots, valid, table = _real_text_fixture()
    for plan in ("greedy", "greedy_to_gold"):
        assert (
            precompute_levels(plan, slots, valid, table, use_indexes=False) == valid
        ), plan


def test_precompute_coverage_default_and_ablation_are_the_same_recording():
    """The ablation's second state is the vanilla-only one, and vanilla is recorded
    unconditionally (build_precompute_configurator) - so it contributes no *level* the
    default state does not. If that ever stops holding, the two names stop being
    interchangeable and the flag's help text is wrong."""
    from reasondb.evaluation.parameter_sweep import (
        plan_default_state,
        precompute_levels,
    )

    slots, valid, table = _real_text_fixture()
    default = precompute_levels("default", slots, valid, table, use_indexes=False)
    ablation = precompute_levels("ablation", slots, valid, table, use_indexes=False)

    assert default == ablation
    assert default == plan_default_state(slots, valid, table, use_indexes=False)[0][0]
    # And it is a real narrowing on this fixture: three of the six levels on disk are
    # not what the default suite activates.
    assert default != valid


def test_precompute_coverage_narrows_the_recorded_operators_but_keeps_vanilla():
    """The point of the flag, at the level the store is actually keyed on: fewer
    compressed operators, both models' vanilla rows still there."""
    from reasondb.evaluation.parameter_sweep import build_precompute_configurator

    slots, valid, table = _real_text_fixture()

    def identifiers(**kwargs):
        configurator = build_precompute_configurator(
            slots, valid, use_indexes=False, **kwargs
        )
        return {
            op.get_operation_identifier() for op in configurator.physical_operators
        }

    wide = identifiers()
    narrow = identifiers(coverage="ablation", storage_table=table)

    assert narrow < wide
    vanilla = {i for i in wide if i.endswith("-vanilla")}
    assert vanilla and vanilla <= narrow, (
        "both model sizes' vanilla operators are recorded at every coverage - the "
        "ablation's second arm and every silver pass need them"
    )


def test_precompute_coverage_needs_the_storage_table_to_plan_states():
    from reasondb.evaluation.parameter_sweep import build_precompute_configurator

    slots, valid, _table = _real_text_fixture()
    with pytest.raises(AssertionError, match="storage table"):
        build_precompute_configurator(slots, valid, use_indexes=False, coverage="default")


# ── plan_kv_operator_states ──────────────────────────────────────────────────
#
# The ``kv_operator`` state axis: the vanilla-only state, then one state per materialized level
# holding the gold model and that level alone. Where the greedy walk prices a search
# space, this prices one operator - so what these tests pin is that a state really does
# hold exactly one, and that step 0 is the arm the rest are read against.


def test_kv_operator_plan_is_the_vanilla_state_then_one_level_each():
    from reasondb.evaluation.parameter_sweep import plan_kv_operator_states

    states = plan_kv_operator_states(
        AVAILABLE, VALID_LEVELS, STORAGE_TABLE, use_indexes=False
    )

    assert states == [
        ({"text_small": [], "text_large": []}, 0),
        ({"text_small": [0.0], "text_large": []}, 1000),
        ({"text_small": [0.3], "text_large": []}, 400),
        ({"text_small": [0.5], "text_large": []}, 100),
        ({"text_small": [], "text_large": [0.0]}, 800),
        ({"text_small": [], "text_large": [0.5]}, 300),
    ]


def test_every_kv_operator_state_but_the_first_holds_exactly_one_level():
    """The claim the experiment rests on, as arithmetic over the states themselves."""
    from reasondb.evaluation.parameter_sweep import (
        KV_OPERATOR_VANILLA_STEP,
        plan_kv_operator_states,
    )

    states = plan_kv_operator_states(
        AVAILABLE, VALID_LEVELS, STORAGE_TABLE, use_indexes=False
    )
    for step, (state, _footprint) in enumerate(states):
        retained = sum(len(crs) for crs in state.values())
        assert retained == (0 if step == KV_OPERATOR_VANILLA_STEP else 1), (step, state)


def test_a_kv_operator_footprint_is_that_one_cache():
    """Not a sum over a state, which is what every other plan's footprint is - so a
    sweep row reports one operator's disk cost directly."""
    from reasondb.evaluation.parameter_sweep import plan_kv_operator_states

    states = plan_kv_operator_states(
        AVAILABLE, VALID_LEVELS, STORAGE_TABLE, use_indexes=False
    )
    for state, footprint in states[1:]:
        [(key, [cr])] = [(k, crs) for k, crs in state.items() if crs]
        assert footprint == STORAGE_TABLE[key][cr]


def test_kv_operator_step_zero_is_the_gold_states_assignment():
    """The same `{slot: []}` assignment `plan_gold_state` and the ablation's second arm
    name. The three differ in the *search space* they build from it, not in the state -
    which is what `state_includes_small_model_vanilla` decides."""
    from reasondb.evaluation.parameter_sweep import (
        KV_OPERATOR_VANILLA_STEP,
        plan_gold_state,
        plan_kv_operator_states,
    )

    states = plan_kv_operator_states(
        AVAILABLE, VALID_LEVELS, STORAGE_TABLE, use_indexes=False
    )
    gold = plan_gold_state(AVAILABLE, VALID_LEVELS, STORAGE_TABLE, use_indexes=False)

    assert states[KV_OPERATOR_VANILLA_STEP] == gold[0]


def test_kv_operator_takes_the_small_model_vanilla_at_step_zero_only():
    """What makes step 0 the ablation's arm 2 and every later step a two-operator space.

    Both halves matter. Answering False at step 0 would leave the reference arm with one
    operator per modality, which is `reorder_only`'s state and not a thing any deployment
    runs; answering True later would put three LLM operators in a state the experiment
    describes as holding two.
    """
    from reasondb.evaluation.parameter_sweep import (
        ABLATION_VANILLA_STEP,
        KV_OPERATOR_VANILLA_STEP,
        state_includes_small_model_vanilla,
    )

    assert state_includes_small_model_vanilla("kv_operator", KV_OPERATOR_VANILLA_STEP)
    assert state_includes_small_model_vanilla("ablation", ABLATION_VANILLA_STEP)
    for step in range(1, 14):
        assert not state_includes_small_model_vanilla("kv_operator", step), step


def test_no_kv_operator_state_is_served_from_memory():
    """A storage experiment measures what is on disk. The predicate therefore disagrees
    with its sibling at step 0 - and cannot act on the disagreement, since a state that
    materializes nothing has no baseline to hold in RAM."""
    from reasondb.evaluation.parameter_sweep import state_includes_in_memory

    for step in range(14):
        assert not state_includes_in_memory("kv_operator", step), step


def test_kv_operator_counts_two_llm_operators_at_every_step():
    """Two everywhere, by two different routes: both vanilla operators at step 0, the
    gold one plus the retained level afterwards. That is what the row reports, and what
    a figure labels its operator axis with."""
    from reasondb.evaluation.parameter_sweep import (
        count_llm_operators,
        plan_kv_operator_states,
        state_includes_small_model_vanilla,
    )

    states = plan_kv_operator_states(
        AVAILABLE, VALID_LEVELS, STORAGE_TABLE, use_indexes=False
    )
    for step, (state, _footprint) in enumerate(states):
        assert count_llm_operators(
            state,
            AVAILABLE,
            state_includes_small_model_vanilla("kv_operator", step),
        ) == 2, (step, state)


def test_precompute_coverage_for_kv_operator_equals_all():
    """It visits every materialized level as a state of its own, so a store narrowed for
    it is not narrowed at all - and one narrowed for another experiment cannot serve it."""
    from reasondb.evaluation.parameter_sweep import precompute_levels

    slots, valid, table = _real_text_fixture()
    assert precompute_levels(
        "kv_operator", slots, valid, table, use_indexes=False
    ) == precompute_levels("all", slots, valid, table, use_indexes=False)


def test_kv_operator_warns_under_use_indexes(caplog):
    """One materialized level then serves every ratio above it, so the states stop being
    single operators. Warned rather than refused: `precompute_levels` calls every plan,
    and a recording wider than this experiment needs is not an error."""
    import logging

    from reasondb.evaluation.parameter_sweep import plan_kv_operator_states

    with caplog.at_level(logging.WARNING):
        plan_kv_operator_states(AVAILABLE, VALID_LEVELS, STORAGE_TABLE, use_indexes=True)

    assert any("direct-mode" in record.message for record in caplog.records)


# ── plan_kv_operator_pairs_states ────────────────────────────────────────────
#
# The ``kv_operator_pairs`` state axis: ``kv_operator``'s rule with "one compressed
# operator per state" widened to
# "one per modality per state", levels matched by model size and compression rank. What
# these tests pin is that a multimodal state really holds one level in each modality, and
# that a single-modality benchmark gets ``kv_operator``'s states back unchanged.

MM_AVAILABLE = [
    ModelSlot("text_small", "text", "fake/text-small", large=False),
    ModelSlot("text_large", "text", "fake/text-large", large=True),
    ModelSlot("image_small", "image", "fake/image-small", large=False),
    ModelSlot("image_large", "image", "fake/image-large", large=True),
]

#: ecommerce's own grids: disjoint ratios per modality, three levels per slot.
MM_VALID_LEVELS = {
    "text_small": [0.0, 0.5, 0.8],
    "text_large": [0.3, 0.6, 0.8],
    "image_small": [0.0, 0.5, 0.9],
    "image_large": [0.5, 0.9, 0.99],
}

MM_STORAGE_TABLE = {
    "text_small": {0.0: 27, 0.5: 13, 0.8: 5},
    "text_large": {0.3: 47, 0.6: 27, 0.8: 13},
    "image_small": {0.0: 288, 0.5: 144, 0.9: 29},
    "image_large": {0.5: 2880, 0.9: 576, 0.99: 56},
}


def test_kv_operator_pairs_on_one_modality_is_the_kv_operator_plan():
    """What lets ``kv_operator_pairs`` describe one rule over every benchmark: with a
    single slot per size there is nothing to pair, and the states - and so the step
    indices - are ``kv_operator``'s."""
    from reasondb.evaluation.parameter_sweep import (
        plan_kv_operator_pairs_states,
        plan_kv_operator_states,
    )

    assert plan_kv_operator_pairs_states(
        AVAILABLE, VALID_LEVELS, STORAGE_TABLE, use_indexes=False
    ) == plan_kv_operator_states(AVAILABLE, VALID_LEVELS, STORAGE_TABLE, use_indexes=False)


def test_kv_operator_pairs_matches_size_and_rank_on_a_multimodal_benchmark():
    from reasondb.evaluation.parameter_sweep import plan_kv_operator_pairs_states

    states = plan_kv_operator_pairs_states(
        MM_AVAILABLE, MM_VALID_LEVELS, MM_STORAGE_TABLE, use_indexes=False
    )

    empty = {"text_small": [], "text_large": [], "image_small": [], "image_large": []}
    assert states == [
        (empty, 0),
        ({**empty, "text_small": [0.0], "image_small": [0.0]}, 27 + 288),
        ({**empty, "text_small": [0.5], "image_small": [0.5]}, 13 + 144),
        ({**empty, "text_small": [0.8], "image_small": [0.9]}, 5 + 29),
        ({**empty, "text_large": [0.3], "image_large": [0.5]}, 47 + 2880),
        ({**empty, "text_large": [0.6], "image_large": [0.9]}, 27 + 576),
        ({**empty, "text_large": [0.8], "image_large": [0.99]}, 13 + 56),
    ]


def test_every_kv_operator_pairs_state_holds_one_level_per_modality():
    """The claim the experiment rests on, as arithmetic over the states themselves."""
    from reasondb.evaluation.parameter_sweep import (
        KV_OPERATOR_VANILLA_STEP,
        plan_kv_operator_pairs_states,
    )

    modality = {slot.key: slot.modality for slot in MM_AVAILABLE}
    states = plan_kv_operator_pairs_states(
        MM_AVAILABLE, MM_VALID_LEVELS, MM_STORAGE_TABLE, use_indexes=False
    )
    for step, (state, _footprint) in enumerate(states):
        held = [modality[key] for key, crs in state.items() for _cr in crs]
        expected = [] if step == KV_OPERATOR_VANILLA_STEP else ["text", "image"]
        assert sorted(held) == sorted(expected), (step, state)


def test_kv_operator_pairs_counts_two_llm_operators_per_modality_at_every_step():
    """Both vanilla operators at step 0, the gold one plus that modality's level after -
    in each modality, which is exactly what the ``kv_operator`` plan's text-cache states on
    ecommerce are not (their image side holds the gold operator alone)."""
    from reasondb.evaluation.parameter_sweep import (
        count_llm_operators,
        plan_kv_operator_pairs_states,
        state_includes_small_model_vanilla,
    )

    states = plan_kv_operator_pairs_states(
        MM_AVAILABLE, MM_VALID_LEVELS, MM_STORAGE_TABLE, use_indexes=False
    )
    for step, (state, _footprint) in enumerate(states):
        assert count_llm_operators(
            state,
            MM_AVAILABLE,
            state_includes_small_model_vanilla("kv_operator_pairs", step),
        ) == 4, (step, state)


def test_kv_operator_pairs_leaves_unmatched_levels_out_and_says_so(caplog):
    """Which ratio a level without a same-rank partner should share a state with is a
    choice, not a fact of the recording, so the plan makes none."""
    import logging

    from reasondb.evaluation.parameter_sweep import plan_kv_operator_pairs_states

    valid = {**MM_VALID_LEVELS, "image_small": [0.0, 0.5]}
    with caplog.at_level(logging.WARNING):
        states = plan_kv_operator_pairs_states(
            MM_AVAILABLE, valid, MM_STORAGE_TABLE, use_indexes=False
        )

    small = [state for state, _ in states[1:] if state["text_small"]]
    assert [state["text_small"] for state in small] == [[0.0], [0.5]]
    assert any("0.8" in record.message and "text_small" in record.message
               for record in caplog.records)


def test_kv_operator_pairs_shares_kv_operators_predicates():
    """Step 0 is the same reference arm, and nothing is served from RAM - for the reason
    the two predicates give for `kv_operator`, which this plan inherits wholesale."""
    from reasondb.evaluation.parameter_sweep import (
        KV_OPERATOR_VANILLA_STEP,
        state_includes_in_memory,
        state_includes_small_model_vanilla,
    )

    assert state_includes_small_model_vanilla("kv_operator_pairs", KV_OPERATOR_VANILLA_STEP)
    for step in range(1, 8):
        assert not state_includes_small_model_vanilla("kv_operator_pairs", step), step
    for step in range(8):
        assert not state_includes_in_memory("kv_operator_pairs", step), step


def test_precompute_coverage_for_kv_operator_pairs_equals_all():
    """Every level is visited when the ranks match, so ``kv_operator_pairs`` replays
    ``kv_operator``'s store."""
    from reasondb.evaluation.parameter_sweep import precompute_levels

    assert precompute_levels(
        "kv_operator_pairs", MM_AVAILABLE, MM_VALID_LEVELS, MM_STORAGE_TABLE,
        use_indexes=False,
    ) == precompute_levels(
        "all", MM_AVAILABLE, MM_VALID_LEVELS, MM_STORAGE_TABLE, use_indexes=False
    )


# ── kv_operator_marginal ─────────────────────────────────────────────────────
#
# The kvop01 reading: ``kv_operator_pairs``' states, with the small models' vanilla operators kept at
# every step, so a state ADDS a KV operator per modality to the reference suite instead of
# replacing its small model with one. Same states, different search space - which is why
# what these tests pin is the predicate and the operator count, not a state list.


def test_kv_operator_marginal_visits_the_same_states_as_the_pairs_plan():
    """A plan name rather than a flag, because a job spec carries the name and the worker
    rebuilds the search space from it - but the states themselves are
    ``kv_operator_pairs``'."""
    from reasondb.evaluation.parameter_sweep import (
        STATE_PLANS,
        plan_kv_operator_pairs_states,
    )

    assert STATE_PLANS["kv_operator_marginal"] is plan_kv_operator_pairs_states


def test_kv_operator_marginal_keeps_the_small_model_vanilla_at_every_step():
    """The whole difference between ``kv_operator_marginal`` and ``kv_operator_pairs``.
    Answering False anywhere would make the gap remove the uncompressed small model along
    with adding the cache, which is the reading the replacing plans already have."""
    from reasondb.evaluation.parameter_sweep import (
        KV_OPERATOR_VANILLA_STEP,
        state_includes_in_memory,
        state_includes_small_model_vanilla,
    )

    for step in range(8):
        assert state_includes_small_model_vanilla("kv_operator_marginal", step), step
        assert not state_includes_in_memory("kv_operator_marginal", step), step
    assert state_includes_small_model_vanilla(
        "kv_operator_marginal", KV_OPERATOR_VANILLA_STEP
    )


def test_kv_operator_marginal_step_zero_is_the_ablations_vanilla_arm():
    """Shared with ``kv_operator`` and ``kv_operator_pairs``, which is what lets the three be read against each
    other - and with abl01's arm 2, the same `{slot: []}` assignment and, since the
    predicate answers the same way, the same search space built from it."""
    from reasondb.evaluation.parameter_sweep import (
        ABLATION_VANILLA_STEP,
        KV_OPERATOR_VANILLA_STEP,
        plan_gold_state,
        plan_kv_operator_pairs_states,
        state_includes_small_model_vanilla,
    )

    states = plan_kv_operator_pairs_states(
        MM_AVAILABLE, MM_VALID_LEVELS, MM_STORAGE_TABLE, use_indexes=False
    )
    gold = plan_gold_state(
        MM_AVAILABLE, MM_VALID_LEVELS, MM_STORAGE_TABLE, use_indexes=False
    )

    assert states[KV_OPERATOR_VANILLA_STEP] == gold[0]
    assert state_includes_small_model_vanilla(
        "kv_operator_marginal", KV_OPERATOR_VANILLA_STEP
    ) == state_includes_small_model_vanilla("ablation", ABLATION_VANILLA_STEP)


def test_kv_operator_marginal_counts_three_llm_operators_per_modality_when_cached():
    """Two at step 0 - both vanilla operators - and three afterwards: the reference suite
    plus the retained level. ``kv_operator_pairs`` holds two throughout, having swapped one for the other."""
    from reasondb.evaluation.parameter_sweep import (
        KV_OPERATOR_VANILLA_STEP,
        count_llm_operators,
        plan_kv_operator_pairs_states,
        state_includes_small_model_vanilla,
    )

    states = plan_kv_operator_pairs_states(
        MM_AVAILABLE, MM_VALID_LEVELS, MM_STORAGE_TABLE, use_indexes=False
    )
    for step, (state, _footprint) in enumerate(states):
        counted = count_llm_operators(
            state,
            MM_AVAILABLE,
            state_includes_small_model_vanilla("kv_operator_marginal", step),
        )
        expected = 4 if step == KV_OPERATOR_VANILLA_STEP else 6  # two modalities
        assert counted == expected, (step, state)


def test_kv_operator_marginal_adds_to_a_single_modality_benchmark_too():
    """It runs every dataset: what an operator adds is not a question about multimodality,
    and on one modality the states are ``kv_operator``'s with one more operator in the
    space."""
    from reasondb.evaluation.parameter_sweep import (
        count_llm_operators,
        plan_kv_operator_pairs_states,
        plan_kv_operator_states,
        state_includes_small_model_vanilla,
    )

    states = plan_kv_operator_pairs_states(
        AVAILABLE, VALID_LEVELS, STORAGE_TABLE, use_indexes=False
    )
    assert states == plan_kv_operator_states(
        AVAILABLE, VALID_LEVELS, STORAGE_TABLE, use_indexes=False
    )
    for step, (state, _footprint) in enumerate(states[1:], start=1):
        assert count_llm_operators(
            state,
            AVAILABLE,
            state_includes_small_model_vanilla("kv_operator_marginal", step),
        ) == 3, (step, state)


def test_precompute_coverage_for_kv_operator_marginal_equals_all():
    from reasondb.evaluation.parameter_sweep import precompute_levels

    assert precompute_levels(
        "kv_operator_marginal", MM_AVAILABLE, MM_VALID_LEVELS, MM_STORAGE_TABLE,
        use_indexes=False,
    ) == precompute_levels(
        "all", MM_AVAILABLE, MM_VALID_LEVELS, MM_STORAGE_TABLE, use_indexes=False
    )
