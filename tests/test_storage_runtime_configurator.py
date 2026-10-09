"""``reasondb.evaluation.parameter_sweep.build_storage_configurator`` must be grounded in
the default configurator's operator suite (same operator families, quality scores,
fake costs — see ``reasondb.interface.default_operator_toolbox``), not a bespoke
continuous quality/cost proxy. These tests pin that: no ``RawTextQaFilter`` (or any
operator family the default configurator doesn't use) survives, the real
quality/fake_cost table is used, and a caller cannot silently request a
(model, cr) baseline the default configurator has no entry for.
"""

import pytest

try:
    from reasondb.evaluation.parameter_sweep import (
        ModelSlot,
        build_storage_configurator,
    )
except ImportError:
    pytest.skip("parameter_sweep deps not installed", allow_module_level=True)

from reasondb.interface.default_operator_toolbox import (
    IMAGE_MODEL_8B,
    IMAGE_MODEL_70B,
    TEXT_MODEL_8B,
    TEXT_MODEL_70B,
    TEXT_SPECS,
)

TEXT_SMALL = ModelSlot("text_small", "text", TEXT_MODEL_8B, large=False)
TEXT_LARGE = ModelSlot("text_large", "text", TEXT_MODEL_70B, large=True)
IMAGE_SMALL = ModelSlot("image_small", "image", IMAGE_MODEL_8B, large=False)
IMAGE_LARGE = ModelSlot("image_large", "image", IMAGE_MODEL_70B, large=True)


def _spec(specs, model, effective_cr):
    return next(s for s in specs if s.model == model and s.effective_cr == effective_cr)


def _text_qa_filters(tb, materialized_only=True):
    """The ``TextQaFilter``s in a toolbox.

    ``filter_operators`` always also carries the fixed TraditionalFilter/
    ImageSimilarityFilter baselines build_toolbox includes regardless of which KV
    baselines are active. ``materialized_only`` additionally drops the vanilla
    baselines, which are present in every configurator (they are the gold operators)
    but are not governed by the ``active`` assignments these tests are about.
    """
    ops = [op for op in tb.filter_operators if type(op).__name__ == "TextQaFilter"]
    if materialized_only:
        ops = [op for op in ops if not op.text_qa_backend.vanilla]
    return ops


def test_join_predicates_use_extract_families_not_raw_qa():
    """The sweep must not fall back to RawTextQaFilter for joins."""
    configurator = build_storage_configurator([(TEXT_SMALL, 0.0)], use_indexes=True)
    predicate_types = {type(op).__name__ for op in configurator.physical_operators.join_predicates}
    assert "RawTextQaFilter" not in predicate_types
    # Vanilla baselines bring the image families in even for a text-only assignment:
    # they are unconditional, exactly as in the default suite.
    assert {"ExtractAndMatchFilter", "ExtractAndQaFilter"} <= predicate_types


def test_indexed_baseline_unlocks_every_effective_ratio_ge_materialized():
    """Under --use-indexes, one materialized baseline serves every higher spec ratio."""
    configurator = build_storage_configurator([(TEXT_SMALL, 0.0)], use_indexes=True)
    tb = configurator.physical_operators
    filters = _text_qa_filters(tb)
    effective_ratios = {op.text_qa_backend.effective_compression_ratio for op in filters}
    expected = {s.effective_cr for s in TEXT_SPECS if s.model == TEXT_MODEL_8B and not s.vanilla}
    assert effective_ratios == expected
    for op in filters:
        assert op.text_qa_backend.materialized_compression_ratio == 0.0


def test_direct_baseline_serves_only_its_own_ratio():
    """Without indexing, a materialized baseline exposes exactly one effective ratio."""
    configurator = build_storage_configurator([(TEXT_SMALL, 0.5)], use_indexes=False)
    filters = _text_qa_filters(configurator.physical_operators)
    assert len(filters) == 1
    backend = filters[0].text_qa_backend
    assert backend.effective_compression_ratio == 0.5
    assert backend.materialized_compression_ratio == 0.5


def test_quality_and_fake_cost_match_default_configurator_spec_table():
    """Values must come from the default configurator's table, not a continuous proxy."""
    configurator = build_storage_configurator([(TEXT_SMALL, 0.8)], use_indexes=False)
    spec = _spec(TEXT_SPECS, TEXT_MODEL_8B, 0.8)
    filter_op = _text_qa_filters(configurator.physical_operators)[0]
    assert filter_op.quality == spec.filter_quality
    assert filter_op._fake_cost == spec.fake_cost


def test_image_slot_builds_image_operator_families():
    configurator = build_storage_configurator([(IMAGE_SMALL, 0.0)], use_indexes=True)
    tb = configurator.physical_operators
    assert any(type(op).__name__ == "ImageQaFilter" for op in tb.filter_operators)
    assert any(type(op).__name__ == "ImageQaExtract" for op in tb.extract_operators)
    predicate_types = {type(op).__name__ for op in tb.join_predicates}
    assert {"ExtractAndMatchImageFilter", "ExtractAndQaImageFilter"} <= predicate_types


def test_unknown_baseline_raises():
    """A (model, cr) pair the default configurator's spec table doesn't define must fail fast."""
    with pytest.raises(AssertionError):
        build_storage_configurator([(TEXT_SMALL, 0.42)], use_indexes=True)


def test_vanilla_operators_are_included_as_gold():
    """The sweep's search space must match the default suite's, vanilla included.

    Excluding vanilla (on the grounds that it bypasses materialization) would silently
    break every accuracy number the sweep produces: the profiler derives a query's labels
    from the highest-quality operator in the search space, so removing vanilla would not
    remove gold, it would hand the role to the best *compressed* baseline. Storage accounting is unaffected either way -
    vanilla specs are filtered out of the storage grid and have no cache to measure.
    """
    configurator = build_storage_configurator(
        [(TEXT_SMALL, 0.0), (TEXT_LARGE, 0.3), (IMAGE_SMALL, 0.0), (IMAGE_LARGE, 0.5)],
        use_indexes=True,
    )
    tb = configurator.physical_operators

    text_vanilla = [
        op for op in tb.filter_operators
        if type(op).__name__ == "TextQaFilter" and op.text_qa_backend.vanilla
    ]
    image_vanilla = [
        op for op in tb.filter_operators
        if type(op).__name__ == "ImageQaFilter"
        and op.image_qa_backend.vision_model.vanilla
    ]
    assert text_vanilla, "no vanilla text filter in the sweep's search space"
    assert image_vanilla, "no vanilla image filter in the sweep's search space"

    # And it really is the top of the pile, i.e. what the profiler will use as gold.
    for family, vanilla_ops in (("TextQaFilter", text_vanilla), ("ImageQaFilter", image_vanilla)):
        same_family = [op for op in tb.filter_operators if type(op).__name__ == family]
        assert max(same_family, key=lambda o: o.quality) in vanilla_ops


def test_search_space_matches_the_default_suite():
    """With every baseline active, the sweep's operator set is the default suite's."""
    from reasondb.interface.default_operator_toolbox import (
        DEFAULT_IMAGE_ACTIVE_DIRECT,
        DEFAULT_TEXT_ACTIVE_DIRECT,
        build_toolbox,
    )

    active = [
        (slot, cr)
        for slot, pairs in (
            (TEXT_SMALL, DEFAULT_TEXT_ACTIVE_DIRECT),
            (TEXT_LARGE, DEFAULT_TEXT_ACTIVE_DIRECT),
            (IMAGE_SMALL, DEFAULT_IMAGE_ACTIVE_DIRECT),
            (IMAGE_LARGE, DEFAULT_IMAGE_ACTIVE_DIRECT),
        )
        for model, cr in pairs
        if model == slot.model
    ]
    sweep = build_storage_configurator(active, use_indexes=False).physical_operators
    default = build_toolbox(use_indexes=False)

    for family in ("filter_operators", "extract_operators", "join_predicates"):
        sweep_ids = {o.get_operation_identifier() for o in getattr(sweep, family)}
        default_ids = {o.get_operation_identifier() for o in getattr(default, family)}
        assert sweep_ids == default_ids, f"{family} differs from the default suite"


def test_quality_falls_as_compression_rises_within_a_model_family():
    """More compression must be cheaper *and* worse, on every quality column.

    If any quality column ran backwards (rising with the compression ratio while
    ``fake_cost`` and the other quality columns fall), the most compressed cache would
    become one of the best operators in its family. Since the profiler derives gold labels from the highest-quality operator
    and optimizers reach for quality first, an inverted column silently points the whole
    system at the wrong end of the trade-off.
    """
    from reasondb.interface.default_operator_toolbox import IMAGE_SPECS, TEXT_SPECS

    for table_name, specs in (("TEXT_SPECS", TEXT_SPECS), ("IMAGE_SPECS", IMAGE_SPECS)):
        by_model = {}
        for spec in specs:
            by_model.setdefault(spec.model, []).append(spec)

        for model, entries in by_model.items():
            compressed = sorted(
                (s for s in entries if not s.vanilla), key=lambda s: s.effective_cr
            )
            for column in ("fake_cost", "filter_quality", "match_quality", "qa_quality"):
                values = [getattr(s, column) for s in compressed]
                assert values == sorted(values, reverse=True), (
                    f"{table_name}[{model}].{column} must fall as effective_cr rises "
                    f"(more compression is cheaper and worse); got {values} for ratios "
                    f"{[s.effective_cr for s in compressed]}."
                )

            # Vanilla is the family's ceiling on every quality column - it is what the
            # profiler must end up using as gold. It may *tie* the family's cr-0.0
            # baseline, and only that one: at cr 0.0 nothing is pruned, so the cache and
            # the live call are the same computation on the same tokens and claiming the
            # live one is better would be a lie (and would point the optimizer at the
            # uncached operator over an equivalent cached one). It must still strictly
            # beat every *compressed* baseline, which an inverted column would break. `test_vanilla_is_the_gold_operator_the_profiler_will_pick`
            # pins the half that matters globally: gold really is a vanilla operator.
            uncompressed = [s for s in compressed if s.effective_cr == 0.0]
            strictly_compressed = [s for s in compressed if s.effective_cr > 0.0]
            for vanilla in (s for s in entries if s.vanilla):
                for column in ("filter_quality", "match_quality", "qa_quality"):
                    best = max(getattr(s, column) for s in strictly_compressed)
                    assert getattr(vanilla, column) > best, (
                        f"{table_name}[{model}] vanilla.{column} must top every "
                        f"compressed baseline; got {getattr(vanilla, column)} vs {best}."
                    )
                    for peer in uncompressed:
                        assert getattr(vanilla, column) == getattr(peer, column), (
                            f"{table_name}[{model}] vanilla.{column} disagrees with its "
                            f"own cr-0.0 baseline ({getattr(vanilla, column)} vs "
                            f"{getattr(peer, column)}). Nothing is pruned at cr 0.0, so "
                            "the two are the same computation and must be scored alike; "
                            "the only difference between them is materialized storage."
                        )


def test_vanilla_is_the_gold_operator_the_profiler_will_pick():
    """The profiler takes the *last* operator of a step as gold, relying on
    ``UnoptimizedPhysicalPlanStep`` having quality-sorted them ascending. This pins the
    other half of that contract: the highest-quality candidate really is vanilla, for
    every operator family the default suite builds."""
    from collections import defaultdict

    from reasondb.interface.default_operator_toolbox import build_toolbox

    toolbox = build_toolbox(use_indexes=False, include_vanilla=True)
    for family in ("filter_operators", "extract_operators", "join_predicates"):
        by_interface = defaultdict(list)
        for operator in getattr(toolbox, family):
            by_interface[operator.get_llm_parameters().name].append(operator)

        for interface, operators in by_interface.items():
            kv_backed = [
                o for o in operators if "-cr" in o.get_operation_identifier()
            ]
            if not kv_backed:
                continue  # TraditionalFilter and friends have no compression axis
            best = max(kv_backed, key=lambda o: o.quality)
            assert "-vanilla" in best.get_operation_identifier(), (
                f"{interface}: highest-quality candidate is "
                f"{best.get_operation_identifier()}, not a vanilla baseline."
            )


# ── plan_default_state must reproduce get_default_configurator ───────────────


@pytest.mark.parametrize("use_indexes", [False, True])
def test_default_state_builds_exactly_the_default_configurator(use_indexes):
    """The invariant the sample_size and tuning experiments rest on.

    Those two producers pin ``state_plan="default"`` on the claim that they profile the
    default operator suite. That is only true if this holds: the state
    ``plan_default_state`` produces, fed back through ``build_storage_configurator``,
    yields the same operators as ``get_default_configurator``. Asserting it here is what
    keeps the two moving together - shrink or widen ``DEFAULT_*_ACTIVE_DIRECT`` and both
    the default suite and those experiments follow, or this fails.

    Compared on ``get_operation_identifier()``, which carries model, effective ratio,
    materialized ratio and vanilla-ness - so an operator served from a different cache
    than the default configurator would serve it from is a difference, not a match.
    """
    from reasondb.evaluation.parameter_sweep import plan_default_state
    from reasondb.interface.config import get_default_configurator

    slots = [TEXT_SMALL, TEXT_LARGE, IMAGE_SMALL, IMAGE_LARGE]
    # Everything the spec tables allow is materialized, so nothing is dropped for want
    # of a cache and the state is the default activation itself.
    from reasondb.evaluation.parameter_sweep import slot_effective_ratios

    valid = {s.key: slot_effective_ratios(s) for s in slots}
    table = {s.key: {cr: 1 for cr in valid[s.key]} for s in slots}

    (state, _footprint) = plan_default_state(slots, valid, table, use_indexes)[0]
    by_key = {s.key: s for s in slots}
    active = [(by_key[k], cr) for k, crs in state.items() for cr in crs]

    planned = build_storage_configurator(active, use_indexes).physical_operators
    default = get_default_configurator(use_indexes=use_indexes).physical_operators

    for family in ("filter_operators", "extract_operators", "join_predicates"):
        assert {
            op.get_operation_identifier() for op in getattr(planned, family)
        } == {
            op.get_operation_identifier() for op in getattr(default, family)
        }, family


# ── plan_kv_operator_states must build two operators, and which two ──────────


def _kv_operator_toolbox(state, slots, step_idx):
    from reasondb.evaluation.parameter_sweep import (
        build_storage_configurator,
        state_includes_in_memory,
        state_includes_small_model_vanilla,
    )

    by_key = {s.key: s for s in slots}
    active = [(by_key[k], cr) for k, crs in state.items() for cr in crs]
    return build_storage_configurator(
        active,
        use_indexes=False,
        include_small_model_vanilla=state_includes_small_model_vanilla(
            "kv_operator", step_idx
        ),
        include_in_memory=state_includes_in_memory("kv_operator", step_idx),
    ).physical_operators


def test_a_kv_operator_state_builds_the_gold_operator_and_one_other():
    """The experiment's whole claim about its own search space.

    Read on the filter family, where both the gold operator and the compressed one are
    ``TextQaFilter``s: exactly two, one vanilla and one at the state's own ratio. If this
    ever holds three, "the speedup of one KV operator" stops being what the figure shows.
    """
    from reasondb.evaluation.parameter_sweep import plan_kv_operator_states

    slots = [TEXT_SMALL, TEXT_LARGE]
    valid = {"text_small": [0.0, 0.5, 0.8], "text_large": [0.3, 0.6, 0.8]}
    table = {key: {cr: 1 for cr in crs} for key, crs in valid.items()}
    states = plan_kv_operator_states(slots, valid, table, use_indexes=False)

    for step_idx, (state, _footprint) in enumerate(states[1:], start=1):
        toolbox = _kv_operator_toolbox(state, slots, step_idx)
        filters = _text_qa_filters(toolbox, materialized_only=False)
        vanilla = [f for f in filters if f.text_qa_backend.vanilla]
        compressed = [f for f in filters if not f.text_qa_backend.vanilla]

        assert len(vanilla) == 1, (step_idx, state)
        assert TEXT_MODEL_70B in vanilla[0].get_operation_identifier()
        assert len(compressed) == 1, (step_idx, state)
        [(key, [cr])] = [(k, crs) for k, crs in state.items() if crs]
        assert compressed[0].text_qa_backend.effective_compression_ratio == cr
        expected = TEXT_MODEL_8B if key == "text_small" else TEXT_MODEL_70B
        assert expected in compressed[0].get_operation_identifier()


def test_kv_operator_step_zero_is_the_ablations_vanilla_only_suite():
    """Byte for byte, which is what makes abl01 a cross-check on kvop01 rather than a
    neighbouring experiment. Both build the same `{slot: []}` state with the small
    model's vanilla operator, so the reference arm is one search space with two names."""
    from reasondb.evaluation.parameter_sweep import (
        ABLATION_VANILLA_STEP,
        KV_OPERATOR_VANILLA_STEP,
        build_storage_configurator,
        state_includes_in_memory,
        state_includes_small_model_vanilla,
    )

    slots = [TEXT_SMALL, TEXT_LARGE]
    state = {slot.key: [] for slot in slots}

    def identifiers(plan, step_idx):
        toolbox = build_storage_configurator(
            [],
            use_indexes=False,
            include_small_model_vanilla=state_includes_small_model_vanilla(
                plan, step_idx
            ),
            include_in_memory=state_includes_in_memory(plan, step_idx),
        ).physical_operators
        return {
            op.get_operation_identifier()
            for op in (
                *toolbox.filter_operators,
                *toolbox.extract_operators,
                *toolbox.join_predicates,
            )
        }

    assert identifiers("kv_operator", KV_OPERATOR_VANILLA_STEP) == identifiers(
        "ablation", ABLATION_VANILLA_STEP
    )
    # And it really is the wider of the two vanilla-only spaces: `gold` is the same state
    # with one LLM operator per modality, which is what reorder_only needs and this does
    # not.
    assert identifiers("gold", 0) < identifiers("kv_operator", KV_OPERATOR_VANILLA_STEP)
    assert state == {slot.key: [] for slot in slots}
