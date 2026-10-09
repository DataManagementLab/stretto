from collections import defaultdict
import math
import pandas as pd
import torch
from torch import Tensor
from typing import TYPE_CHECKING, Dict, Optional, Sequence, Tuple, Set
from reasondb.optimizer.decision import Decision

from torch.nn import Sigmoid, Softmax
from reasondb.database.database import Database
from reasondb.database.indentifier import (
    VirtualTableIdentifier,
)
from reasondb.database.intermediate_state import (
    IntermediateState,
)
from reasondb.optimizer.sampler import ProfilingSampleSpecification
from reasondb.utils.answer_normalization import postprocess_dataframe
from reasondb.query_plan.physical_operator import PhysicalOperator, ProfilingCost
from reasondb.query_plan.tuning_workflow import (
    TuningPipeline,
)
from reasondb.query_plan.unoptimized_physical_plan import (
    UnoptimizedPhysicalPlan,
    UnoptimizedPhysicalPlanStep,
)
from reasondb.reasoning.exceptions import Mistake
from reasondb.reasoning.observation import Observation
from reasondb.utils.logging import FileLogger
from reasondb.utils.timing import measure

if TYPE_CHECKING:
    from reasondb.optimizer.gd_optimizer import DifferentiableConfig


class ProfilingOutput:
    def __init__(self):
        self.merged_output_tuples: Dict[Tuple[int, int], pd.DataFrame] = {}
        self.per_operator_output_tuples: Dict[Tuple[int, int, int], pd.DataFrame] = {}
        self.labels: Dict[Tuple[int, int], Tensor] = {}
        self.profiler_outputs: Dict[Tuple[int, int, int], Tensor] = {}
        self.output_masks: Dict[Tuple[int, int, int], Tensor] = {}
        self.observations: Dict[Tuple[int, int, int], Observation] = {}
        self.per_operator_cost: Dict[Tuple[int, int, int], ProfilingCost] = {}
        self.num_input_tuples: Dict[Tuple[int, int], int] = {}
        #: Tuples each *operator* was actually run on, which differs from
        #: ``num_input_tuples`` when profiling happens over several rounds: an operator
        #: that failed or was pruned contributes cost only for the rounds it ran, so its
        #: per-tuple cost must be computed over those rounds' tuples only.
        self.num_input_tuples_per_operator: Dict[Tuple[int, int, int], int] = {}
        #: ``(cascade_id, level, operator_id)`` of every operator here that supplies
        #: labels rather than being an executable candidate. Their profiling cost is
        #: annotation effort, not query cost -- see ``total_cost``.
        self.label_only_keys: Set[Tuple[int, int, int]] = set()
        self._index_values_to_numeric: Dict[Tuple[int, ...], int] = {}
        self._numeric_indexes: Dict[Tuple[int, int], Tensor] = {}

    @property
    def device(self) -> Optional[torch.device]:
        """Where this output's tensors currently live, or None if it holds none.

        Every tensor here moves together (see :meth:`to`), so the first one answers
        for all of them.
        """
        for store in (
            self.labels,
            self.profiler_outputs,
            self.output_masks,
            self._numeric_indexes,
        ):
            for tensor in store.values():
                return tensor.device
        return None

    def to(self, device: torch.device) -> "ProfilingOutput":
        for key in self.labels:
            self.labels[key] = self.labels[key].to(device)
        for key in self.profiler_outputs:
            self.profiler_outputs[key] = self.profiler_outputs[key].to(device)
        for key in self.output_masks:
            self.output_masks[key] = self.output_masks[key].to(device)
        for key in self._numeric_indexes:
            self._numeric_indexes[key] = self._numeric_indexes[key].to(device)
        return self

    @property
    def total_cost(self) -> ProfilingCost:
        """What profiling this plan cost the *system*.

        Excludes label operators: annotation effort is not query cost, and this value
        is both the reported tuning cost and an input to the "sample more?" decision in
        ``post_optimization_check``. The annotation burden is reported separately as
        :attr:`n_labels_requested`.
        """
        total = ProfilingCost(0.0, 0.0, 0.0)
        for key, cost in self.per_operator_cost.items():
            if key in self.label_only_keys:
                continue
            total += cost
        return total

    @property
    def n_labels_requested(self) -> int:
        """How many tuples a human had to label to tune this plan.

        The annotation burden, measured rather than priced: one label per profiled tuple
        per label-supplying step. Zero unless a label operator was attached.
        """
        return sum(
            self.num_input_tuples.get((cascade_id, level), 0)
            for cascade_id, level, _ in self.label_only_keys
        )

    def get_numeric_index(
        self,
        cascade_id: int,
        level: int,
    ) -> torch.Tensor:
        return self._numeric_indexes[(cascade_id, level)]

    @property
    def total_cost_per_sample(self) -> ProfilingCost:
        total = ProfilingCost(0.0, 0.0, 0.0)
        for key, cost in self.per_operator_cost.items():
            if key in self.label_only_keys:
                # See `total_cost`: annotation effort is not query cost.
                continue
            total += cost / self._operator_input_tuples(key)
        return total

    def _operator_input_tuples(self, key: Tuple[int, int, int]) -> int:
        """Tuples to divide an operator's accumulated cost by.

        The per-operator count when there is one, the level-wide count otherwise (in a
        single-round run the two are equal by construction).
        """
        cascade_id, level, _ = key
        return self.num_input_tuples_per_operator.get(
            key, self.num_input_tuples[(cascade_id, level)]
        )

    def per_operator_and_sample_cost(
        self, cascade_id: int, level: int, operator_id: int
    ) -> ProfilingCost:
        if (cascade_id, level, operator_id) in self.label_only_keys:
            # This feeds the differentiable cost model's `cost_vector`, whose sum is the
            # denominator every job's cost is scaled by. A large entry there makes every
            # plan's scaled cost ~0 and the optimizer cost-blind -- and the operator is
            # never executed, so it costs the plan nothing.
            return ProfilingCost(0.0, 0.0, 0.0)
        key = (cascade_id, level, operator_id)
        cost = self.per_operator_cost.get(key, ProfilingCost(0.0, 0.0, 0.0))
        return cost / self._operator_input_tuples(key)

    def add(self, cascade_output: "ProfileLevelOutput"):
        cascade_id = cascade_output.cascade_id
        level = cascade_output.level
        self.num_input_tuples[(cascade_id, level)] = cascade_output.num_input_tuples

        self.merged_output_tuples[(cascade_id, level)] = (
            cascade_output.merged_output_tuples
        )
        for operator_id, mask in cascade_output.masks.items():
            self.output_masks[(cascade_id, level, operator_id)] = mask
            self.profiler_outputs[(cascade_id, level, operator_id)] = (
                cascade_output.profiler_outputs[operator_id]
            )
            self.per_operator_output_tuples[(cascade_id, level, operator_id)] = (
                cascade_output.per_operatator_output_tuples[operator_id]
            )

        self.labels[(cascade_id, level)] = cascade_output.labels
        self.label_only_keys.update(
            (cascade_id, level, operator_id)
            for operator_id in cascade_output.label_only_operator_ids
        )
        for operator_id, observation in cascade_output.observations.items():
            self.observations[(cascade_id, level, operator_id)] = observation
            self.per_operator_cost[(cascade_id, level, operator_id)] = (
                cascade_output.per_operator_costs[operator_id]
            )
            self.num_input_tuples_per_operator[(cascade_id, level, operator_id)] = (
                cascade_output.num_input_tuples
            )

    def prepend(self, other: Optional["ProfilingOutput"], logger: FileLogger):
        if other is None:
            self.compute_numeric_indexes()
            return
        # `other` is the previous sampling round's output, which the optimizer moved
        # onto its own device; this round's has just been profiled and is on the CPU.
        # Every merge below is a `torch.cat` of one against the other, so align them
        # first -- onto ours, since what follows is CPU-side bookkeeping (pandas
        # indices, `.numpy()` in the baselines).
        if self.device is not None and other.device != self.device:
            other.to(self.device)
        # Union rather than assume they match: an operator that failed to profile in one
        # round is absent from that round's output.
        self.label_only_keys |= other.label_only_keys
        all_merged_sorted_indices = {}
        for cascade_id, level in other.merged_output_tuples.keys():
            self.merged_output_tuples[(cascade_id, level)] = pd.concat(
                [
                    other.merged_output_tuples[(cascade_id, level)],
                    self.merged_output_tuples[(cascade_id, level)],
                ],
                ignore_index=False,
                axis=0,
            )
            self.labels[(cascade_id, level)] = torch.cat(
                [
                    other.labels[(cascade_id, level)],
                    self.labels[(cascade_id, level)],
                ],
                dim=0,
            )
            # Warning if only few positives in labels
            num_positives = self.labels[(cascade_id, level)].sum()
            num_total = len(self.labels[(cascade_id, level)])
            logger.info(
                __name__,
                f"Cascade {cascade_id}, level {level} has {num_positives} positive labels of {num_total} items.",
            )

            assert len(self.merged_output_tuples[(cascade_id, level)]) == len(
                self.labels[(cascade_id, level)]
            )
            # sort both after output_tuples index
            merged_sorted_indices = self.merged_output_tuples[
                (cascade_id, level)
            ].index.argsort()
            self.merged_output_tuples[(cascade_id, level)] = self.merged_output_tuples[
                (cascade_id, level)
            ].iloc[merged_sorted_indices]
            self.labels[(cascade_id, level)] = self.labels[(cascade_id, level)][
                merged_sorted_indices
            ]
            all_merged_sorted_indices[(cascade_id, level)] = merged_sorted_indices
            self.num_input_tuples[(cascade_id, level)] = (
                other.num_input_tuples[(cascade_id, level)]
                + self.num_input_tuples[(cascade_id, level)]
            )

        for cascade_id, level, operator_id in list(other.profiler_outputs.keys()):
            if (cascade_id, level, operator_id) not in self.profiler_outputs:
                # This round did not profile that operator: it was pruned (no feasible
                # restart picked it) or it failed. Drop the previous round's half rather
                # than carrying it: a half-accumulated operator has rows for some of the
                # sample and not the rest.
                #
                # Everything downstream treats a missing operator as one that failed to
                # profile (`get_transition_probabilities` skips it,
                # `simulate_pipeline_pass` calls `add_failed_operator`, and
                # `get_tuned_pipeline_from_config` will not plan it). Pruning is thus a
                # commitment: the operator does not come back.
                self._forget_operator((cascade_id, level, operator_id))
                continue
            # merge first
            self.profiler_outputs[(cascade_id, level, operator_id)] = torch.cat(
                [
                    other.profiler_outputs[(cascade_id, level, operator_id)],
                    self.profiler_outputs[(cascade_id, level, operator_id)],
                ],
                dim=0,
            )
            self.output_masks[(cascade_id, level, operator_id)] = torch.cat(
                [
                    other.output_masks[(cascade_id, level, operator_id)],
                    self.output_masks[(cascade_id, level, operator_id)],
                ],
                dim=0,
            )
            self.observations[(cascade_id, level, operator_id)] = other.observations[
                (cascade_id, level, operator_id)
            ]
            self.per_operator_output_tuples[(cascade_id, level, operator_id)] = (
                pd.concat(
                    [
                        other.per_operator_output_tuples[
                            (cascade_id, level, operator_id)
                        ],
                        self.per_operator_output_tuples[
                            (cascade_id, level, operator_id)
                        ],
                    ],
                    ignore_index=False,
                    axis=0,
                )
            )

            # now sort all
            merged_sorted_indices = all_merged_sorted_indices[(cascade_id, level)]
            per_operator_sorted_indices = self.per_operator_output_tuples[
                (cascade_id, level, operator_id)
            ].index.argsort()
            assert len(merged_sorted_indices) == len(
                self.output_masks[(cascade_id, level, operator_id)]
            )
            assert len(per_operator_sorted_indices) == len(
                self.profiler_outputs[(cascade_id, level, operator_id)]
            )
            self.profiler_outputs[(cascade_id, level, operator_id)] = (  # not padded
                self.profiler_outputs[(cascade_id, level, operator_id)][
                    per_operator_sorted_indices
                ]
            )
            self.per_operator_output_tuples[(cascade_id, level, operator_id)] = (
                self.per_operator_output_tuples[
                    (cascade_id, level, operator_id)
                ].iloc[per_operator_sorted_indices]
            )  # not padded
            self.output_masks[(cascade_id, level, operator_id)] = self.output_masks[
                (cascade_id, level, operator_id)
            ][merged_sorted_indices]  # is padded

            # finally add costs
            self.per_operator_cost[(cascade_id, level, operator_id)] = (
                other.per_operator_cost[(cascade_id, level, operator_id)]
                + self.per_operator_cost[(cascade_id, level, operator_id)]
            )
            # ...and the tuples that cost was incurred over, which is what makes the
            # sum divisible back into a per-tuple price. Summed in lockstep with the
            # cost, so the two can never describe different sets of rounds.
            self.num_input_tuples_per_operator[(cascade_id, level, operator_id)] = other.num_input_tuples_per_operator.get(
                (cascade_id, level, operator_id), 0
            ) + self.num_input_tuples_per_operator.get(
                (cascade_id, level, operator_id), 0
            )

        self.compute_numeric_indexes()

    def _forget_operator(self, key: Tuple[int, int, int]) -> None:
        """Remove every trace of one operator from this accumulated output."""
        self.profiler_outputs.pop(key, None)
        self.output_masks.pop(key, None)
        self.per_operator_output_tuples.pop(key, None)
        self.observations.pop(key, None)
        self.per_operator_cost.pop(key, None)
        self.num_input_tuples_per_operator.pop(key, None)
        self.label_only_keys.discard(key)

    def compute_numeric_indexes(self):
        self._index_values_to_numeric = {}
        current_index = 0
        for output_df in self.merged_output_tuples.values():
            for index_tuple in output_df.index:
                if index_tuple not in self._index_values_to_numeric:
                    self._index_values_to_numeric[index_tuple] = current_index
                    current_index += 1
        for cascade_id, level in self.merged_output_tuples.keys():
            numeric_indexes = []
            index_tuples = self.merged_output_tuples[(cascade_id, level)].index
            numeric_indexes = index_tuples.map(
                lambda idx: self._index_values_to_numeric[idx]
            )
            # On this output's own device, not the default one: these index tensors
            # feed `scatter_reduce` against the optimizer's device tensors in
            # `compute_merged_scores`, and all tensors here move together.
            self._numeric_indexes[(cascade_id, level)] = torch.tensor(
                numeric_indexes, dtype=torch.long, device=self.device or "cpu"
            )

    def get(self, cascade_id: int, level: int, operator_id: int):
        return self.profiler_outputs[(cascade_id, level, operator_id)]

    def get_labels(self, cascade_id: int, level: int):
        return self.labels[(cascade_id, level)]

    def get_output_tuples(self, cascade_id: int, level: int):
        return self.merged_output_tuples[(cascade_id, level)]

    def get_output_mask(self, cascade_id: int, level: int, operator_id: int):
        return self.output_masks[(cascade_id, level, operator_id)]

    def get_observation(self, cascade_id: int, level: int, operator_id: int):
        return self.observations[(cascade_id, level, operator_id)]


class ProfileLevelOutput:
    def __init__(
        self,
        cascade_id: int,
        level: int,
        labels: Tensor,
        output_tuples: Dict[int, pd.DataFrame],
        profiler_outputs: Dict[int, Tensor],
        observations: Dict[int, Observation],
        per_operator_costs: Dict[int, ProfilingCost],
        num_input_tuples: int,
        label_only_operator_ids: Optional[Set[int]] = None,
    ):
        self.cascade_id = cascade_id
        self.level = level
        self.labels = labels
        self._output_tuples = output_tuples
        self._masks = {}
        self._merged_output = None
        self.profiler_outputs = profiler_outputs
        self.observations = observations
        self.per_operator_costs = per_operator_costs
        self.num_input_tuples = num_input_tuples
        #: Operators here that supply labels rather than being executable candidates.
        #: Their profiling cost is annotation effort, not query cost, and their extracted
        #: values are the reference every other candidate is compared against.
        self.label_only_operator_ids = set(label_only_operator_ids or ())

    @property
    def merged_output_tuples(self) -> pd.DataFrame:
        assert self._merged_output is not None, "Call consolidate() first."
        return self._merged_output

    @property
    def per_operatator_output_tuples(self) -> Dict[int, pd.DataFrame]:
        return self._output_tuples

    @property
    def masks(self) -> Dict[int, Tensor]:
        return self._masks

    def consoldidate(self):
        for ot in self._output_tuples.values():
            assert (ot.index == ot.sort_index().index).all()

        self._output_tuples = {
            op_id: postprocess_dataframe(df)
            for op_id, df in self._output_tuples.items()
        }

        concatenated = pd.concat(self._output_tuples)
        index_names = list(concatenated.index.names)
        if None in index_names:
            concatenated = concatenated.reset_index([None], drop=True)
            index_names = [n for n in index_names if n is not None]
        concatenated.reset_index(inplace=True)
        concatenated.fillna(0.0, inplace=True)
        no_duplicates = concatenated.drop_duplicates().copy()
        no_duplicates.set_index(index_names, inplace=True)
        no_duplicates.sort_index(inplace=True)
        self._merged_output = no_duplicates

        check_df = no_duplicates.reset_index()
        for op_id, v in self._output_tuples.items():
            v.fillna(0.0, inplace=True)
            v_set = set(tuple(t) for t in v.reset_index().values)
            mask = check_df.apply(lambda row: tuple(row) in v_set, axis=1)
            assert len(v_set) == mask.sum()
            self._masks[op_id] = torch.tensor(mask.values, dtype=torch.bool)

        padded_labels = torch.zeros(len(no_duplicates), dtype=torch.bool)
        label_op_id = max(self._output_tuples.keys())
        label_mask = self._masks[label_op_id]
        padded_labels[label_mask] = self.labels
        self.labels = padded_labels


class Profiler:
    def __init__(self, database: "Database"):
        self.database = database

    async def profile(
        self,
        pipeline: TuningPipeline,
        intermediate_state: IntermediateState,
        previous_observations: Optional[Dict[Tuple[int, int, int], Observation]],
        sample: "ProfilingSampleSpecification",
        logger: FileLogger,
        operator_filter: Optional[Dict[Tuple[int, int], Set[int]]] = None,
    ) -> ProfilingOutput:
        with measure("profiling"):
            input_data = await self.get_inputs_for_profiling(
                pipeline=pipeline,
                intermediate_state=intermediate_state,
                sample=sample,
                logger=logger,
            )
            profiling_output = ProfilingOutput()
            for cascade_id, cascade in enumerate(pipeline.steps_in_parallel):
                await self.profile_cascade(
                    cascade_id=cascade_id,
                    cascade=cascade,
                    input_data=input_data,
                    intermediate_state=intermediate_state,
                    sample=sample,
                    profiling_output=profiling_output,
                    previous_observations=previous_observations,
                    logger=logger,
                    operator_filter=operator_filter,
                )
            return profiling_output

    async def profile_cascade(
        self,
        cascade_id: int,
        cascade: Sequence[UnoptimizedPhysicalPlanStep],
        input_data: dict,
        intermediate_state: IntermediateState,
        sample: "ProfilingSampleSpecification",
        profiling_output: ProfilingOutput,
        previous_observations: Optional[Dict[Tuple[int, int, int], Observation]],
        logger: FileLogger,
        operator_filter: Optional[Dict[Tuple[int, int], Set[int]]] = None,
    ):
        profile_level_outputs = {}
        profiling_pipeline = UnoptimizedPhysicalPlan()
        for level, step in enumerate(cascade):
            level_input_data = input_data
            if level > 0:
                level_input_data = self.get_input_for_next_level(
                    inputs=step.inputs,
                    input_data=input_data,
                    profile_level_outputs=profile_level_outputs,
                    logger=logger,
                )
            allowed_operator_ids = (
                None
                if operator_filter is None
                else operator_filter.get((cascade_id, level))
            )
            profile_level_output = await self.profile_level(
                input_data=level_input_data,
                cascade_id=cascade_id,
                cascade=cascade,
                intermediate_state=intermediate_state,
                sample=sample,
                previous_observations=previous_observations,
                level=level,
                logger=logger,
                allowed_operator_ids=allowed_operator_ids,
            )
            profile_level_outputs[step.output] = profile_level_output

            # The label source's observation advances the profiling state to the next
            # cascade level. That stays right when the source is a human: an
            # observation describes the step's *schema* -- which hidden column holds the
            # verdict, which column an extract adds -- not its verdicts, and every
            # candidate for a step produces the same one.
            gold_operator_id = len(step.operators) - 1
            intermediate_state = profiling_pipeline.append(
                step=step,
                observation=profile_level_output.observations[gold_operator_id],
                database=intermediate_state,
            )
            profiling_output.add(profile_level_output)

    def get_input_for_next_level(
        self,
        inputs: Sequence[VirtualTableIdentifier],
        input_data: Dict,
        profile_level_outputs: Dict[VirtualTableIdentifier, ProfileLevelOutput],
        logger: FileLogger,
    ) -> Dict[VirtualTableIdentifier, pd.DataFrame]:
        result = dict(input_data)
        for identifier in inputs:
            if identifier in profile_level_outputs:
                data = profile_level_outputs[identifier].merged_output_tuples
                result[identifier] = data
        return result

    async def profile_level(
        self,
        input_data: dict,
        cascade_id: int,
        cascade: Sequence[UnoptimizedPhysicalPlanStep],
        intermediate_state: IntermediateState,
        sample: "ProfilingSampleSpecification",
        previous_observations: Optional[Dict[Tuple[int, int, int], Observation]],
        level: int,
        logger: FileLogger,
        allowed_operator_ids: Optional[Set[int]] = None,
    ) -> ProfileLevelOutput:
        step = cascade[level]
        keep_labels = None
        all_profiler_outputs = {}
        all_output_tuples = {}
        all_observations = {}
        per_operator_costs = defaultdict(lambda: ProfilingCost(0.0, 0.0, 0.0))
        data_sample = [input_data[tbl] for tbl in step.inputs]
        sample_size = math.prod([len(d) for d in data_sample])
        dataset_size = int(sample_size / sample.sample_fraction)

        allowed_operator_id_set = (
            None if allowed_operator_ids is None else set(allowed_operator_ids)
        )
        for operator_id, operator in enumerate(step.operators):
            llm_parameters = step.llm_configurations[operator.get_llm_parameters().name]
            # The last candidate is where this step's labels come from -- either the
            # highest-quality model, or a human label source when one is attached. Either
            # way it must be profiled, because every other candidate is scored against
            # it. Whether it may also be *executed* is a separate question, answered by
            # `is_label_only` in the optimizer.
            is_gold = operator_id == len(step.operators) - 1

            if (
                allowed_operator_id_set is not None
                and operator_id not in allowed_operator_id_set
                and not is_gold  # gold operator is always needed to derive labels
            ):
                continue

            if previous_observations is not None:
                try:
                    observation = previous_observations[cascade_id, level, operator_id]
                except KeyError:
                    logger.warning(
                        __name__,
                        f"Previous profiling output missing observation for operator {operator} at cascade {cascade_id}, level {level}.",
                    )
                    assert (
                        not is_gold
                    ), "Gold operator missing in previous profiling output."
                    continue

            else:
                try:
                    observation = await operator.get_observation(
                        database_state=intermediate_state,
                        inputs=step.inputs,
                        output=step.output,
                        output_columns=step.get_output_columns(),
                        llm_parameters=llm_parameters,
                        data_sample=data_sample,
                        logical_plan_step=step.logical_plan_step,
                        logger=logger,
                    )
                except Mistake as m:
                    logger.warning(
                        __name__,
                        f"Operator {operator} failed to get observation. Skipping. Error: {m}",
                    )
                    if not is_gold:
                        continue
                    logger.warning(
                        __name__,
                        f"Gold Operator {operator} failed to get observation. Skipping. Error: {m}",
                    )
                    observation = None

            try:
                if observation is None:
                    raise Mistake("No observation available.")
                (
                    output_tuples,
                    profile_output,
                    profile_cost,
                ) = await operator.profile(
                    inputs=step.inputs,
                    database_state=intermediate_state,
                    observation=observation,
                    llm_parameters=llm_parameters,
                    sample=sample,
                    data_sample=data_sample,
                    logger=logger,
                )
                assert len(output_tuples) == profile_output.shape[0]
            except Mistake as m:
                logger.warning(
                    __name__,
                    f"Operator {operator} failed to profile. Skipping. Error: {m}",
                )
                if not is_gold:
                    continue
                # `fallback_gold_profile` fabricates an all-KEEP matrix, i.e. it declares
                # every tuple a positive label. That is a last resort for a model that
                # failed, but never acceptable for a human label source (which raises
                # `MissingLabelsError`, not `Mistake`, so this is not expected to trigger).
                assert not getattr(operator, "is_label_only", False), (
                    f"Label operator {operator} failed to profile; refusing to fabricate "
                    f"all-KEEP labels for it. Original error: {m}"
                )
                output_tuples, profile_output, profile_cost = (
                    self.fallback_gold_profile(
                        operator=operator,
                        input_data=data_sample,
                    )
                )

            argsorted = output_tuples.index.argsort()
            output_tuples = output_tuples.iloc[argsorted]
            profile_output = profile_output[torch.from_numpy(argsorted)]
            duplicated = output_tuples.assign(  # need to also take index into account when deciding duplicates
                **{
                    name: output_tuples.index.get_level_values(i)
                    for i, name in enumerate(output_tuples.index.names)
                }
            ).duplicated()
            output_tuples = output_tuples[~duplicated]
            profile_output = profile_output[torch.from_numpy(~duplicated.values)]

            all_profiler_outputs[operator_id] = profile_output

            if is_gold:
                assert keep_labels is None, "Multiple gold operators found."
                keep_labels = self.get_labels(
                    operator=operator,
                    profile_output=profile_output,
                )

            all_output_tuples[operator_id] = output_tuples
            all_observations[operator_id] = observation

            scaled_profile_cost = operator.scale_cost(
                cost=profile_cost, sample_size=sample_size, dataset_size=dataset_size
            )
            if scaled_profile_cost != profile_cost:
                logger.info(
                    __name__,
                    f"Scaled cost from {profile_cost} to {scaled_profile_cost}",
                )
            per_operator_costs[operator_id] = scaled_profile_cost
            per_sample_cost = scaled_profile_cost / len(data_sample[0])

            logger.info(
                __name__,
                f"Using per sample cost of {per_sample_cost} for operator {operator.get_operation_identifier()}",
            )
        assert keep_labels is not None, "No gold operator found."

        result = ProfileLevelOutput(
            cascade_id=cascade_id,
            level=level,
            labels=keep_labels,
            output_tuples=all_output_tuples,
            profiler_outputs=all_profiler_outputs,
            observations=all_observations,
            per_operator_costs=per_operator_costs,
            num_input_tuples=len(data_sample[0]),
            label_only_operator_ids={
                op_id
                for op_id in all_output_tuples
                if getattr(step.operators[op_id], "is_label_only", False)
            },
        )
        result.consoldidate()
        return result

    def fallback_gold_profile(
        self, operator: PhysicalOperator, input_data: Sequence[pd.DataFrame]
    ) -> Tuple[pd.DataFrame, Tensor, ProfilingCost]:
        in_data = operator.potentially_cartesion_product_input_data(input_data)
        m = torch.ones(len(in_data), 1, 3)
        m[:, 0, 0] = 1000  # Keep
        m[:, 0, 1] = -1000  # Discard
        m[:, 0, 2] = -1000  # Unsure
        return in_data, m, ProfilingCost(0.0, 0.0, 0.0)

    def get_transition_probabilities(
        self,
        step: UnoptimizedPhysicalPlanStep,
        cascade_id: int,
        level: int,
        profiling_output: ProfilingOutput,
        pick_temperature: float,
        params_temperature: float,
        config: "DifferentiableConfig",
    ) -> Dict[int, Tensor]:
        softmax = Softmax(dim=2)
        sigmoid = Sigmoid()
        all_probabilities = {}
        for operator_id, operator in enumerate(step.operators):
            parameters = config.get_parameters(
                cascade_id=cascade_id,
                level=level,
                physical_operator_id=operator_id,
            )
            try:
                profile_output = profiling_output.get(cascade_id, level, operator_id)
            except KeyError:
                continue
            decisions: Tensor = operator.profile_get_decision_matrix(
                parameters=parameters,
                profile_output=profile_output,
            )  # Shape: (num_jobs, num_tuples, 3)
            soft_decisions = softmax(decisions / params_temperature)
            mask = profiling_output.get_output_mask(
                cascade_id=cascade_id, level=level, operator_id=operator_id
            )
            padded_soft_decisions = torch.zeros(
                mask.shape[0], soft_decisions.shape[1], 3, device=soft_decisions.device
            )
            padded_soft_decisions[:, :, Decision.DISCARD] = 1.0
            padded_soft_decisions[mask, :, :] = soft_decisions

            operator_pick_score = config.get_operator_pick_score(
                cascade_id=cascade_id,
                level=level,
                physical_operator_id=operator_id,
            )  # Shape: (num_jobs, )
            soft_operator_pick = sigmoid(
                operator_pick_score / pick_temperature
            ).reshape(1, -1)
            keep_prob = padded_soft_decisions[:, :, Decision.KEEP] * soft_operator_pick
            discard_prob = (
                padded_soft_decisions[:, :, Decision.DISCARD] * soft_operator_pick
            )
            unsure_prob = 1 - (keep_prob + discard_prob)
            combined_probabilities = torch.stack(
                [keep_prob, discard_prob, unsure_prob], dim=2
            )
            all_probabilities[operator_id] = combined_probabilities
            assert not torch.isnan(combined_probabilities).any()
        return all_probabilities

    def get_labels(
        self,
        operator: "PhysicalOperator",
        profile_output: Tensor,
    ) -> Tensor:
        def get_parameters(x: str) -> Tensor:
            return torch.tensor([operator.get_default_tuning_parameters()[x]])

        decisions = operator.profile_get_decision_matrix(
            parameters=get_parameters,
            profile_output=profile_output,
        )
        hard_decisions = torch.argmax(decisions, dim=2).reshape(
            -1
        )  # Shape: len(step.inputs) x (num_tuples, )
        keep_labels = hard_decisions == Decision.KEEP
        return keep_labels

    async def get_inputs_for_profiling(
        self,
        pipeline: TuningPipeline,
        intermediate_state: IntermediateState,
        sample: "ProfilingSampleSpecification",
        logger: FileLogger,
    ) -> Dict[VirtualTableIdentifier, pd.DataFrame]:
        input_data = {}
        for cascade in pipeline.steps_in_parallel:
            step = cascade[0]
            missing_data_table = sorted(set(step.inputs) - set(input_data.keys()))
            if missing_data_table:
                input_sample = await step.get_input_sample(
                    database_state=intermediate_state,
                    sample=sample,
                    logger=logger,
                    override_inputs=missing_data_table,
                    for_prompt=False,
                )
                for tbl, data in zip(missing_data_table, input_sample):
                    data.sort_index(inplace=True)
                    input_data[tbl] = data
        return input_data
