from typing import Type

from reasondb.query_plan.logical_plan import (
    LogicalExtract,
    LogicalFilter,
    LogicalGroupBy,
    LogicalJoin,
    LogicalPlanStep,
    LogicalProject,
    LogicalRename,
    LogicalSorting,
    LogicalTransform,
)


class BaseCapability:
    def __init__(
        self,
        description: str,
        logical_operator: Type[LogicalPlanStep],
    ):
        self._description = description
        self._logical_operator = logical_operator

    def __eq__(self, value: object, /) -> bool:
        return (
            isinstance(value, BaseCapability)
            and value._description == self._description
        )

    def __hash__(self) -> int:
        return hash(self._description)


class Capability:
    def __init__(self, capability: BaseCapability):
        self._input_dtypes = set()
        self._output_dtypes = set()
        self._base_capability = capability

    def add_input_dtypes(self, *dtypes):
        self._input_dtypes.update(dtypes)

    def add_output_dtypes(self, *dtypes):
        self._output_dtypes.update(dtypes)

    def __str__(self):
        if not self._input_dtypes:
            return self._base_capability._description

        input_dtype_str = " or ".join(sorted(x.name for x in self._input_dtypes))
        desc = self._base_capability._description.strip(" .")
        result = f"If the input has data type {input_dtype_str}, then this is possible: {desc}."
        if not self._output_dtypes:
            return result

        output_type_str = " or ".join(sorted(x.name for x in self._output_dtypes))
        output_str = f" The output will have data type {output_type_str}."
        return result + output_str

    def __eq__(self, value: object, /) -> bool:
        return (
            isinstance(value, Capability)
            and self._base_capability == value._base_capability
        )

    def __hash__(self) -> int:
        return hash(self._base_capability)


class Capabilities:
    EXACT_MATCH_JOIN = BaseCapability(
        description="A join of two tables where two rows are join partners if the values of the join keys match exactly.",
        logical_operator=LogicalJoin,
    )
    GENERALIZED_JOIN = BaseCapability(
        description="A join of two tables where whether two rows match is determined by an arbitrary natural languague condition (e.g. join cities that are close to each other, join cities with reports that mention the city, ...).",
        logical_operator=LogicalJoin,
    )

    EXACT_MATCH_FILTER = BaseCapability(
        description="Filter rows where the values of a certain column must match a provided value exactly.",
        logical_operator=LogicalFilter,
    )
    PATTERN_MATCH_FILTER = BaseCapability(
        description="Filter rows where the values of a certain column must matches a pattern.",
        logical_operator=LogicalFilter,
    )
    EXPRESSION_FILTER = BaseCapability(
        description="Filter rows where the values of a certain column must satisfy a mathematical equation.",
        logical_operator=LogicalFilter,
    )
    TEXT_ANALYSIS_FILTER = BaseCapability(
        description="Filter rows by analyzing text and evaluating an arbitrary natural language condition (e.g. select only texts that have positive sentiment, seem to be spam, ...).",
        logical_operator=LogicalFilter,
    )
    IMAGE_ANALYSIS_FILTER = BaseCapability(
        description="Filter rows by analyzing images and evaluating an arbitrary natural language condition (e.g. select only images that depict a dog, are pixel art, ...).",
        logical_operator=LogicalFilter,
    )
    AUDIO_ANALYSIS_FILTER = BaseCapability(
        description="Filter rows by analyzing audio and evaluating an arbitrary natural language condition (e.g. select only audio files that contain music, ...).",
        logical_operator=LogicalFilter,
    )
    PERFECT_FILTER = BaseCapability(
        description="Filters the correct rows based on ground truth information.",
        logical_operator=LogicalFilter,
    )
    DATE_EXTRACT = BaseCapability(
        description="Extract a part of a date value, such as the year, month, or day.",
        logical_operator=LogicalExtract,
    )
    TEXT_EXTRACT = BaseCapability(
        description="Extract arbitrary information from a text, such as what was diagnosed in a medical report.",
        logical_operator=LogicalExtract,
    )
    PERFECT_EXTRACT = BaseCapability(
        description="Extract the correct information based on ground truth information.",
        logical_operator=LogicalExtract,
    )
    IMAGE_EXTRACT = BaseCapability(
        description="Extract information from an image, such as the number of people in a photo.",
        logical_operator=LogicalExtract,
    )
    AUDIO_EXTRACT = BaseCapability(
        description="Extract information from an audio file, such as the spoken words in a recording.",
        logical_operator=LogicalExtract,
    )
    DTYPE_TRANSFORM = BaseCapability(
        description="Convert data from one data type to another, such as converting a string to a number or date",
        logical_operator=LogicalTransform,
    )
    IMAGE_TRANSFORM = BaseCapability(
        description="Arbitrary transformation on images, such as letting the people look older in a photo.",
        logical_operator=LogicalTransform,
    )
    TEXT_TRANSFORM = BaseCapability(
        description="Arbitrary transformation on text, such as translating a text to another language.",
        logical_operator=LogicalTransform,
    )
    # AUDIO_TRANSFORM ?
    TRADITIONAL_PROJECT = BaseCapability(
        description="Project a subset of columns from a table.",
        logical_operator=LogicalProject,
    )
    TRANDITIONAL_RENAME = BaseCapability(
        description="Rename columns in a table.",
        logical_operator=LogicalRename,
    )
    TRADITIONAL_SORT = BaseCapability(
        description="Sort a table by one or more columns.",
        logical_operator=LogicalSorting,
    )
    TRADITIONAL_LIMIT = BaseCapability(
        description="Limit the number of rows in a table.",
        logical_operator=LogicalSorting,
    )

    TRADITIONAL_GROUP_BY = BaseCapability(
        description="Group rows by a set of columns.",
        logical_operator=LogicalGroupBy,
    )

    TRADITIONAL_AGGREGATE = BaseCapability(
        description="Aggregate a set of columns using traditional aggregation functions such as sum, average, min, max, and count.",
        logical_operator=LogicalGroupBy,
    )
