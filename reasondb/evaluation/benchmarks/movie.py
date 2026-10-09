import os
import re
import logging
from pathlib import Path
from typing import Dict, Literal, Sequence, Union
from reasondb.database.database import ExperimentalDatabase
from reasondb.database.indentifier import (
    InPlaceColumn,
    VirtualTableIdentifier,
)
from reasondb.evaluation.benchmark import Benchmark, RandomBenchmark
from reasondb.query_plan.logical_plan import (
    LogicalFilter,
    LogicalJoin,
    LogicalPlan,
    LogicalExtract,
    LogicalRename,
    LogicalLimit,
)
from reasondb.query_plan.query import (
    OperatorOption,
    OperatorPlaceholder,
    Queries,
    Query,
    QueryShape,
    RandomOrder,
)

logger = logging.getLogger(__name__)


MOVIE_QUERIES = Queries(
    Query(
        "Which pair of reviews discuss the same movie",
        _ground_truth_logical_plan=LogicalPlan(
            [
                LogicalRename(
                    explanation="First, we rename the reviews table to prepare for the self-join.",
                    inputs=[VirtualTableIdentifier("reviews")],
                    output=VirtualTableIdentifier("reviews_other"),
                    expression="Rename {reviews.reviewtext} to [reviewtext_other]",
                ),
                LogicalFilter(
                    explanation="First, we need to filter the reviews that are clearly positive.",
                    inputs=[VirtualTableIdentifier("reviews")],
                    output=VirtualTableIdentifier("positive_reviews"),
                    expression="{reviews.reviewtext} is clearly positive",
                ),
                LogicalJoin(
                    explanation="We need to match the reviews that discuss the same movie.",
                    inputs=[
                        VirtualTableIdentifier("positive_reviews"),
                        VirtualTableIdentifier("reviews_other"),
                    ],
                    output=VirtualTableIdentifier("output"),
                    expression="{positive_reviews.reviewtext} and {reviews_other.reviewtext_other} discuss the same movie",
                ),
            ]
        ),
    ),
)


class Movie(Benchmark):
    @classmethod
    def name(cls) -> str:
        return "movie"

    @property
    def has_ground_truth(self) -> bool:
        return False

    @staticmethod
    def get_csv():
        orig_csv = Path(__file__).parent / "files" / "reviews_100.csv"
        return orig_csv

    @staticmethod
    def get_queries() -> Queries:
        return MOVIE_QUERIES

    @staticmethod
    def urls():
        return {}

    @classmethod
    def download(cls, split: Literal["train", "dev", "test"]) -> Benchmark:
        assert split == "dev"
        os.makedirs(cls.dir(), exist_ok=True)
        return cls.load_from_disk(split)

    @classmethod
    def load_from_disk(cls, split: Literal["train", "dev", "test"]) -> "Benchmark":
        # Everything here goes through `cls`, so a subclass overriding `name()`,
        # `get_csv()` and `get_queries()` (see benchmarks/curated.py) inherits this table
        # wiring and gets its own cache dir and DuckDB database name.
        assert split == "dev"
        os.makedirs(cls.dir(), exist_ok=True)
        benchmark = cls(
            split,
            ExperimentalDatabase.load_from_files(
                db_name=cls.name(),
                split=split,
                table_names=["reviews"],
                paths=[cls.get_csv()],
                text_columns=[InPlaceColumn("reviews.reviewtext")],
            ),
            cls.get_queries(),
        )
        return benchmark


class MovieRandom(RandomBenchmark):
    @classmethod
    def name(cls) -> str:
        return "movie_random"

    @property
    def has_ground_truth(self) -> bool:
        return False

    @staticmethod
    def get_csv():
        orig_csv = Path(__file__).parent / "files" / "reviews_1000.csv"
        return orig_csv

    @staticmethod
    def urls():
        return {}

    @staticmethod
    def download(split: Literal["train", "dev", "test"]) -> Benchmark:
        assert split == "dev"
        os.makedirs(MovieRandom.dir(), exist_ok=True)
        return MovieRandom.load_from_disk(split)

    @classmethod
    def _load_database(cls, split: Literal["train", "dev", "test"]):
        assert split == "dev"
        os.makedirs(cls.dir(), exist_ok=True)
        return ExperimentalDatabase.load_from_files(
            db_name=cls.name(),
            split=split,
            table_names=["reviews"],
            paths=[cls.get_csv()],
            text_columns=[
                InPlaceColumn("reviews.reviewtext"),
            ],
        )

    @classmethod
    def _get_query_shapes(cls) -> Sequence[QueryShape]:
        return MOVIE_QUERY_SHAPES

    @classmethod
    def _get_operator_options(cls) -> Sequence[OperatorOption]:
        return MOVIE_OPERATOR_OPTIONS

    @classmethod
    def _single_filter_shape(cls) -> Union[QueryShape, Dict[str, QueryShape]]:
        return SINGLE_FILTER_SHAPE

    @classmethod
    def get_join_queries(cls) -> "Queries":
        return MOVIE_JOIN_QUERIES


class MovieRandomHuge(RandomBenchmark):
    @classmethod
    def name(cls) -> str:
        return "movie_random_huge"

    @property
    def has_ground_truth(self) -> bool:
        return False

    @staticmethod
    def get_csv():
        orig_csv = Path(__file__).parent / "files" / "reviews_10000.csv"
        return orig_csv

    @staticmethod
    def urls():
        return {}

    @staticmethod
    def download(split: Literal["train", "dev", "test"]) -> Benchmark:
        assert split == "dev"
        os.makedirs(MovieRandomHuge.dir(), exist_ok=True)
        return MovieRandomHuge.load_from_disk(split)

    @classmethod
    def _load_database(cls, split: Literal["train", "dev", "test"]):
        assert split == "dev"
        os.makedirs(cls.dir(), exist_ok=True)
        return ExperimentalDatabase.load_from_files(
            db_name=cls.name(),
            split=split,
            table_names=["reviews"],
            paths=[cls.get_csv()],
            text_columns=[
                InPlaceColumn("reviews.reviewtext"),
            ],
        )

    @classmethod
    def _get_query_shapes(cls) -> Sequence[QueryShape]:
        return MOVIE_QUERY_SHAPES_HUGE

    @classmethod
    def _get_operator_options(cls) -> Sequence[OperatorOption]:
        return MOVIE_OPERATOR_OPTIONS

    @classmethod
    def _single_filter_shape(cls) -> Union[QueryShape, Dict[str, QueryShape]]:
        return SINGLE_FILTER_SHAPE

    @classmethod
    def get_join_queries(cls) -> "Queries":
        return MOVIE_JOIN_QUERIES


SINGLE_FILTER_SHAPE = QueryShape(
    OperatorPlaceholder(
        LogicalFilter,
        inputs=[VirtualTableIdentifier("reviews")],
        output=VirtualTableIdentifier("output"),
    ),
)
MOVIE_OPERATOR_OPTIONS = [
    # joins on reviewtext (self-join via renamed table)
    OperatorOption(
        LogicalJoin,
        "{:0:.reviewtext} and {:1:.reviewtext_other} discuss movies of the same genre, "
        "where the genre is exactly one of action, comedy, drama, horror, romance, "
        "science fiction, thriller, documentary, animation or other",
    ),
    OperatorOption(
        LogicalJoin,
        "{:0:.reviewtext} and {:1:.reviewtext_other} have contradicting opinions, one clearly positive and the other clearly negative",
    ),
    OperatorOption(
        LogicalJoin,
        "{:0:.reviewtext} and {:1:.reviewtext_other} both praise the direction or the director's work",
    ),
    OperatorOption(
        LogicalJoin,
        "{:0:.reviewtext} and {:1:.reviewtext_other} both criticize the performance of the lead actor",
    ),
    OperatorOption(
        LogicalJoin,
        "{:0:.reviewtext} and {:1:.reviewtext_other} both express a positive sentiment",
    ),
    OperatorOption(
        LogicalJoin,
        "{:0:.reviewtext} and {:1:.reviewtext_other} both express a negative sentiment",
    ),
    OperatorOption(
        LogicalJoin,
        "{:0:.reviewtext} and {:1:.reviewtext_other} both have mixed feelings about the movie (both positive and negative)",
    ),
    OperatorOption(
        LogicalJoin,
        "{:0:.reviewtext} and {:1:.reviewtext_other} both contain specific plot spoilers",
    ),
    OperatorOption(
        LogicalJoin,
        "{:0:.reviewtext} and {:1:.reviewtext_other} both mention and praise the film's soundtrack or musical score",
    ),
    OperatorOption(
        LogicalJoin,
        "{:0:.reviewtext} and {:1:.reviewtext_other} both specifically praise the movie's ending or plot twist",
    ),
    ## filters on reviewtext
    OperatorOption(
        LogicalFilter,
        "{:0:.reviewtext} is clearly positive",
    ),
    OperatorOption(
        LogicalFilter,
        "{:0:.reviewtext} is clearly negative",
    ),
    OperatorOption(
        LogicalFilter,
        "{:0:.reviewtext} mentions excellent acting",
    ),
    OperatorOption(
        LogicalFilter,
        "{:0:.reviewtext} mentions poor plot",
    ),
    OperatorOption(
        LogicalFilter,
        "{:0:.reviewtext} mentions great cinematography",
    ),
    OperatorOption(
        LogicalFilter,
        "{:0:.reviewtext} mentions terrible dialogue",
    ),
    OperatorOption(
        LogicalFilter,
        "{:0:.reviewtext} is a rave review",
    ),
    OperatorOption(
        LogicalFilter,
        "{:0:.reviewtext} is a scathing review",
    ),
    OperatorOption(
        LogicalFilter,
        "{:0:.reviewtext} praises the soundtrack",
    ),
    OperatorOption(
        LogicalFilter,
        "{:0:.reviewtext} criticizes the special effects",
    ),
    # extract from reviewtext
    OperatorOption(
        LogicalExtract,
        "Extract the [sentiment] of {:0:.reviewtext}? (positive or negative)",
    ),
    OperatorOption(
        LogicalExtract,
        "Extract the movie [title] mentioned in {:0:.reviewtext}",
    ),
    OperatorOption(
        LogicalExtract,
        "Extract one [actor] that is praised particularly in {:0:.reviewtext} (or 'none' if no actor is praised)",
    ),
    OperatorOption(
        LogicalExtract,
        "Extract one aspect of the movie that is [criticized] in {:0:.reviewtext} (choose from plot, acting, cinematography, soundtrack, special effects, dialogue, or 'none' if no aspect is criticized)",
    ),
    OperatorOption(
        LogicalExtract,
        "Extract whether the reviewer [would_recommend] the movie based on {:0:.reviewtext} (yes/no)",
    ),
    OperatorOption(
        LogicalExtract,
        "Extract the main [emotion] expressed by the reviewer in {:0:.reviewtext} (e.g., joy, disappointment, anger, excitement, confusion)",
    ),
    OperatorOption(
        LogicalExtract,
        "Extract the [director]'s name mentioned in {:0:.reviewtext} (or 'none' if not mentioned)",
    ),
    OperatorOption(
        LogicalExtract,
        "Extract whether the review in {:0:.reviewtext} contains [spoilers] (yes/no)",
    ),
    OperatorOption(
        LogicalExtract,
        "Extract whether the reviewer [compares] the movie to another film in {:0:.reviewtext} (yes/no)",
    ),
    OperatorOption(
        LogicalExtract,
        "Extract the [target_audience] implied or stated in {:0:.reviewtext} (e.g., families, children, adults, fans of action, 'none')",
    ),
]


def _make_join_query(expression: str) -> Query:
    expression = re.sub(
        r"{\:0:\.([a-z_][a-z0-9_]*)}",
        r"reviews.\1",
        expression,
    )
    expression = re.sub(
        r"{\:1:\.([a-z_][a-z0-9_]*)}",
        r"reviews_other.\1",
        expression,
    )
    return Query(
        expression,
        _ground_truth_logical_plan=LogicalPlan(
            [
                LogicalRename(
                    explanation="Rename reviews table for self-join.",
                    inputs=[VirtualTableIdentifier("reviews")],
                    output=VirtualTableIdentifier("reviews_other"),
                    expression="Rename {reviews.reviewtext} to [reviewtext_other]",
                    labels=None,
                ),
                LogicalJoin(
                    explanation=expression,
                    inputs=[
                        VirtualTableIdentifier("reviews"),
                        VirtualTableIdentifier("reviews_other"),
                    ],
                    output=VirtualTableIdentifier("output"),
                    expression=expression,
                    labels=None,
                ),
            ]
        ),
    )


MOVIE_JOIN_QUERIES = Queries(
    *[
        _make_join_query(opt.expression)
        for opt in MOVIE_OPERATOR_OPTIONS
        if opt.operator_type == LogicalJoin
    ]
)

MOVIE_QUERY_SHAPES = [
    QueryShape(
        RandomOrder(
            OperatorPlaceholder(
                LogicalFilter,
                inputs=[VirtualTableIdentifier("reviews")],
                output=VirtualTableIdentifier("intermediate"),
            ),
            OperatorPlaceholder(
                LogicalExtract,
                inputs=[VirtualTableIdentifier("intermediate")],
                output=VirtualTableIdentifier("output"),
            ),
        ),
        additional_info={
            "num_semops": 2,
            "num_sem_filter": 1,
            "num_sem_extract": 1,
        },
    ),
    QueryShape(
        RandomOrder(
            OperatorPlaceholder(
                LogicalFilter,
                inputs=[VirtualTableIdentifier("reviews")],
                output=VirtualTableIdentifier("intermediate"),
            ),
            OperatorPlaceholder(
                LogicalFilter,
                inputs=[VirtualTableIdentifier("intermediate")],
                output=VirtualTableIdentifier("output"),
            ),
        ),
        additional_info={
            "num_semops": 2,
            "num_sem_filter": 2,
            "num_sem_extract": 0,
        },
    ),
    QueryShape(
        RandomOrder(
            OperatorPlaceholder(
                LogicalExtract,
                inputs=[VirtualTableIdentifier("reviews")],
                output=VirtualTableIdentifier("intermediate"),
            ),
            OperatorPlaceholder(
                LogicalExtract,
                inputs=[VirtualTableIdentifier("intermediate")],
                output=VirtualTableIdentifier("output"),
            ),
        ),
        additional_info={
            "num_semops": 2,
            "num_sem_filter": 0,
            "num_sem_extract": 2,
        },
    ),
    QueryShape(
        RandomOrder(
            OperatorPlaceholder(
                LogicalFilter,
                inputs=[VirtualTableIdentifier("reviews")],
                output=VirtualTableIdentifier("intermediate1"),
            ),
            OperatorPlaceholder(
                LogicalExtract,
                inputs=[VirtualTableIdentifier("intermediate1")],
                output=VirtualTableIdentifier("intermediate2"),
            ),
            OperatorPlaceholder(
                LogicalExtract,
                inputs=[VirtualTableIdentifier("intermediate2")],
                output=VirtualTableIdentifier("output"),
            ),
        ),
        additional_info={
            "num_semops": 3,
            "num_sem_filter": 1,
            "num_sem_extract": 2,
        },
    ),
    QueryShape(
        RandomOrder(
            OperatorPlaceholder(
                LogicalFilter,
                inputs=[VirtualTableIdentifier("reviews")],
                output=VirtualTableIdentifier("intermediate1"),
            ),
            OperatorPlaceholder(
                LogicalExtract,
                inputs=[VirtualTableIdentifier("intermediate1")],
                output=VirtualTableIdentifier("intermediate2"),
            ),
            OperatorPlaceholder(
                LogicalExtract,
                inputs=[VirtualTableIdentifier("intermediate2")],
                output=VirtualTableIdentifier("intermediate3"),
            ),
            OperatorPlaceholder(
                LogicalExtract,
                inputs=[VirtualTableIdentifier("intermediate3")],
                output=VirtualTableIdentifier("output"),
            ),
        ),
        additional_info={
            "num_semops": 4,
            "num_sem_filter": 1,
            "num_sem_extract": 3,
        },
    ),
    QueryShape(
        LogicalRename(
            explanation="Rename reviews table for self-join.",
            inputs=[VirtualTableIdentifier("reviews")],
            output=VirtualTableIdentifier("reviews_other"),
            expression="Rename {reviews.reviewtext} to [reviewtext_other]",
            labels=None,
        ),
        OperatorPlaceholder(
            LogicalJoin,
            inputs=[
                VirtualTableIdentifier("reviews"),
                VirtualTableIdentifier("reviews_other"),
            ],
            output=VirtualTableIdentifier("output"),
        ),
        additional_info={
            "num_semops": 1,
            "num_sem_filter": 0,
            "num_sem_join": 1,
            "num_sem_extract": 0,
        },
    ),
    QueryShape(
        LogicalRename(
            explanation="Rename reviews table for self-join.",
            inputs=[VirtualTableIdentifier("reviews")],
            output=VirtualTableIdentifier("reviews_other"),
            expression="Rename {reviews.reviewtext} to [reviewtext_other]",
            labels=None,
        ),
        OperatorPlaceholder(
            LogicalFilter,
            inputs=[
                VirtualTableIdentifier("reviews"),
            ],
            output=VirtualTableIdentifier("filtered"),
        ),
        OperatorPlaceholder(
            LogicalJoin,
            inputs=[
                VirtualTableIdentifier("filtered"),
                VirtualTableIdentifier("reviews_other"),
            ],
            output=VirtualTableIdentifier("output"),
        ),
        additional_info={
            "num_semops": 2,
            "num_sem_filter": 1,
            "num_sem_join": 1,
            "num_sem_extract": 0,
        },
    ),
]

#: How many queries `movie_random_huge` keeps from each of the two join shapes.
HUGE_JOIN_QUERIES_KEPT = 5

#: `movie_random_huge` runs the same shapes as `movie_random` over 10,000 instead of
#: 1,000 reviews, so a self-join query has 10^8 candidate pairs and dominates the
#: workload; fewer join queries are kept.
#:
#: Expressed as `queries_kept` rather than a lower `num_queries_per_shape`, so the
#: shared RNG stream is untouched (see `RandomBenchmark.queries_kept_per_shape`).
MOVIE_QUERY_SHAPES_HUGE = [
    QueryShape(
        *shape.shape,
        additional_info=shape.additional_info,
        queries_kept=HUGE_JOIN_QUERIES_KEPT,
    )
    if LogicalJoin in shape.get_required_operators_per_type()
    else shape
    for shape in MOVIE_QUERY_SHAPES
]
