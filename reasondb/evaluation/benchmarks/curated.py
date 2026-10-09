"""Curated benchmarks whose every semantic step is hand-labeled.

``--human-labels`` measures precision/recall against per-tuple ground truth instead of
against the best model's own verdicts. That requires labelled predicates, which the
``*_random`` benchmarks (whose predicates are drawn from a generated operator pool) do
not have.

Every ``LogicalFilter`` and ``LogicalExtract`` here carries a :class:`LabelsDefinition`;
each query consists of one or two such steps, plus (for rotowire) traditional
``LogicalJoin`` (``equals``) steps that need no labels.
``tests/test_curated_benchmarks.py`` checks this property.

Each class subclasses its fixed counterpart and overrides only ``name()`` and
``get_queries()``; the table wiring, CSVs and column types are inherited.
``MovieHugeCurated`` also overrides ``get_csv()`` to use the 10000-review table. A distinct
``name()`` gives each one its own cache directory and DuckDB database.

**Label provenance.** For artwork, email and rotowire a human judged the predicate itself,
per tuple. ``ecommerce_curated``'s labels are derived by a deterministic mapping from the
Myntra catalog attributes shipped with SemBench, and ``movie_huge_curated``'s from the
critic's fresh/rotten score in SemBench's Rotten Tomatoes data -- an annotation of the
item rather than of the question. ``LABEL_PROVENANCE`` records this distinction;
``scripts/build_ecommerce_labels.py`` / ``scripts/build_movie_labels.py`` document which
predicates the mappings support.

**Usage**: run with ``--producer run_benchmark`` (names resolve through
``ALL_BENCHMARKS``), e.g.

    python scripts/run_coordinator.py --local --producer run_benchmark --task-id <task-id> \\
      --benchmarks artwork_curated --human-labels --device cuda:0

The parameter sweeps use ``RANDOM_BENCHMARKS`` only, since a sweep needs many comparable
queries per configuration.

**Extract labels and missing values.** ``PerfectExtract`` does
``result_labels.fillna(0)``, so a blank cell becomes a gold answer of ``0`` -- a box-score
stat the report never mentions is scored as though the report said zero. The rotowire
columns are therefore picked by label density over the full tuple space each step sees:

===================== ============ ==================================================
column                 labelled     used by
===================== ============ ==================================================
Points                 93%          players query 1
Total points           88%          teams query
Wins                   81%          teams query
Total rebounds         57%          players query 2
Assists                43%          players query 1
Steals                 22%          unused -- too sparse to score
Minutes played         18%          unused
Blocks                 13%          unused
Points in 4th quarter  11%          unused
===================== ============ ==================================================

For sparse columns most gold answers are the ``fillna`` zero, so agreement would mostly
measure whether an operator defaults to 0. All columns in use are labelled on a majority
of the tuple space except ``Assists`` at 43%.
"""

from pathlib import Path
from typing import Literal

from reasondb.database.indentifier import VirtualTableIdentifier
from reasondb.evaluation.benchmark import Benchmark, LabelsDefinition
from reasondb.evaluation.benchmarks.artwork import Artwork
from reasondb.evaluation.benchmarks.ecommerce import EcommerceLarge
from reasondb.evaluation.benchmarks.email import EnronEmail
from reasondb.evaluation.benchmarks.movie import Movie
from reasondb.evaluation.benchmarks.rotowire import Rotowire
from reasondb.query_plan.logical_plan import (
    LogicalExtract,
    LogicalFilter,
    LogicalJoin,
    LogicalPlan,
)
from reasondb.query_plan.query import Queries, Query

ARTWORK_LABELS = Path(
    "reasondb/evaluation/ground_truth/artwork/artwork_no_duplicated.csv"
)
EMAIL_LABELS = Path("reasondb/evaluation/ground_truth/emails/enron-eval.csv")
ROTOWIRE_PLAYER_LABELS = Path(
    "reasondb/evaluation/ground_truth/rotowire/rotowire_players_ground_truth.csv"
)
ROTOWIRE_TEAM_LABELS = Path(
    "reasondb/evaluation/ground_truth/rotowire/rotowire_teams_ground_truth.csv"
)
ECOMMERCE_LABELS = Path(
    "reasondb/evaluation/ground_truth/ecommerce/ecommerce_products.csv"
)
MOVIE_HUGE_LABELS = Path(
    "reasondb/evaluation/ground_truth/movie/movie_reviews_huge.csv"
)


# ── artwork: 65 paintings, image-only, all six label columns ─────────────────

ARTWORK_CURATED_QUERIES = Queries(
    # The only query that pairs a labelled filter with a labelled extract, and the only
    # reader of the `century` column.
    Query(
        "Which paintings depict Madonna and Child, and in which century were they created?",
        _ground_truth_logical_plan=LogicalPlan(
            [
                LogicalFilter(
                    explanation="First, we need to filter the paintings that depict Madonna and Child.",
                    inputs=[VirtualTableIdentifier("artworks")],
                    output=VirtualTableIdentifier("madonna_and_child"),
                    expression="{artworks.image} depicts Madonna and Child",
                    labels=LabelsDefinition(ARTWORK_LABELS, "m&c", ["artworks"]),
                ),
                # The boundary rule is stated explicitly because the labels use the strict
                # ordinal century `(year - 1) // 100 + 1` (1500 is the 15th century), which
                # differs from `year // 100 + 1` for years ending in 00.
                LogicalExtract(
                    explanation="We also need to extract the century from the inception date.",
                    inputs=[VirtualTableIdentifier("madonna_and_child")],
                    output=VirtualTableIdentifier("madonna_and_child_with_century"),
                    expression=(
                        "Extract the [century] from "
                        "{madonna_and_child.inception} as a plain number, counting "
                        "centuries from 1, where a year ending in 00 belongs to the "
                        "century before it: 1500 is 15 and 1501 is 16"
                    ),
                    labels=LabelsDefinition(ARTWORK_LABELS, "century", ["artworks"]),
                ),
            ]
        ),
    ),
    # A cascade of two labelled filters where the second strictly refines the first, so
    # errors in composing the guarantee across steps are observable.
    Query(
        "Which paintings depict more than two people, and of those, which depict more than three?",
        _ground_truth_logical_plan=LogicalPlan(
            [
                LogicalFilter(
                    explanation="First, we need to filter the paintings that depict more than two people.",
                    inputs=[VirtualTableIdentifier("artworks")],
                    output=VirtualTableIdentifier("more_than_two_people"),
                    expression="{artworks.image} depicts more than two people",
                    labels=LabelsDefinition(
                        ARTWORK_LABELS, "more_than_2_people", ["artworks"]
                    ),
                ),
                LogicalFilter(
                    explanation="Then, we narrow that down to paintings depicting more than three people.",
                    inputs=[VirtualTableIdentifier("more_than_two_people")],
                    output=VirtualTableIdentifier("more_than_three_people"),
                    expression="{more_than_two_people.image} depicts more than three people",
                    labels=LabelsDefinition(
                        ARTWORK_LABELS, "more_than_3_people", ["artworks"]
                    ),
                ),
            ]
        ),
    ),
    Query(
        "Which paintings depict saints identifiable by their halos in a scene where death is a dominant theme?",
        _ground_truth_logical_plan=LogicalPlan(
            [
                LogicalFilter(
                    explanation="First, we need to filter the paintings that depict saints identifiable by their halos.",
                    inputs=[VirtualTableIdentifier("artworks")],
                    output=VirtualTableIdentifier("saints_with_halos"),
                    expression="{artworks.image} depicts saints identifiable by their halos",
                    labels=LabelsDefinition(
                        ARTWORK_LABELS, "saints_with_halos", ["artworks"]
                    ),
                ),
                LogicalFilter(
                    explanation="Then, we keep those depicting a scene in which death is a dominant theme.",
                    inputs=[VirtualTableIdentifier("saints_with_halos")],
                    output=VirtualTableIdentifier("saints_and_death"),
                    expression="{saints_with_halos.image} depicts a scene in which death is a dominant theme",
                    labels=LabelsDefinition(
                        ARTWORK_LABELS, "death_theme", ["artworks"]
                    ),
                ),
            ]
        ),
    ),
)


# ── enron email: 1000 emails, one or two labelled steps over the same labels ──

# The same table and ground truth at one and at two semantic steps. Precision/recall
# targets are end-to-end, so these measure how the guarantee composes across operators.
#
# The prompts are taken verbatim from palimpzest
# (`tests/pytest/fixtures/workloads.py::enron_workload`), the source of this dataset and
# its labels. The full entity list matters: the four names reproduce `mentions_entity` on
# 997/1000 emails with zero false positives, while "Raptor" alone covers only 63 of the
# 79 positives.
#
# Known label quirk: eight positives refer to the Toronto Raptors; seven of these are
# `fraudulent=0` and are removed by the second filter of the two-step query.
#
# Each expression is self-contained, since every operator is profiled on its own
# expression without seeing the preceding step.
EMAIL_ENTITY_FILTER = (
    '{{{table}.text}} refers to a fraudulent scheme (i.e., "Raptor", "Deathstar", '
    '"Chewco", and/or "Fat Boy")'
)
EMAIL_FIRSTHAND_FILTER = (
    "{{{table}.text}} is not quoting from a news article or an article written by "
    "someone outside of Enron"
)

EMAIL_CURATED_QUERIES = Queries(
    Query(
        "Which E-Mails refer to a fraudulent scheme?",
        _ground_truth_logical_plan=LogicalPlan(
            [
                LogicalFilter(
                    explanation="We need the E-Mails that refer to a fraudulent scheme.",
                    inputs=[VirtualTableIdentifier("emails")],
                    output=VirtualTableIdentifier("entity_emails"),
                    expression=EMAIL_ENTITY_FILTER.format(table="emails"),
                    labels=LabelsDefinition(
                        EMAIL_LABELS, "mentions_entity", ["emails"]
                    ),
                ),
            ]
        ),
    ),
    Query(
        "Which E-Mails refer to a fraudulent scheme without quoting a news article or an outside source?",
        _ground_truth_logical_plan=LogicalPlan(
            [
                LogicalFilter(
                    explanation="First, the E-Mails that refer to a fraudulent scheme.",
                    inputs=[VirtualTableIdentifier("emails")],
                    output=VirtualTableIdentifier("entity_emails"),
                    expression=EMAIL_ENTITY_FILTER.format(table="emails"),
                    labels=LabelsDefinition(
                        EMAIL_LABELS, "mentions_entity", ["emails"]
                    ),
                ),
                LogicalFilter(
                    explanation="Then, the ones not quoting a news article or an outside author.",
                    inputs=[VirtualTableIdentifier("entity_emails")],
                    output=VirtualTableIdentifier("firsthand_emails"),
                    expression=EMAIL_FIRSTHAND_FILTER.format(table="entity_emails"),
                    labels=LabelsDefinition(EMAIL_LABELS, "fraudulent", ["emails"]),
                ),
            ]
        ),
    ),
    # `fraudulent` is only meaningful after the entity filter: it was annotated within
    # the entity-filtered subset, so it never leads a query.
    Query(
        "Who sent the E-Mails that refer to a fraudulent scheme?",
        _ground_truth_logical_plan=LogicalPlan(
            [
                LogicalFilter(
                    explanation="First, the E-Mails referring to a fraudulent scheme.",
                    inputs=[VirtualTableIdentifier("emails")],
                    output=VirtualTableIdentifier("scheme_emails"),
                    expression=EMAIL_ENTITY_FILTER.format(table="emails"),
                    labels=LabelsDefinition(
                        EMAIL_LABELS, "mentions_entity", ["emails"]
                    ),
                ),
                # The expression names the *form* of the answer: `sender` is labelled as a
                # bare e-mail address, while From: lines may also carry a display name.
                # This also serves as the task description for a generated parser
                # (`CodegenExtract`, written from a few sample rows), so stating both
                # formats keeps the parser independent of which rows were sampled.
                LogicalExtract(
                    explanation="Then, we extract the sender's e-mail address from those E-Mails.",
                    inputs=[VirtualTableIdentifier("scheme_emails")],
                    output=VirtualTableIdentifier("scheme_senders"),
                    expression=(
                        "Extract the e-mail address of the [sender] from "
                        "{scheme_emails.text}. Take it from the From: line, which is "
                        "written either as a bare address such as jane.doe@enron.com or "
                        "as a display name followed by the address in angle brackets "
                        "such as Jane Doe <jane.doe@enron.com>. Return only the address "
                        "itself, never the display name"
                    ),
                    labels=LabelsDefinition(EMAIL_LABELS, "sender", ["emails"]),
                ),
            ]
        ),
    ),
)


# ── rotowire: extracts over joined tables, players and teams ─────────────────


def _player_stat_query(question: str, extracts) -> Query:
    """A players query: join players->games->reports, then one labelled extract per stat.

    The two joins are traditional (``equals``) and carry no labels. The extracts are keyed
    by ``["players", "reports"]``, i.e. ``_index_players`` + ``_index_reports``, which is
    what ``rotowire_players_ground_truth.csv`` provides -- one row per (player, game),
    4699 of them, exactly matching ``players_to_games.csv``.

    ``extracts`` is ``(field, slug, label column)``: the field is the prompt's wording and
    may contain spaces ("total rebounds"); the slug names the intermediate table and may
    not, since it becomes a DuckDB identifier and is matched by the ``{table.column}``
    placeholder regexes.
    """
    steps = [
        LogicalJoin(
            explanation="First, we need to join the players and players_to_games tables.",
            inputs=[
                VirtualTableIdentifier("players"),
                VirtualTableIdentifier("players_to_games"),
            ],
            output=VirtualTableIdentifier("joined_players_games"),
            expression="{players.name} equals {players_to_games.name}",
        ),
        LogicalJoin(
            explanation="Then, we need to join the result with the reports table.",
            inputs=[
                VirtualTableIdentifier("joined_players_games"),
                VirtualTableIdentifier("reports"),
            ],
            output=VirtualTableIdentifier("joined_all"),
            expression="{joined_players_games.game_id} equals {reports.game_id}",
        ),
    ]
    source = "joined_all"
    for field, slug, column in extracts:
        output = f"with_{slug}"
        steps.append(
            LogicalExtract(
                explanation=f"Next, we extract the {field} recorded by each player according to the report.",
                inputs=[VirtualTableIdentifier(source)],
                output=VirtualTableIdentifier(output),
                expression=(
                    f"Extract the [{field}] from {{{source}.report}} "
                    f"for each {{{source}.name}}"
                ),
                labels=LabelsDefinition(
                    ROTOWIRE_PLAYER_LABELS, column, ["players", "reports"]
                ),
            )
        )
        source = output
    return Query(question, _ground_truth_logical_plan=LogicalPlan(steps))


ROTOWIRE_CURATED_QUERIES = Queries(
    _player_stat_query(
        "Which players scored how many points and assists in which games?",
        [("points", "points", "Points"), ("assists", "assists", "Assists")],
    ),
    # A single extract: the remaining stats are too sparsely labelled to score (see the
    # module docstring).
    _player_stat_query(
        "How many rebounds did each player record in each game?",
        [("total rebounds", "total_rebounds", "Total rebounds")],
    ),
    # The teams side, reading the label file re-keyed by
    # scripts/build_rotowire_team_labels.py. Same plan shape as the players queries, one
    # table over: teams -> teams_to_games -> reports, keyed ["reports", "teams"].
    Query(
        "How many total points did each team score, and how many games had they won at that point in the season?",
        _ground_truth_logical_plan=LogicalPlan(
            [
                LogicalJoin(
                    explanation="First, we need to join the teams and teams_to_games tables.",
                    inputs=[
                        VirtualTableIdentifier("teams"),
                        VirtualTableIdentifier("teams_to_games"),
                    ],
                    output=VirtualTableIdentifier("joined_teams_games"),
                    expression="{teams.name} equals {teams_to_games.name}",
                ),
                LogicalJoin(
                    explanation="Then, we need to join the result with the reports table.",
                    inputs=[
                        VirtualTableIdentifier("joined_teams_games"),
                        VirtualTableIdentifier("reports"),
                    ],
                    output=VirtualTableIdentifier("joined_team_reports"),
                    expression="{joined_teams_games.game_id} equals {reports.game_id}",
                ),
                LogicalExtract(
                    explanation="Next, we extract the total points scored by each team according to the report.",
                    inputs=[VirtualTableIdentifier("joined_team_reports")],
                    output=VirtualTableIdentifier("with_total_points"),
                    expression=(
                        "Extract the [total points] from {joined_team_reports.report} "
                        "for each {joined_team_reports.name}"
                    ),
                    labels=LabelsDefinition(
                        ROTOWIRE_TEAM_LABELS, "Total points", ["reports", "teams"]
                    ),
                ),
                LogicalExtract(
                    explanation="Next, we extract each team's win count according to the report.",
                    inputs=[VirtualTableIdentifier("with_total_points")],
                    output=VirtualTableIdentifier("with_wins"),
                    expression=(
                        "Extract the [wins] from {with_total_points.report} "
                        "for each {with_total_points.name}"
                    ),
                    labels=LabelsDefinition(
                        ROTOWIRE_TEAM_LABELS, "Wins", ["reports", "teams"]
                    ),
                ),
            ]
        ),
    ),
)


# ── ecommerce: 1000 products, labels derived from the SemBench catalog ───────

# Labels are derived by `scripts/build_ecommerce_labels.py` from the Myntra catalog
# attributes shipped with SemBench (articleType, gender, subCategory, masterCategory):
# non-model ground truth, but an annotation of the *item* rather than of the question, so
# results measure agreement with the catalog.
#
# Only predicates with a faithful catalog counterpart are used. Where a predicate and a
# catalog attribute nearly agree, the question is phrased to match the attribute (e.g.
# the t-shirt query asks for the union `Tshirts` + `Tops`).
ECOMMERCE_CURATED_QUERIES = Queries(
    # `articleType` separates `Tshirts` (polos included) from `Tops`, a distinction a
    # product photo does not carry, so the question asks for their union.
    Query(
        "Which images show t-shirts, polo shirts or tops?",
        _ground_truth_logical_plan=LogicalPlan(
            [
                LogicalFilter(
                    explanation="We need to filter the products whose image shows a t-shirt, polo shirt or top.",
                    inputs=[VirtualTableIdentifier("products")],
                    output=VirtualTableIdentifier("tshirts"),
                    expression="{products.product_image} shows a t-shirt, a polo shirt or a top",
                    labels=LabelsDefinition(
                        ECOMMERCE_LABELS, "is_tshirt_or_top", ["products"]
                    ),
                ),
            ]
        ),
    ),
    # Control query: `masterCategory == "Footwear"` (220 of 1000 products) leaves no
    # definitional ambiguity between the catalog and the image.
    Query(
        "Which images show footwear?",
        _ground_truth_logical_plan=LogicalPlan(
            [
                LogicalFilter(
                    explanation="We need to filter the products whose image shows footwear.",
                    inputs=[VirtualTableIdentifier("products")],
                    output=VirtualTableIdentifier("footwear"),
                    expression="{products.product_image} shows footwear, i.e. a shoe, a sandal or a flip-flop",
                    labels=LabelsDefinition(
                        ECOMMERCE_LABELS, "is_footwear", ["products"]
                    ),
                ),
            ]
        ),
    ),
    # The less selective filter comes first (`is_menswear` keeps 53%, `is_tshirt_or_top`
    # 19%), leaving room for operator reordering.
    Query(
        "Which images show t-shirts, polo shirts or tops intended for a male audience?",
        _ground_truth_logical_plan=LogicalPlan(
            [
                LogicalFilter(
                    explanation="First, we need to filter the products intended for a male audience.",
                    inputs=[VirtualTableIdentifier("products")],
                    output=VirtualTableIdentifier("menswear"),
                    expression="{products.product_image} shows an item intended for a male audience",
                    labels=LabelsDefinition(
                        ECOMMERCE_LABELS, "is_menswear", ["products"]
                    ),
                ),
                LogicalFilter(
                    explanation="Then, we keep the ones whose image shows a t-shirt, polo shirt or top.",
                    inputs=[VirtualTableIdentifier("menswear")],
                    output=VirtualTableIdentifier("menswear_tshirts"),
                    expression="{menswear.product_image} shows a t-shirt, a polo shirt or a top",
                    labels=LabelsDefinition(
                        ECOMMERCE_LABELS, "is_tshirt_or_top", ["products"]
                    ),
                ),
            ]
        ),
    ),
    Query(
        "Which images show items designed to carry or hold objects?",
        _ground_truth_logical_plan=LogicalPlan(
            [
                LogicalFilter(
                    explanation="We need to filter the products whose image shows an item for carrying things.",
                    inputs=[VirtualTableIdentifier("products")],
                    output=VirtualTableIdentifier("carriers"),
                    expression="{products.product_image} shows an item designed to carry or hold objects",
                    labels=LabelsDefinition(
                        ECOMMERCE_LABELS, "carries_objects", ["products"]
                    ),
                ),
            ]
        ),
    ),
)


# ── movie: 10000 reviews, labels derived from the SemBench Rotten Tomatoes dump ──

# Labels are derived by `scripts/build_movie_labels.py` from `scoreSentiment` in
# SemBench's Rotten Tomatoes data (whether the critic's score was fresh or rotten):
# non-model ground truth, but an annotation of the score rather than of the question.
#
# The wording ("favorable overall verdict") differs from `movie.py`'s pool ("is clearly
# positive") because the label records no intensity. As a consequence, a
# `movie_random_huge` precompute store does not cover these expressions; this benchmark
# needs its own.
#
# `is_positive` and `is_negative` encode the same bit; it is the only attribute of the
# data that covers every row, as a label file must.
MOVIE_HUGE_CURATED_QUERIES = Queries(
    # Majority class (66%): the precision target is the binding one. Query 2 is the
    # complement.
    Query(
        "Which reviews give the movie a favorable verdict?",
        _ground_truth_logical_plan=LogicalPlan(
            [
                LogicalFilter(
                    explanation="We need the reviews whose text expresses a favorable verdict.",
                    inputs=[VirtualTableIdentifier("reviews")],
                    output=VirtualTableIdentifier("favorable_reviews"),
                    expression="{reviews.reviewtext} expresses a favorable overall verdict on the movie",
                    labels=LabelsDefinition(
                        MOVIE_HUGE_LABELS, "is_positive", ["reviews"]
                    ),
                ),
            ]
        ),
    ),
    # The complement (34%): together with query 1 this measures how the guarantee behaves
    # against base rate with the operator's difficulty held fixed.
    Query(
        "Which reviews give the movie an unfavorable verdict?",
        _ground_truth_logical_plan=LogicalPlan(
            [
                LogicalFilter(
                    explanation="We need the reviews whose text expresses an unfavorable verdict.",
                    inputs=[VirtualTableIdentifier("reviews")],
                    output=VirtualTableIdentifier("unfavorable_reviews"),
                    expression="{reviews.reviewtext} expresses an unfavorable overall verdict on the movie",
                    labels=LabelsDefinition(
                        MOVIE_HUGE_LABELS, "is_negative", ["reviews"]
                    ),
                ),
            ]
        ),
    ),
)


# ── the benchmarks ───────────────────────────────────────────────────────────


class ArtworkCurated(Artwork):
    """65 paintings, three fully-labeled image queries over all six label columns."""

    @classmethod
    def name(cls) -> str:
        return "artwork_curated"

    @staticmethod
    def get_queries() -> Queries:
        return ARTWORK_CURATED_QUERIES


class EnronEmailCurated(EnronEmail):
    """1000 emails at one and two labelled steps over the same ground truth."""

    @classmethod
    def name(cls) -> str:
        return "email_curated"

    @staticmethod
    def get_queries() -> Queries:
        return EMAIL_CURATED_QUERIES


class RotowireCurated(Rotowire):
    """Labelled extracts over joined tables, on both the player and the team side."""

    @classmethod
    def name(cls) -> str:
        return "rotowire_curated"

    @staticmethod
    def get_queries() -> Queries:
        return ROTOWIRE_CURATED_QUERIES


class EcommerceCurated(EcommerceLarge):
    """1000 products, three labelled image filters over catalog-derived ground truth.

    `has_ground_truth` is overridden to True: unlike the parent's query set, every step
    here has a label source.
    """

    @classmethod
    def name(cls) -> str:
        return "ecommerce_curated"

    @property
    def has_ground_truth(self) -> bool:
        return True

    @staticmethod
    def get_queries() -> Queries:
        return ECOMMERCE_CURATED_QUERIES


class MovieHugeCurated(Movie):
    """10000 reviews, labelled text queries over the SemBench sentiment signal.

    Overrides `get_csv` as well as `name`/`get_queries`, since `Movie` uses the 100-review
    table and `MovieRandomHuge` (which owns `reviews_10000.csv`) is a `RandomBenchmark`.
    `has_ground_truth` is True because every step here has a label source.

    The table is `movie_random_huge`'s but the benchmark name is not, so KV caches live
    under `{CACHE_DIR}/movie_huge_curated_dev/`. Cache filenames are content hashes, so
    that directory can point at the existing `movie_random_huge` caches.
    """

    @classmethod
    def name(cls) -> str:
        return "movie_huge_curated"

    @property
    def has_ground_truth(self) -> bool:
        return True

    @staticmethod
    def get_csv():
        return Path(__file__).parent / "files" / "reviews_10000.csv"

    @staticmethod
    def get_queries() -> Queries:
        return MOVIE_HUGE_CURATED_QUERIES


#: Every curated benchmark, for the registry and for the tests that assert the
#: fully-labeled property across all of them at once.
CURATED_BENCHMARKS = (
    ArtworkCurated,
    EnronEmailCurated,
    RotowireCurated,
    EcommerceCurated,
    MovieHugeCurated,
)

#: How each benchmark's labels were produced. ``annotated`` -- a human judged the
#: predicate itself, per tuple. ``derived`` -- a deterministic, non-model mapping from
#: attributes a human wrote for another purpose. Both are valid references for
#: ``--human-labels``, which only requires that the labels not come from a model.
LABEL_PROVENANCE = {
    "artwork_curated": "annotated",
    "email_curated": "annotated",
    "rotowire_curated": "annotated",
    "ecommerce_curated": "derived",
    "movie_huge_curated": "derived",
}


def semantic_steps(benchmark_cls) -> list:
    """Every step of *benchmark_cls*'s queries that a model would have to answer.

    Filters and extracts reach ``run_outside_db``; joins, projects and renames are
    traditional and are pushed into SQL, so they neither have nor need ground truth.
    """
    return [
        step
        for query in benchmark_cls.get_queries()
        for step in query.get_gt_logical_plan().plan_steps
        if isinstance(step, (LogicalFilter, LogicalExtract))
    ]
