import logging
from pathlib import Path
from typing import Dict, Literal, Sequence, Union
from reasondb.database.database import ExperimentalDatabase
from reasondb.database.indentifier import (
    InPlaceColumn,
    VirtualTableIdentifier,
)
from reasondb.evaluation.benchmark import Benchmark, LabelsDefinition, RandomBenchmark
from reasondb.query_plan.logical_plan import (
    LogicalExtract,
    LogicalFilter,
    LogicalJoin,
    LogicalPlan,
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


ROTOWIRE_QUERIES = Queries(
    Query(
        "Which players scored how many points and assists in which games?",
        _ground_truth_logical_plan=LogicalPlan(
            [
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
                LogicalExtract(
                    explanation="Next, we need to extract the points scored by each player according to the report.",
                    inputs=[VirtualTableIdentifier("joined_all")],
                    output=VirtualTableIdentifier("with_points"),
                    expression="Extract the [points] from {joined_all.report} for each {joined_all.name}",
                    labels=LabelsDefinition(
                        Path(
                            "reasondb/evaluation/ground_truth/rotowire/rotowire_players_ground_truth.csv"
                        ),
                        "Points",
                        ["players", "reports"],
                    ),
                ),
                LogicalExtract(
                    explanation="Next, we need to extract the assists made by each player according to the report.",
                    inputs=[VirtualTableIdentifier("with_points")],
                    output=VirtualTableIdentifier("with_points_and_assists"),
                    expression="Extract the [assists] from {with_points.report} for each {with_points.name}",
                    labels=LabelsDefinition(
                        Path(
                            "reasondb/evaluation/ground_truth/rotowire/rotowire_players_ground_truth.csv"
                        ),
                        "Assists",
                        ["players", "reports"],
                    ),
                ),
            ]
        ),
    ),
)


class Rotowire(Benchmark):
    @staticmethod
    def urls():
        return {}

    @staticmethod
    def get_queries() -> Queries:
        return ROTOWIRE_QUERIES

    @property
    def has_ground_truth(self) -> bool:
        return True

    @staticmethod
    def download(split: Literal["train", "dev", "test"]) -> Benchmark:
        return Rotowire.load_from_disk(split)

    @staticmethod
    def load_from_disk(split: Literal["train", "dev", "test"]) -> "Benchmark":
        (
            players_csv,
            teams_csv,
            reports_csv,
            players_to_games_csv,
            teams_to_games_csv,
        ) = (
            Path(__file__).parent / "files" / filename
            for filename in [
                "players.csv",
                "teams.csv",
                "reports.csv",
                "players_to_games.csv",
                "teams_to_games.csv",
            ]
        )
        benchmark = Rotowire(
            split,
            ExperimentalDatabase.load_from_files(
                db_name=Rotowire.name(),
                split=split,
                table_names=[
                    "players",
                    "teams",
                    "reports",
                    "players_to_games",
                    "teams_to_games",
                ],
                paths=[
                    players_csv,
                    teams_csv,
                    reports_csv,
                    players_to_games_csv,
                    teams_to_games_csv,
                ],
                text_columns=[InPlaceColumn("reports.report")],
            ),
            Rotowire.get_queries(),
        )
        return benchmark


class RotowireRandom(RandomBenchmark):
    @classmethod
    def name(cls) -> str:
        return "rotowire_random"

    @property
    def has_ground_truth(self) -> bool:
        return False

    @staticmethod
    def urls():
        return {}

    @staticmethod
    def download(split: Literal["train", "dev", "test"]) -> Benchmark:
        return RotowireRandom.load_from_disk(split)

    @staticmethod
    def load_from_disk(split: Literal["train", "dev", "test"]) -> "Benchmark":
        (
            players_csv,
            teams_csv,
            reports_csv,
            players_to_games_csv,
            teams_to_games_csv,
        ) = (
            Path(__file__).parent / "files" / filename
            for filename in [
                "players.csv",
                "teams.csv",
                "reports.csv",
                "players_to_games.csv",
                "teams_to_games.csv",
            ]
        )
        benchmark = RotowireRandom(
            split,
            ExperimentalDatabase.load_from_files(
                db_name=RotowireRandom.name(),
                split=split,
                table_names=[
                    "players",
                    "teams",
                    "reports",
                    "players_to_games",
                    "teams_to_games",
                ],
                paths=[
                    players_csv,
                    teams_csv,
                    reports_csv,
                    players_to_games_csv,
                    teams_to_games_csv,
                ],
                text_columns=[InPlaceColumn("reports.report")],
            ),
            RotowireRandom.generate_random_queries(split, num_queries_per_shape=5),
        )
        return benchmark

    @classmethod
    def _get_query_shapes(
        cls,
    ) -> Union[Sequence[QueryShape], Dict[str, Sequence[QueryShape]]]:
        return {
            "teams": ROTOWIRE_TEAMS_QUERY_SHAPES,
            "players": ROTOWIRE_PLAYERS_QUERY_SHAPES,
        }

    @classmethod
    def _get_operator_options(
        cls,
    ) -> Union[Sequence[OperatorOption], Dict[str, Sequence[OperatorOption]]]:
        return {
            "teams": ROTOWIRE_TEAMS_OPERATOR_OPTIONS,
            "players": ROTOWIRE_PLAYERS_OPERATOR_OPTIONS,
        }

    @classmethod
    def _single_filter_shape(cls) -> Dict[str, QueryShape]:
        return {
            "teams": ROTOWIRE_TEAMS_SINGLE_FILTER_SHAPE,
            "players": ROTOWIRE_PLAYERS_SINGLE_FILTER_SHAPE,
        }


# "Losses", "Total points", "Points in 4th quarter", "Wins", "Percentage of field goals",
# "Rebounds", "Number of team assists", "Points in 3rd quarter", "Turnovers", "Percentage of 3 points",
# "Points in 1st quarter", "Points in 2nd quarter"
ROTOWIRE_TEAMS_OPERATOR_OPTIONS = [
    OperatorOption(
        LogicalExtract,
        "From {joined_all.report} extract the number of losses [Losses] for {joined_all.name} ",
    ),
    OperatorOption(
        LogicalExtract,
        "From {joined_all.report} extract the total points [Total points] for {joined_all.name} ",
    ),
    OperatorOption(
        LogicalExtract,
        "From {joined_all.report} extract the number of points in 4th quarter [Points_in_4th_quarter] for {joined_all.name} ",
    ),
    OperatorOption(
        LogicalExtract,
        "From {joined_all.report} extract the number of wins [Wins] for {joined_all.name} ",
    ),
    OperatorOption(
        LogicalExtract,
        "From {joined_all.report} extract the percentage of field goals [Percentage_of_field_goals] for {joined_all.name} ",
    ),
    OperatorOption(
        LogicalExtract,
        "From {joined_all.report} extract the number of rebounds [Rebounds] for {joined_all.name} ",
    ),
    OperatorOption(
        LogicalExtract,
        "From {joined_all.report} extract the number of team assists [Number_of_team_assists] for {joined_all.name} ",
    ),
    OperatorOption(
        LogicalExtract,
        "From {joined_all.report} extract the number of points in 3rd quarter [Points_in_3rd_quarter] for {joined_all.name} ",
    ),
    OperatorOption(
        LogicalExtract,
        "From {joined_all.report} extract the number of turnovers [Turnovers] for {joined_all.name} ",
    ),
    OperatorOption(
        LogicalExtract,
        "From {joined_all.report} extract the percentage of 3 points [Percentage_of_3_points] for {joined_all.name} ",
    ),
    OperatorOption(
        LogicalExtract,
        "From {joined_all.report} extract the number of points in 1st quarter [Points_in_1st_quarter] for {joined_all.name} ",
    ),
    OperatorOption(
        LogicalExtract,
        "From {joined_all.report} extract the number of points in 2nd quarter [Points_in_2nd_quarter] for {joined_all.name} ",
    ),
    OperatorOption(
        LogicalFilter,
        "{joined_all.name} won the game according to {joined_all.report}",
    ),
    OperatorOption(
        LogicalFilter,
        "{joined_all.name} lost the game according to {joined_all.report}",
    ),
    OperatorOption(
        LogicalFilter,
        "{joined_all.name} scored more than 100 total points according to {joined_all.report}",
    ),
    OperatorOption(
        LogicalFilter,
        "{joined_all.name} had more than 20 rebounds according to {joined_all.report}",
    ),
    OperatorOption(
        LogicalFilter,
        "{joined_all.name} had more than 25 assists according to {joined_all.report}",
    ),
    OperatorOption(
        LogicalFilter,
        "{joined_all.name} had more than 15 turnovers according to {joined_all.report}",
    ),
    OperatorOption(
        LogicalFilter,
        "{joined_all.name} was the favorite team according to {joined_all.report}",
    ),
    OperatorOption(
        LogicalFilter,
        "{joined_all.name} was the underdog team according to {joined_all.report}",
    ),
    OperatorOption(
        LogicalFilter,
        "{joined_all.name} scored more than 30 points in the 4th quarter according to {joined_all.report}",
    ),
    OperatorOption(
        LogicalFilter,
        "{joined_all.name} scored more than 25 points in the 3rd quarter according to {joined_all.report}",
    ),
    OperatorOption(
        LogicalFilter,
        "{joined_all.name} was behind at halftime according to {joined_all.report}",
    ),
    OperatorOption(
        LogicalFilter,
        "{joined_all.name} was leading at halftime according to {joined_all.report}",
    ),
]

# "Assists", "Points", "Total rebounds", "Steals", "Defensive rebounds",
# "Field goals attempted", "Field goals made", "Free throws attempted", "Free throws made",
# "Minutes played", "Personal fouls", "Turnovers", "Blocks", "Offensive rebounds", "Field goal percentage",
# "Free throw percentage"
ROTOWIRE_PLAYERS_OPERATOR_OPTIONS = [
    # extract
    OperatorOption(
        LogicalExtract,
        "From {joined_all.report} extract the number of assists [Assists] for {joined_all.name} ",
    ),
    OperatorOption(
        LogicalExtract,
        "From {joined_all.report} extract the number of points [Points] for {joined_all.name} ",
    ),
    OperatorOption(
        LogicalExtract,
        "From {joined_all.report} extract the number of total rebounds [Total_rebounds] for {joined_all.name} ",
    ),
    OperatorOption(
        LogicalExtract,
        "From {joined_all.report} extract the number of steals [Steals] for {joined_all.name} ",
    ),
    OperatorOption(
        LogicalExtract,
        "From {joined_all.report} extract the number of defensive rebounds [Defensive_rebounds] for {joined_all.name} ",
    ),
    OperatorOption(
        LogicalExtract,
        "From {joined_all.report} extract the number of field goals attempted [Field_goals_attempted] for {joined_all.name} ",
    ),
    OperatorOption(
        LogicalExtract,
        "From {joined_all.report} extract the number of field goals made [Field_goals_made] for {joined_all.name} ",
    ),
    OperatorOption(
        LogicalExtract,
        "From {joined_all.report} extract the number of free throws attempted [Free_throws_attempted] for {joined_all.name} ",
    ),
    OperatorOption(
        LogicalExtract,
        "From {joined_all.report} extract the number of free throws made [Free_throws_made] for {joined_all.name} ",
    ),
    OperatorOption(
        LogicalExtract,
        "From {joined_all.report} extract the number of minutes played [Minutes_played] for {joined_all.name} ",
    ),
    OperatorOption(
        LogicalExtract,
        "From {joined_all.report} extract the number of personal fouls [Personal_fouls] for {joined_all.name} ",
    ),
    OperatorOption(
        LogicalExtract,
        "From {joined_all.report} extract the number of turnovers [Turnovers] for {joined_all.name} ",
    ),
    OperatorOption(
        LogicalExtract,
        "From {joined_all.report} extract the number of blocks [Blocks] for {joined_all.name} ",
    ),
    OperatorOption(
        LogicalExtract,
        "From {joined_all.report} extract the number of offensive rebounds [Offensive_rebounds] for {joined_all.name} ",
    ),
    OperatorOption(
        LogicalExtract,
        "From {joined_all.report} extract the field goal percentage [Field_goal_percentage] for {joined_all.name} ",
    ),
    OperatorOption(
        LogicalExtract,
        "From {joined_all.report} extract the free throw percentage [Free_throw_percentage] for {joined_all.name} ",
    ),
    # filter
    OperatorOption(
        LogicalFilter,
        "{joined_all.name} had more than 10 assists according to {joined_all.report}",
    ),
    OperatorOption(
        LogicalFilter,
        "{joined_all.name} scored more than 20 points according to {joined_all.report}",
    ),
    OperatorOption(
        LogicalFilter,
        "{joined_all.name} had more than 5 rebounds according to {joined_all.report}",
    ),
    OperatorOption(
        LogicalFilter,
        "{joined_all.name} had more than 3 steals according to {joined_all.report}",
    ),
    OperatorOption(
        LogicalFilter,
        "{joined_all.name} had more than 2 blocks according to {joined_all.report}",
    ),
    OperatorOption(
        LogicalFilter,
        "{joined_all.name} played in the winning team according to {joined_all.report}",
    ),
    OperatorOption(
        LogicalFilter,
        "{joined_all.name} played in the loosing team according to {joined_all.report}",
    ),
    OperatorOption(
        LogicalFilter,
        "{joined_all.name} was part of the starting lineup according to {joined_all.report}",
    ),
    OperatorOption(
        LogicalFilter,
        "{joined_all.name} attempted more than 15 field goals according to {joined_all.report}",
    ),
    OperatorOption(
        LogicalFilter,
        "{joined_all.name} played more than 30 minutes according to {joined_all.report}",
    ),
    OperatorOption(
        LogicalFilter,
        "{joined_all.name} had more than 5 turnovers according to {joined_all.report}",
    ),
]

ROTOWIRE_PLAYERS_SINGLE_FILTER_SHAPE = QueryShape(
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
    OperatorPlaceholder(
        LogicalFilter,
        inputs=[VirtualTableIdentifier("joined_all")],
        output=VirtualTableIdentifier("output"),
    ),
    additional_info={
        "num_semops": 1,
        "num_sem_filter": 1,
        "num_tradops": 2,
    },
)

ROTOWIRE_TEAMS_SINGLE_FILTER_SHAPE = QueryShape(
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
        output=VirtualTableIdentifier("joined_all"),
        expression="{joined_teams_games.game_id} equals {reports.game_id}",
    ),
    OperatorPlaceholder(
        LogicalFilter,
        inputs=[VirtualTableIdentifier("joined_all")],
        output=VirtualTableIdentifier("output"),
    ),
    additional_info={
        "num_semops": 1,
        "num_sem_filter": 1,
        "num_tradops": 2,
    },
)


ROTOWIRE_PLAYERS_QUERY_SHAPES = [
    QueryShape(
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
        RandomOrder(
            OperatorPlaceholder(
                LogicalFilter,
                inputs=[VirtualTableIdentifier("joined_all")],
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
            "num_tradops": 2,
        },
    ),
    QueryShape(
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
        RandomOrder(
            OperatorPlaceholder(
                LogicalFilter,
                inputs=[VirtualTableIdentifier("joined_all")],
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
            "num_tradops": 2,
        },
    ),
    QueryShape(
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
        RandomOrder(
            OperatorPlaceholder(
                LogicalFilter,
                inputs=[VirtualTableIdentifier("joined_all")],
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
            "num_tradops": 2,
        },
    ),
    QueryShape(
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
        RandomOrder(
            OperatorPlaceholder(
                LogicalFilter,
                inputs=[VirtualTableIdentifier("joined_all")],
                output=VirtualTableIdentifier("intermediate1"),
            ),
            OperatorPlaceholder(
                LogicalFilter,
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
            "num_sem_filter": 2,
            "num_sem_extract": 1,
            "num_tradops": 2,
        },
    ),
    QueryShape(
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
        RandomOrder(
            OperatorPlaceholder(
                LogicalFilter,
                inputs=[VirtualTableIdentifier("joined_all")],
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
            "num_tradops": 2,
        },
    ),
    QueryShape(
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
        RandomOrder(
            OperatorPlaceholder(
                LogicalFilter,
                inputs=[VirtualTableIdentifier("joined_all")],
                output=VirtualTableIdentifier("intermediate1"),
            ),
            OperatorPlaceholder(
                LogicalFilter,
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
            "num_sem_filter": 2,
            "num_sem_extract": 2,
            "num_tradops": 2,
        },
    ),
]

ROTOWIRE_TEAMS_QUERY_SHAPES = [
    QueryShape(
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
            output=VirtualTableIdentifier("joined_all"),
            expression="{joined_teams_games.game_id} equals {reports.game_id}",
        ),
        RandomOrder(
            OperatorPlaceholder(
                LogicalFilter,
                inputs=[VirtualTableIdentifier("joined_all")],
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
            "num_tradops": 2,
        },
    ),
    QueryShape(
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
            output=VirtualTableIdentifier("joined_all"),
            expression="{joined_teams_games.game_id} equals {reports.game_id}",
        ),
        RandomOrder(
            OperatorPlaceholder(
                LogicalFilter,
                inputs=[VirtualTableIdentifier("joined_all")],
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
            "num_tradops": 2,
        },
    ),
    QueryShape(
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
            output=VirtualTableIdentifier("joined_all"),
            expression="{joined_teams_games.game_id} equals {reports.game_id}",
        ),
        RandomOrder(
            OperatorPlaceholder(
                LogicalFilter,
                inputs=[VirtualTableIdentifier("joined_all")],
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
            "num_tradops": 2,
        },
    ),
    QueryShape(
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
            output=VirtualTableIdentifier("joined_all"),
            expression="{joined_teams_games.game_id} equals {reports.game_id}",
        ),
        RandomOrder(
            OperatorPlaceholder(
                LogicalFilter,
                inputs=[VirtualTableIdentifier("joined_all")],
                output=VirtualTableIdentifier("intermediate1"),
            ),
            OperatorPlaceholder(
                LogicalFilter,
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
            "num_sem_filter": 2,
            "num_sem_extract": 1,
            "num_tradops": 2,
        },
    ),
    QueryShape(
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
            output=VirtualTableIdentifier("joined_all"),
            expression="{joined_teams_games.game_id} equals {reports.game_id}",
        ),
        RandomOrder(
            OperatorPlaceholder(
                LogicalFilter,
                inputs=[VirtualTableIdentifier("joined_all")],
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
            "num_tradops": 2,
        },
    ),
    QueryShape(
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
            output=VirtualTableIdentifier("joined_all"),
            expression="{joined_teams_games.game_id} equals {reports.game_id}",
        ),
        RandomOrder(
            OperatorPlaceholder(
                LogicalFilter,
                inputs=[VirtualTableIdentifier("joined_all")],
                output=VirtualTableIdentifier("intermediate1"),
            ),
            OperatorPlaceholder(
                LogicalFilter,
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
            "num_semops": 3,
            "num_sem_filter": 2,
            "num_sem_extract": 2,
            "num_tradops": 2,
        },
    ),
]
