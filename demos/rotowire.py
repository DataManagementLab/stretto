import logging
from pathlib import Path
import warnings
from reasondb.database.indentifier import InPlaceColumn
from reasondb.interface.connect import RaccoonDB
from reasondb.optimizer.guarantees import PrecisionGuarantee, RecallGuarantee

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

warnings.filterwarnings("error", category=SyntaxWarning)

players_csv, teams_csv, reports_csv, players_to_games_csv, teams_to_games_csv = (
    Path(__file__).parents[1]
    / "reasondb"
    / "evaluation"
    / "benchmarks"
    / "files"
    / filename
    for filename in [
        "players.csv",
        "teams.csv",
        "reports.csv",
        "players_to_games.csv",
        "teams_to_games.csv",
    ]
)


with RaccoonDB("rotowire") as rc:
    players = rc.add_table(
        path=players_csv,
        table_name="players",
    )
    teams = rc.add_table(
        path=teams_csv,
        table_name="teams",
    )
    reports = rc.add_table(
        path=reports_csv,
        table_name="reports",
        text_columns=[InPlaceColumn("reports.report")],
    )
    players_to_games = rc.add_table(
        path=players_to_games_csv,
        table_name="players_to_games",
    )
    teams_to_games = rc.add_table(
        path=teams_to_games_csv,
        table_name="teams_to_games",
    )

    # df_query = (
    #     players.join(players_to_games, "Join on {name}")
    #     .join(reports, "Join on {game_id}")
    #     .extract("How many [points] did {name} score according to {report}?")
    #     .extract("How many [assists] did {name} have according to {report}?")
    #     .project("Keep {game_id}, {name}, {points}, and {assists}")
    # )
    df_query = (
        players.join(players_to_games, "Join on {name}")
        .join(reports, "Join on {game_id}")
        .extract("How many [points] did {name} score according to {report}?")
        .filter("Only keep if the home team won the game in {report}")
        .project("Keep {game_id}, {name}, {points}")
    )

    result = df_query.execute(
        "num_points_per_player", RecallGuarantee(0.9), PrecisionGuarantee(0.9)
    )
    result.pprint()
