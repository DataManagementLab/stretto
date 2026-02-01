import logging
from reasondb.database.indentifier import InPlaceColumn
from reasondb.evaluation.benchmarks.biodex import Biodex
from reasondb.interface.connect import RaccoonDB


logger = logging.getLogger(__name__)

Biodex.load("train")
reports_csv = Biodex.load_patient_reports_table("train")
reactions_csv = Biodex.load_reactions_table("train")


logging.basicConfig(level=logging.INFO)


with RaccoonDB("biodex") as rc:
    reports_table = rc.add_table(
        reports_csv,
        "patient_reports",
        text_columns=[
            InPlaceColumn("patient_reports.fulltext"),
            InPlaceColumn("patient_reports.abstract"),
        ],
    )
    reactions_table = rc.add_table(
        reactions_csv,
        "reactions",
    )
    reports_table.pprint()
    reactions_table.pprint()

    join_result = reports_table.join(
        reactions_table,
        "Patient described in Abstract {left.abstract} and full text {left.fulltext} experienced medical condition {right.reaction}.",
    ).execute(name="join_result")

    join_result.pprint()
