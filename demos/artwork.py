import logging
from pathlib import Path
from reasondb.database.indentifier import RemoteColumn
from reasondb.interface.connect import RaccoonDB
from reasondb.optimizer.guarantees import PrecisionGuarantee, RecallGuarantee

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

csv = orig_csv = (
    Path(__file__).parents[1]
    / "reasondb"
    / "evaluation"
    / "benchmarks"
    / "files"
    / "paintings_sampled_no_duplicated.csv"
)

with RaccoonDB("artworks_sampled") as rc:
    artworks = rc.add_table(
        path=csv,
        table_name="artworks",
        image_columns=[RemoteColumn("images.image_url", "images.image", url=True)],
    )

    nl_query = rc.nl_query(
        "What is number of paintings that depict Madonna and Child for each century?",
    )

    result = nl_query.execute(
        "madonna_and_child_count_by_century",
        PrecisionGuarantee(0.7),
        RecallGuarantee(0.7),
    )
    result.pprint()

    #  result = nl_query.execute("madonna_and_child_count_by_century")
    #  result.pprint()
    #
    # print("*" * 100)
    # print(
    #     "(The below is expected to fail until the labgroup implemented the functionality)"
    # )
    # print()
    #
    # df_query = (
    #     artworks.filter("{image} depict Madonna and Child")
    #     .extract("Extract the [century] from {inception}")
    #     .groupby("Group by {century}")
    #     .aggregate("Count the number of paintings [count]")
    # )

    # result = df_query.execute(
    #     "madonna_and_child_count_by_century_2",
    #     PrecisionGuarantee(0.9),
    #     RecallGuarantee(0.9),
    # )
    # result.pprint()
    # result = df_query.execute("madonna_and_child_count_by_century_2")
    # result.pprint()
    #
    # result = df_query.execute("madonna_and_child_count_by_century_2")
    # result.pprint()

    # evaluator = MetricsManager(
    # predictions_path="reasondb/evaluation_manager/tmp_output.csv",
    # ground_truth_path="reasondb/evaluation_manager/ground_truth/artwork.csv",
    # attribute_eval="m&c",
    # )

    # results = evaluator.evaluate_all()
    # evaluator._delete_predictions_file()
    # print(results)
