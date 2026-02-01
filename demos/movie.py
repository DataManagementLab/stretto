import logging
from pathlib import Path
from reasondb.database.indentifier import InPlaceColumn
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
    / "reviews_1000.csv"
)

with RaccoonDB("movie_sampled") as rc:
    reviews = rc.add_table(
        path=csv,
        table_name="reviews",
        text_columns=[
            InPlaceColumn("reviews.reviewtext"),
        ],
    )

    # Q1

    #    nl_query = rc.nl_query(
    #       "Which are five clearly positive reviews?",
    #    )

    # df_query = (
    #     reviews.filter("{reviewtext} is clearly positive")
    #     .project("Keep {reviewid}")
    #     .limit("Keep 5 rows")
    # )
    # result = df_query.execute(
    #     "positive_reviews",
    #     PrecisionGuarantee(0.8),
    #     RecallGuarantee(0.8),
    # )
    # result.pprint()

    # Q2

    #    nl_query = rc.nl_query(
    #       "Which are five clearly positive reviews for the movie 'taken_3'?",
    #    )

    df_query = (
        reviews.filter("{id} is taken_3")
        .filter("{reviewtext} is clearly positive")
        .project("Keep {reviewid}")
        .limit("Keep 5 rows")
    )
    result = df_query.execute(
        "positive_reviews_for_taken_3",
        PrecisionGuarantee(0.8),
        RecallGuarantee(0.8),
    )
    result.pprint()

    # Join query
    # df_query = reviews.join(
    #     reviews.rename("Rename {reviewtext} to [reviewtext_other]"),
    #     "{left.reviewtext} and {right.reviewtext_other} share the same sentiment",
    # )
    # result = df_query.execute(
    #     "reviews_with_same_sentiment",
    #     PrecisionGuarantee(0.8),
    #     RecallGuarantee(0.8),
    # )
    # result.pprint()

# Q3

#    nl_query = rc.nl_query(
#       "How many are the positive reviews for the movie 'taken_3'?",
#    )

# df_query = (
#     reviews.filter("{id} is taken_3")
#     .filter("{reviewtext} is clearly positive")
#     .aggregate("Count reviews as [positive_reviews_for_taken_3_count]")
# )
# result = df_query.execute(
#     "positive_reviews_for_taken_3_count",
#     PrecisionGuarantee(0.8),
#     RecallGuarantee(0.8),
# )
# result.pprint()

# Q4

#    nl_query = rc.nl_query(
#       "Which is the positivity ratio of reviews for the movie 'taken_3' (positive_reviews / total_reviews)?",
#    )

#    df_query = (
#        reviews.filter("{id} is taken_3")
#        .....
#    )

#    result = nl_query.execute(
#        "positive_reviews_for_taken_3_count",
#        PrecisionGuarantee(0.8),
#        RecallGuarantee(0.8),
#    )
#    result.pprint()
