import logging
from pathlib import Path
from reasondb.database.indentifier import RemoteColumn
from reasondb.interface.connect import RaccoonDB
from reasondb.optimizer.guarantees import PrecisionGuarantee, RecallGuarantee

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# data_path = Path("SemBench/files/animals")
# if not data_path.exists():
#     data_path.mkdir(parents=True, exist_ok=True)
#     gdown.download(
#         "https://drive.google.com/uc?id=1HG6tvXIA0BtpqbZqCR46oNSeY2yZ4Lko",
#         output="SemBench/files/medical/animals.zip",
#     )
#     with zipfile.ZipFile("SemBench/files/medical/animals.zip", "r") as zip_ref:
#         zip_ref.extractall("SemBench/files/animals/")

# csv_image = orig_csv = (
#    Path(__file__).parents[1]
#    / "reasondb"
#    / "evaluation"
#    / "benchmarks"
#    / "files"
#    / "animals_500.csv"
# )

csv_audio = (
    Path(__file__).parents[1]
    / "reasondb"
    / "evaluation"
    / "benchmarks"
    / "files"
    / "animals_audio.csv"
)

with RaccoonDB("animals_sampled") as rc:
    # animals = rc.add_table(
    #    path=csv_image,
    #    table_name="animals",
    #    image_columns=[RemoteColumn("animals.ImagePath", "animals.picture")],
    # )

    animals_audio = rc.add_table(
        path=csv_audio,
        table_name="animals_audio",
        audio_columns=[RemoteColumn("animals_audio.AudioPath", "animals_audio.audio")],
    )

    # Q1

    #    nl_query = rc.nl_query(
    #        "How many pictures of zebras do we have in our database?",
    #    )

    # df_query = (
    #    animals.filter("{picture} depicts ZEBRA")
    #    .aggregate("Count the rows as [zebras_count]")
    # )
    #
    # result = df_query.execute(
    #    "zebras_count",
    #    PrecisionGuarantee(0.8),
    #    RecallGuarantee(0.8),
    # )
    # result.pprint()

    # Q2
    df_query = animals_audio.filter("{audio} contains sound of a lion").aggregate(
        "Count the rows as [lion_sounds_count]"
    )
    # nl_query = rc.nl_query(
    #     "How many audio clips contain the sound of an elephant?",
    # )
    result = df_query.execute(
        "lion_sounds_count",
        PrecisionGuarantee(0.8),
        RecallGuarantee(0.8),
    )
    result.pprint()

    # Q3
    #    df_query = (
    #        animals.filter("{picture} depicts ZEBRA")
    #        .project("Keep {city}")
    #    )

#    nl_query = rc.nl_query(
#        "What is the city where we captured most pictures of zebras?",
#        # "What is the number of images with a zebra for each city?",
#    )
#    #
#    result = nl_query.execute(
#        "count_zebras_per_city",
#        PrecisionGuarantee(0.8),
#        RecallGuarantee(0.8),
#    )
#    result.pprint()

# Q10

#    df_query = (
#        animals.filter("{picture} depicts ZEBRA")
#        .project("Keep {city}, {station}")
#    )
#
#    nl_query = rc.nl_query(
#        #"What is the city and station with most associated pictures showing zebras?"
#        "What is the number of images with a zebra for each city and station?",
#    )
#
#    result = nl_query.execute(
#        "city_station_with_most_zebras",
#        PrecisionGuarantee(0.8),
#        RecallGuarantee(0.8),
#    )
#    result.pprint()
#
