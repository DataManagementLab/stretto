import logging
from pathlib import Path
from reasondb.database.indentifier import RemoteColumn
from reasondb.evaluation.benchmarks.real_estate import RealEstate
from reasondb.interface.connect import RaccoonDB
from reasondb.optimizer.guarantees import PrecisionGuarantee, RecallGuarantee


logging.basicConfig(level=logging.INFO)

path = Path("palimpzest/testdata/real-estate-eval")
RealEstate.load("train")
listings_csv_path = RealEstate.load_listings_table(path)
pictures_csv_path = RealEstate.load_pictures_table(path)

with RaccoonDB("real_estate") as rc:
    listings = rc.add_table(
        path=listings_csv_path,
        table_name="listings",
        text_columns=[RemoteColumn("listings.text_path", "listings.text")],
    )
    pictures = rc.add_table(
        path=pictures_csv_path,
        table_name="pictures",
        image_columns=[RemoteColumn("pictures.picture_path", "pictures.picture")],
    )

    df_query = listings.extract("Extract the [address] from {text}").project(
        "Keep {id} and {address}"
    )
    result = df_query.execute(
        "addresses", PrecisionGuarantee(0.8), RecallGuarantee(0.8)
    )
    print()
    print(" Results:")
    result.pprint()

    print("***" * 10)
    df_query = listings.join(
        pictures,
        "Join on {id}",
    ).filter("{picture} shows modern interior")
    result = df_query.execute(
        "modern_interior", RecallGuarantee(0.8), PrecisionGuarantee(0.8)
    )
    print()
    print(" Results:")
    result.pprint()

    print("***" * 10)
    df_query = listings.join(
        pictures,
        "Join on {id}",
    ).filter("{picture} shows not very modern interior")
    result = df_query.execute(
        "old_interior", RecallGuarantee(0.8), PrecisionGuarantee(0.8)
    )
    print()
    print(" Results:")
    result.pprint()
