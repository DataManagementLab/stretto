import os
import pandas as pd
import logging
from pathlib import Path
from typing import Literal
from reasondb.database.database import ExperimentalDatabase
from reasondb.database.indentifier import (
    RemoteColumn,
    VirtualTableIdentifier,
)
from reasondb.evaluation.benchmark import Benchmark
from reasondb.query_plan.logical_plan import (
    LogicalExtract,
    LogicalFilter,
    LogicalJoin,
    LogicalPlan,
    LogicalProject,
)
from reasondb.query_plan.query import Queries, Query

logger = logging.getLogger(__name__)


REAL_ESTATE_QUERIES = Queries(
    Query(
        "Which listings have a modern interior?",
        _ground_truth_logical_plan=LogicalPlan(
            [
                LogicalJoin(
                    explanation="First, we need to join the listings and pictures tables.",
                    inputs=[
                        VirtualTableIdentifier("listings"),
                        VirtualTableIdentifier("pictures"),
                    ],
                    output=VirtualTableIdentifier("joined_table"),
                    expression="{listings.id} equals {pictures.id}",
                ),
                LogicalFilter(
                    explanation="Then, we need to filter by listings that have a modern interior.",
                    inputs=[VirtualTableIdentifier("joined_table")],
                    output=VirtualTableIdentifier("filtered_table"),
                    expression="{joined_table.picture} shows a modern interior",
                ),
                LogicalProject(
                    explanation="Finally, we need to keep distinct listings.",
                    inputs=[VirtualTableIdentifier("filtered_table")],
                    output=VirtualTableIdentifier("result_table"),
                    expression="Keep distinct {filtered_table.id}, {filtered_table.text}",
                ),
            ]
        ),
    ),
    Query(
        "What is the address of each listing?",
        _ground_truth_logical_plan=LogicalPlan(
            [
                LogicalExtract(
                    explanation="First, we need to extract the address from the listings.",
                    inputs=[VirtualTableIdentifier("listings")],
                    output=VirtualTableIdentifier("listings"),
                    expression="Extract the [address] from {listings.text}",
                ),
                LogicalProject(
                    explanation="Finally, we need to keep the id and address of the listings.",
                    inputs=[VirtualTableIdentifier("listings")],
                    output=VirtualTableIdentifier("listings"),
                    expression="Keep {listings.id}, {listings.address}",
                ),
            ]
        ),
    ),
)


class RealEstate(Benchmark):
    @staticmethod
    def urls():
        return {}

    @property
    def has_ground_truth(self) -> bool:
        return False

    @staticmethod
    def download(split: Literal["train", "dev", "test"]) -> Benchmark:
        os.makedirs(RealEstate.dir(), exist_ok=True)
        Benchmark.run_script(
            Path("testdata/download-testdata.sh"), cwd=Path("palimpzest")
        )
        return RealEstate.load_from_disk(split)

    @staticmethod
    def load_from_disk(split: Literal["train", "dev", "test"]) -> "Benchmark":
        listings_table = RealEstate.load_listings_table(
            Path("palimpzest/testdata/real-estate-eval")
        )
        pictures_table = RealEstate.load_pictures_table(
            Path("palimpzest/testdata/real-estate-eval")
        )
        benchmark = RealEstate(
            split,
            ExperimentalDatabase.load_from_files(
                db_name=RealEstate.name(),
                split=split,
                table_names=["listings", "pictures"],
                paths=[listings_table, pictures_table],
                text_columns=[RemoteColumn("listings.text_path", "listings.text")],
                image_columns=[
                    RemoteColumn("pictures.picture_path", "pictures.picture")
                ],
            ),
            REAL_ESTATE_QUERIES,
        )
        return benchmark

    @staticmethod
    def load_listings_table(path: Path) -> Path:
        data = []
        for listing_dir in path.iterdir():
            if not listing_dir.is_dir():
                continue
            id = int(listing_dir.name.split("listing")[1])
            listing_path = listing_dir / "listing-text.txt"
            data.append([id, listing_path])
        columns = ["id", "text_path"]
        df = pd.DataFrame(data, columns=columns)
        path = path / "listings.csv"
        with open(path, "w") as f:
            df.to_csv(f, index=False)
        return path

    @staticmethod
    def load_pictures_table(path: Path) -> Path:
        data = []
        for listing_dir in path.iterdir():
            if not listing_dir.is_dir():
                continue
            listing_id = int(listing_dir.name.split("listing")[1])
            for image_path in listing_dir.glob("*.png"):
                data.append([listing_id, image_path])
        columns = ["id", "picture_path"]
        df = pd.DataFrame(data, columns=columns)
        path = path / "pictures.csv"
        with open(path, "w") as f:
            df.to_csv(f, index=False)
        return path
