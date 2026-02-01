import os
import logging
from pathlib import Path
from typing import Literal
from reasondb.database.database import ExperimentalDatabase
from reasondb.database.indentifier import (
    RemoteColumn,
    VirtualTableIdentifier,
)
from reasondb.evaluation.benchmark import Benchmark, LabelsDefinition
from reasondb.optimizer.guarantees import PrecisionGuarantee, RecallGuarantee
from reasondb.query_plan.logical_plan import (
    LogicalFilter,
    LogicalPlan,
    LogicalExtract,
)
from reasondb.query_plan.query import Queries, Query

logger = logging.getLogger(__name__)


ANIMALS_QUERIES = Queries(
    Query(
        "How many pictures of zebras do we have in our database?",
        _ground_truth_logical_plan=LogicalPlan(
            [
                LogicalFilter(
                    explanation="First, we need to filter the images that depict zebras.",
                    inputs=[VirtualTableIdentifier("animals")],
                    output=VirtualTableIdentifier("zebras"),
                    expression="{animals.image} depicts zebras",
                    labels=LabelsDefinition(
                        Path(
                            "reasondb/evaluation/benchmarks/files/animals_500.csv" # Demo atm
                        ),
                        "species",
                        ["animals"],
                    ),
                ),
                LogicalExtract(
                    explanation="We also need to count the number of images.",
                    inputs=[VirtualTableIdentifier("zebras")],
                    output=VirtualTableIdentifier("zebras_count"),
                    expression="COUNT({zebras.image})",
                    labels=LabelsDefinition(
                        Path(
                            "reasondb/evaluation/ground_truth/animals/Q1_500.csv"
                        ),
                        "count_star()",
                        ["animals"],
                    ),
                ),
            ]
        ),
    ),
    Query(
        "What is the city where we captured most pictures of zebras?",
        #"What are the images that depict zebras and in which city are they located?",
        _ground_truth_logical_plan=LogicalPlan(
            [
                LogicalFilter(
                    explanation="First, we need to filter the images that depict zebras.",
                    inputs=[VirtualTableIdentifier("animals")],
                    output=VirtualTableIdentifier("zebras"),
                    expression="{animals.image} depicts zebras",
                    labels=LabelsDefinition(
                        Path(
                            "reasondb/evaluation/benchmarks/files/animals_500.csv"
                        ),
                        "species",
                        ["animals"],
                    ),
                ),
                LogicalExtract(
                    explanation="We also need to extract the city from the image metadata.",
                    inputs=[VirtualTableIdentifier("zebras")],
                    output=VirtualTableIdentifier("zebras_with_city"),
                    expression="Extract the [city] from {zebras.metadata}",
                    labels=LabelsDefinition(
                        Path(
                            "reasondb/evaluation/benchmarks/files/animals_500.csv"
                        ),
                        "city",
                        ["animals"],
                    ),
                ),
            ]
        ),
    ),
    Query(
        "What is the city and station with most associated pictures showing zebras?"
        #"What are the images that depict zebras and in which city and station are they located?",
        _ground_truth_logical_plan=LogicalPlan(
            [
                LogicalFilter(
                    explanation="First, we need to filter the images that depict zebras.",
                    inputs=[VirtualTableIdentifier("animals")],
                    output=VirtualTableIdentifier("zebras"),
                    expression="{animals.image} depicts zebras",
                    labels=LabelsDefinition(
                        Path(
                            "reasondb/evaluation/benchmarks/files/animals_500.csv"
                        ),
                        "species",
                        ["animals"],
                    ),
                ),
                LogicalExtract(
                    explanation="We also need to extract the city and station from the image metadata.",
                    inputs=[VirtualTableIdentifier("zebras")],
                    output=VirtualTableIdentifier("zebras_with_city"),
                    expression="Extract the [city] and [station] from {zebras.metadata}",
                    labels=LabelsDefinition(
                        Path(
                            "reasondb/evaluation/benchmarks/files/animals_500.csv"
                        ),
                        ["city",
                        "stationid"],
                        ["animals"],
                    ),
                ),
            ]
        ),
    ),
)


class Animals(Benchmark):
    @classmethod
    def name(cls) -> str:
        return "animals"

    @property
    def has_ground_truth(self) -> bool:
        return True

    @staticmethod
    def get_csv():
        orig_csv = (
            Path(__file__).parent / "files" / "animals_500.csv"
        )
        return orig_csv

    @staticmethod
    def urls():
        return {}

    @staticmethod
    def download(split: Literal["train", "dev", "test"]) -> Benchmark:
        assert split == "dev"
        os.makedirs(Animals.dir(), exist_ok=True)
        return Animals.load_from_disk(split)

    @staticmethod
    def load_from_disk(split: Literal["train", "dev", "test"]) -> "Benchmark":
        assert split == "dev"
        os.makedirs(Animals.dir(), exist_ok=True)
        benchmark = Animals(
            split,
            ExperimentalDatabase.load_from_files(
                db_name=Animals.name(),
                split=split,
                table_names=["animals"],
                paths=[Animals.get_csv()],
                image_columns=[
                    RemoteColumn("animals.ImagePath", "animals.picture")
                ],
            ),
            ANIMALS_QUERIES,
        )
        return benchmark