import os
import datasets
import logging
import pandas as pd
from pathlib import Path
from typing import Literal
from reasondb.database.database import ExperimentalDatabase
from reasondb.database.indentifier import (
    InPlaceColumn,
    VirtualTableIdentifier,
)
from reasondb.evaluation.benchmark import Benchmark
from reasondb.optimizer.guarantees import PrecisionGuarantee, RecallGuarantee
from reasondb.query_plan.logical_plan import (
    LogicalJoin,
    LogicalPlan,
)
from reasondb.query_plan.query import Queries, Query

logger = logging.getLogger(__name__)
SAMPLE_SIZE = 100


BIODEX_QUERIES = Queries(
    Query(
        "Which reactions from the reactions table are experimenced by each patient as discussed in each patient report",
        PrecisionGuarantee(0.8),
        RecallGuarantee(0.8),
        _ground_truth_logical_plan=LogicalPlan(
            [
                LogicalJoin(
                    explanation="We need to match reactions each patient has experienced as discussed in each patient report with the reactions in the reactions table.",
                    inputs=[
                        VirtualTableIdentifier("patient_reports"),
                        VirtualTableIdentifier("reactions"),
                    ],
                    output=VirtualTableIdentifier("matched_reactions"),
                    expression="The patient described in the patient report with abstract {patient_reports.abstract} and fulltext {patient_reports.fulltext} has experienced the reaction {reactions.reaction}.",
                ),
            ]
        ),
    ),
)


class Biodex(Benchmark):
    @staticmethod
    def urls():
        return {}

    @staticmethod
    def get_queries() -> Queries:
        return BIODEX_QUERIES

    @staticmethod
    def download(split: Literal["train", "dev", "test"]) -> Benchmark:
        os.makedirs(Biodex.dir(), exist_ok=True)
        Benchmark.run_script(
            Path("testdata/download-testdata.sh"), cwd=Path("palimpzest")
        )
        report_df = (
            datasets.load_dataset("BioDEX/BioDEX-Reactions", split=split)
            .to_pandas()  # type: ignore
            .iloc[:SAMPLE_SIZE]  # type: ignore
        )
        assert isinstance(report_df, pd.DataFrame)
        report_df[["pmid", "title", "abstract", "fulltext"]].to_csv(
            Biodex.dir() / f"{split}_reports.csv", index=False
        )
        with open("palimpzest/testdata/reaction_terms.txt") as f:
            reactions = [line.strip() for line in f.readlines()]
            reactions_df = pd.DataFrame(reactions, columns=["reaction"])
            reactions_df.to_csv(Biodex.dir() / "reactions.csv", index=False)
        return Biodex.load_from_disk(split)

    @staticmethod
    def _compute_pmid_to_label(dataset: list[dict]) -> dict:
        """Compute the label for a BioDEX report given its entry in the dataset."""
        pmid_to_label = {}
        for entry in dataset:
            pmid = str(entry["pmid"])
            reactions_lst = [
                reaction.strip().lower().replace("'", "").replace("^", "")
                for reaction in entry["reactions"].split(",")
            ]
            pmid_to_label[pmid] = reactions_lst

        return pmid_to_label

    @staticmethod
    def load_from_disk(split: Literal["train", "dev", "test"]) -> "Benchmark":
        patient_report_table = Biodex.load_patient_reports_table(split)
        reactions_table = Biodex.load_reactions_table(split)
        benchmark = Biodex(
            split,
            ExperimentalDatabase.load_from_files(
                db_name=Biodex.name(),
                split=split,
                table_names=["patient_reports", "reactions"],
                paths=[patient_report_table, reactions_table],
                text_columns=[
                    InPlaceColumn("patient_reports.abstract"),
                    InPlaceColumn("patient_reports.fulltext"),
                ],
            ),
            Biodex.get_queries(),
        )
        return benchmark

    @staticmethod
    def load_patient_reports_table(split: Literal["train", "dev", "test"]) -> Path:
        return Biodex.dir() / f"{split}_reports.csv"

    @staticmethod
    def load_reactions_table(split: Literal["train", "dev", "test"]) -> Path:
        return Biodex.dir() / "reactions.csv"
