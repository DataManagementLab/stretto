import os
import logging
import json
import re
from pathlib import Path
from typing import Literal
from reasondb.database.database import DataType, Database, DataTable
from reasondb.database.indentifier import VirtualTableIdentifier
from reasondb.evaluation.benchmark import URL, Benchmark
from reasondb.query_plan.logical_plan import LogicalJoin, LogicalPlan, LogicalProject
from reasondb.query_plan.query import Queries, Query
from reasondb.utils.logging import FileLogger

logger = logging.getLogger(__name__)


FEVER_QUERIES = Queries(
    Query(
        "Which claims are refuted by the evidence?",
        _ground_truth_logical_plan=LogicalPlan(
            [
                LogicalJoin(
                    explanation="First, we need to join the claims and evidence tables.",
                    inputs=[
                        VirtualTableIdentifier("claims"),
                        VirtualTableIdentifier("evidence"),
                    ],
                    output=VirtualTableIdentifier("refuted_claims"),
                    expression="{evidence.text} refutes {claims.claim}",
                ),
                LogicalProject(
                    explanation="Finally, we need to keep the refuted claims.",
                    inputs=[VirtualTableIdentifier("refuted_claims")],
                    output=VirtualTableIdentifier("refuted_claims"),
                    expression="Keep {refuted_claims.claim}",
                ),
            ]
        ),
    ),
    Query(
        "Which claims are supported by the evidence?",
        _ground_truth_logical_plan=LogicalPlan(
            [
                LogicalJoin(
                    explanation="First, we need to join the claims and evidence tables.",
                    inputs=[
                        VirtualTableIdentifier("claims"),
                        VirtualTableIdentifier("evidence"),
                    ],
                    output=VirtualTableIdentifier("supported_claims"),
                    expression="{evidence.text} supports {claims.claim}",
                ),
                LogicalProject(
                    explanation="Finally, we need to keep the supported claims.",
                    inputs=[VirtualTableIdentifier("supported_claims")],
                    output=VirtualTableIdentifier("supported_claims"),
                    expression="Keep {supported_claims.claim}",
                ),
            ]
        ),
    ),
)


class Fever(Benchmark):
    @staticmethod
    def urls():
        return {
            "train": URL(
                "https://fever.ai/download/fever/train.jsonl",
                "eba7e8f87076753f8494718b9a857827af7bf73e76c9e4b75420207d26e588b6",
                Fever.dir() / "train.jsonl",
            ),
            "dev": URL(
                "https://fever.ai/download/fever/shared_task_dev.jsonl",
                "e89865bfe1b4dd054e03dd57d7241a6fde24862905f31117cf0cd719f7c78df7",
                Fever.dir() / "dev.jsonl",
            ),
            "test": URL(
                "https://fever.ai/download/fever/shared_task_test.jsonl",
                "76dd0872d8fa1f49efe1194fe8a88b7dd4c715c77d87a142b615d4be583e1e51",
                Fever.dir() / "test.jsonl",
            ),
            "wiki": URL(
                "https://fever.ai/download/fever/wiki-pages.zip",
                "4b06d95da6adf7fe02d2796176c670dacccb21348da89cba4c50676ab99665f2",
                Fever.dir() / "wiki-pages.zip",
            ),
        }

    @staticmethod
    def download(split: Literal["train", "dev", "test"]) -> Benchmark:
        url = Fever.urls()[split]
        os.makedirs(Fever.dir(), exist_ok=True)
        Benchmark.download_file(url)
        Benchmark.download_file(Fever.urls()["wiki"])
        claims_table = Fever.load_claims_table(url.path)
        evidence_table = Fever.load_evidence_table(Fever.dir() / "wiki-pages.zip")
        benchmark = Fever(
            split,
            Database.load_from_data_tables(
                name=Fever.name(),
                split=split,
                tables=[claims_table, evidence_table],
                logger=FileLogger() / "download" / "fever",
            ),
            FEVER_QUERIES,
        )
        return benchmark

    @staticmethod
    def load_claims_table(path: Path) -> DataTable:
        with open(
            path,
        ) as f:
            data = [json.loads(line) for line in f]
        columns = ["id", "claim"]
        return DataTable(
            "claims", columns, data, data_types=[DataType.STRING, DataType.TEXT]
        )

    @staticmethod
    def load_evidence_table(path: Path) -> DataTable:
        data = []
        for file_name, file in Benchmark.load_zipped(path):
            match = re.match(r"wiki-pages/wiki-([0-9]+).jsonl", file_name)
            if match:
                for i, line in enumerate(file):
                    json_obj = json.loads(line)
                    data.append(
                        [f"{match.group(1)}-{i}", json_obj["id"], json_obj["text"]]
                    )
        return DataTable(
            "evidence",
            ["id", "title", "text"],
            data,
            data_types=[DataType.STRING, DataType.STRING, DataType.TEXT],
        )
