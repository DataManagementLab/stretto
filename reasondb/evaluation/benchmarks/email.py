import os
import logging
import pandas as pd
from pathlib import Path
from typing import Dict, Literal, Sequence, Union
from reasondb.database.database import ExperimentalDatabase
from reasondb.database.indentifier import (
    RemoteColumn,
    VirtualTableIdentifier,
)
from reasondb.evaluation.benchmark import Benchmark, LabelsDefinition, RandomBenchmark
from reasondb.query_plan.logical_plan import (
    LogicalExtract,
    LogicalFilter,
    LogicalPlan,
    LogicalProject,
)
from reasondb.query_plan.query import (
    OperatorOption,
    OperatorPlaceholder,
    Queries,
    Query,
    QueryShape,
    RandomOrder,
)

logger = logging.getLogger(__name__)


EMAIL_QUERIES = Queries(
    Query(
        'What are the senders of E-Mails that refer to a fraudulent scheme (i.e., "Raptor", ...)?',
        _ground_truth_logical_plan=LogicalPlan(
            [
                LogicalFilter(
                    explanation="First, we need to filter by E-Mails that pertain to fraudulent Enron Entity.",
                    inputs=[VirtualTableIdentifier("emails")],
                    output=VirtualTableIdentifier("mention_entity_emails"),
                    expression="{emails.text} refers to a fraudulent scheme (i.e., 'Raptor', ...)",
                    labels=LabelsDefinition(
                        Path("reasondb/evaluation/ground_truth/emails/enron-eval.csv"),
                        "mentions_entity",
                        ["emails"],
                    ),
                ),
                LogicalFilter(
                    explanation="Then, we need to filter by E-Mails that do not quote a news article or outside source.",
                    inputs=[VirtualTableIdentifier("mention_entity_emails")],
                    output=VirtualTableIdentifier("fraudulent_emails"),
                    expression="{mention_entity_emails.text} does not quote a news article or outside source.",
                    labels=LabelsDefinition(
                        Path("reasondb/evaluation/ground_truth/emails/enron-eval.csv"),
                        "fraudulent",
                        ["emails"],
                    ),
                ),
                LogicalExtract(
                    explanation="Then, we need to extract the sender from the E-Mails.",
                    inputs=[VirtualTableIdentifier("fraudulent_emails")],
                    output=VirtualTableIdentifier("fraudulent_emails_sender"),
                    expression="Extract the [sender] from {fraudulent_emails.text}",
                    labels=LabelsDefinition(
                        Path("reasondb/evaluation/ground_truth/emails/enron-eval.csv"),
                        "sender",
                        ["emails"],
                    ),
                ),
                LogicalProject(
                    explanation="Finally, we need to keep distinct senders.",
                    inputs=[VirtualTableIdentifier("fraudulent_emails_sender")],
                    output=VirtualTableIdentifier("senders"),
                    expression="Keep distinct {fraudulent_emails_sender.sender}",
                ),
            ]
        ),
    ),
)


class EnronEmail(Benchmark):
    @staticmethod
    def urls():
        return {}

    @staticmethod
    def get_queries() -> Queries:
        return EMAIL_QUERIES

    @property
    def has_ground_truth(self) -> bool:
        return True

    @staticmethod
    def download(split: Literal["train", "dev", "test"]) -> Benchmark:
        os.makedirs(EnronEmail.dir(), exist_ok=True)
        Benchmark.run_script(
            Path("testdata/download-testdata.sh"), cwd=Path("palimpzest")
        )
        return EnronEmail.load_from_disk(split)

    @staticmethod
    def load_from_disk(split: Literal["train", "dev", "test"]) -> "Benchmark":
        email_table = EnronEmail.load_email_table(
            Path("palimpzest/testdata/enron-eval")
        )
        benchmark = EnronEmail(
            split,
            ExperimentalDatabase.load_from_files(
                db_name=EnronEmail.name(),
                split=split,
                table_names=["emails"],
                paths=[email_table],
                text_columns=[RemoteColumn("emails.text_path", "emails.text")],
            ),
            EnronEmail.get_queries(),
        )
        return benchmark

    @staticmethod
    def load_email_table(path: Path) -> Path:
        data = []
        for id, email_path in enumerate(sorted(path.iterdir())):
            data.append([id, email_path.absolute()])
        columns = ["id", "text_path"]
        df = pd.DataFrame(data, columns=columns)
        path = path / "emails.csv"
        with open(path, "w") as f:
            df.to_csv(f, index=False)
        return path


class EnronEmailRandom(RandomBenchmark):
    @staticmethod
    def urls():
        return {}

    @property
    def has_ground_truth(self) -> bool:
        return False

    @staticmethod
    def download(split: Literal["train", "dev", "test"]) -> Benchmark:
        os.makedirs(EnronEmailRandom.dir(), exist_ok=True)
        Benchmark.run_script(
            Path("testdata/download-testdata.sh"), cwd=Path("palimpzest")
        )
        return EnronEmailRandom.load_from_disk(split)

    @staticmethod
    def load_from_disk(split: Literal["train", "dev", "test"]) -> "Benchmark":
        email_table = EnronEmail.load_email_table(
            Path("palimpzest/testdata/enron-eval")
        )
        benchmark = EnronEmailRandom(
            split,
            ExperimentalDatabase.load_from_files(
                db_name=EnronEmailRandom.name(),
                split=split,
                table_names=["emails"],
                paths=[email_table],
                text_columns=[RemoteColumn("emails.text_path", "emails.text")],
            ),
            EnronEmailRandom.generate_random_queries(split),
        )
        return benchmark

    @classmethod
    def _get_query_shapes(cls) -> Sequence[QueryShape]:
        return EMAIL_QUERY_SHAPES

    @classmethod
    def _get_operator_options(cls) -> Sequence[OperatorOption]:
        return EMAIL_OPERATOR_OPTIONS

    @classmethod
    def _single_filter_shape(cls) -> Union[QueryShape, Dict[str, QueryShape]]:
        return SINGLE_FILTER_SHAPE


SINGLE_FILTER_SHAPE = QueryShape(
    OperatorPlaceholder(
        LogicalFilter,
        inputs=[VirtualTableIdentifier("emails")],
        output=VirtualTableIdentifier("output"),
    ),
)

EMAIL_OPERATOR_OPTIONS = [
    # 10 filter for emails
    OperatorOption(
        LogicalFilter,
        "{emails.text} refers to a fraudulent scheme (i.e., 'Raptor', ...)",
    ),
    OperatorOption(
        LogicalFilter,
        "{emails.text} contains confidential information",
    ),
    OperatorOption(
        LogicalFilter,
        "{emails.text} is written in a formal tone",
    ),
    OperatorOption(
        LogicalFilter,
        "{emails.text} discusses financial matters",
    ),
    OperatorOption(
        LogicalFilter,
        "{emails.text} is addressed to multiple recipients",
    ),
    OperatorOption(
        LogicalFilter,
        "{emails.text} includes an attachment",
    ),
    OperatorOption(
        LogicalFilter,
        "{emails.text} was sent during business hours",
    ),
    OperatorOption(
        LogicalFilter,
        "{emails.text} contains a greeting",
    ),
    OperatorOption(
        LogicalFilter,
        "{emails.text} discusses legal issues",
    ),
    OperatorOption(
        LogicalFilter,
        "{emails.text} references a meeting or event",
    ),
    # 10 extract for emails
    OperatorOption(
        LogicalExtract,
        "Extract the [sender] from {emails.text}",
    ),
    OperatorOption(
        LogicalExtract,
        "Extract the first [recipient] from {emails.text}",
    ),
    OperatorOption(
        LogicalExtract,
        "Extract the [date] from {emails.text}",
    ),
    OperatorOption(
        LogicalExtract,
        "Extract the [subject] from {emails.text}",
    ),
    OperatorOption(
        LogicalExtract,
        "Classify the [urgency_level] from {emails.text} (categories: high, medium, low)",
    ),
    OperatorOption(
        LogicalExtract,
        "Classify the [sentiment] from {emails.text} (categories: positive, negative, neutral)",
    ),
    OperatorOption(
        LogicalExtract,
        "Classify the [topic] from {emails.text} (categories: finance, legal, personal, other)",
    ),
    OperatorOption(
        LogicalExtract,
        "Extract the [number of attachments] from {emails.text}",
    ),
    OperatorOption(
        LogicalExtract,
        "Classify if {emails.text} contains a [request_for_action] (yes/no)",
    ),
    OperatorOption(
        LogicalExtract,
        "Classify the [confidentiality_level] from {emails.text} (categories: public, internal, confidential)",
    ),
]

EMAIL_QUERY_SHAPES = [
    QueryShape(
        RandomOrder(
            OperatorPlaceholder(
                LogicalFilter,
                inputs=[VirtualTableIdentifier("emails")],
                output=VirtualTableIdentifier("intermediate"),
            ),
            OperatorPlaceholder(
                LogicalExtract,
                inputs=[VirtualTableIdentifier("intermediate")],
                output=VirtualTableIdentifier("output"),
            ),
        ),
        additional_info={
            "num_semops": 2,
            "num_sem_filter": 1,
            "num_sem_extract": 1,
        },
    ),
    QueryShape(
        RandomOrder(
            OperatorPlaceholder(
                LogicalFilter,
                inputs=[VirtualTableIdentifier("emails")],
                output=VirtualTableIdentifier("intermediate"),
            ),
            OperatorPlaceholder(
                LogicalFilter,
                inputs=[VirtualTableIdentifier("intermediate")],
                output=VirtualTableIdentifier("output"),
            ),
        ),
        additional_info={
            "num_semops": 2,
            "num_sem_filter": 2,
            "num_sem_extract": 0,
        },
    ),
    QueryShape(
        RandomOrder(
            OperatorPlaceholder(
                LogicalFilter,
                inputs=[VirtualTableIdentifier("emails")],
                output=VirtualTableIdentifier("intermediate1"),
            ),
            OperatorPlaceholder(
                LogicalExtract,
                inputs=[VirtualTableIdentifier("intermediate1")],
                output=VirtualTableIdentifier("intermediate2"),
            ),
            OperatorPlaceholder(
                LogicalExtract,
                inputs=[VirtualTableIdentifier("intermediate2")],
                output=VirtualTableIdentifier("output"),
            ),
        ),
        additional_info={
            "num_semops": 3,
            "num_sem_filter": 1,
            "num_sem_extract": 2,
        },
    ),
    QueryShape(
        RandomOrder(
            OperatorPlaceholder(
                LogicalFilter,
                inputs=[VirtualTableIdentifier("emails")],
                output=VirtualTableIdentifier("intermediate1"),
            ),
            OperatorPlaceholder(
                LogicalFilter,
                inputs=[VirtualTableIdentifier("intermediate1")],
                output=VirtualTableIdentifier("intermediate2"),
            ),
            OperatorPlaceholder(
                LogicalExtract,
                inputs=[VirtualTableIdentifier("intermediate2")],
                output=VirtualTableIdentifier("output"),
            ),
        ),
        additional_info={
            "num_semops": 3,
            "num_sem_filter": 2,
            "num_sem_extract": 1,
        },
    ),
    QueryShape(
        RandomOrder(
            OperatorPlaceholder(
                LogicalFilter,
                inputs=[VirtualTableIdentifier("emails")],
                output=VirtualTableIdentifier("intermediate1"),
            ),
            OperatorPlaceholder(
                LogicalExtract,
                inputs=[VirtualTableIdentifier("intermediate1")],
                output=VirtualTableIdentifier("intermediate2"),
            ),
            OperatorPlaceholder(
                LogicalExtract,
                inputs=[VirtualTableIdentifier("intermediate2")],
                output=VirtualTableIdentifier("intermediate3"),
            ),
            OperatorPlaceholder(
                LogicalExtract,
                inputs=[VirtualTableIdentifier("intermediate3")],
                output=VirtualTableIdentifier("output"),
            ),
        ),
        additional_info={
            "num_semops": 4,
            "num_sem_filter": 1,
            "num_sem_extract": 3,
        },
    ),
    QueryShape(
        RandomOrder(
            OperatorPlaceholder(
                LogicalFilter,
                inputs=[VirtualTableIdentifier("emails")],
                output=VirtualTableIdentifier("intermediate1"),
            ),
            OperatorPlaceholder(
                LogicalFilter,
                inputs=[VirtualTableIdentifier("intermediate1")],
                output=VirtualTableIdentifier("intermediate2"),
            ),
            OperatorPlaceholder(
                LogicalExtract,
                inputs=[VirtualTableIdentifier("intermediate2")],
                output=VirtualTableIdentifier("intermediate3"),
            ),
            OperatorPlaceholder(
                LogicalExtract,
                inputs=[VirtualTableIdentifier("intermediate3")],
                output=VirtualTableIdentifier("output"),
            ),
        ),
        additional_info={
            "num_semops": 4,
            "num_sem_filter": 2,
            "num_sem_extract": 2,
        },
    ),
]
