import os
import re
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
    LogicalJoin,
    LogicalPlan,
    LogicalProject,
    LogicalRename,
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


#: The annotations *and* the row order of the emails table, in one file.
#:
#: `LabelsDefinition` joins ground truth on `_index_<table>`, the loaded table's rowid,
#: so the order of the emails is the labels' join key. `load_email_table` therefore reads
#: that order from this file rather than from a directory listing.
EMAIL_LABELS = Path("reasondb/evaluation/ground_truth/emails/enron-eval.csv")


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
                        EMAIL_LABELS,
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
                        EMAIL_LABELS,
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
                        EMAIL_LABELS,
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

    @classmethod
    def download(cls, split: Literal["train", "dev", "test"]) -> Benchmark:
        os.makedirs(cls.dir(), exist_ok=True)
        Benchmark.run_script(
            Path("testdata/download-testdata.sh"), cwd=Path("palimpzest")
        )
        return cls.load_from_disk(split)

    @classmethod
    def load_from_disk(cls, split: Literal["train", "dev", "test"]) -> "Benchmark":
        # Via `cls` throughout so a subclass overriding only `name()`/`get_queries()`
        # inherits this wiring -- see reasondb/evaluation/benchmarks/curated.py.
        email_table = cls.load_email_table(Path("palimpzest/testdata/enron-eval"))
        benchmark = cls(
            split,
            ExperimentalDatabase.load_from_files(
                db_name=cls.name(),
                split=split,
                table_names=["emails"],
                paths=[email_table],
                text_columns=[RemoteColumn("emails.text_path", "emails.text")],
            ),
            cls.get_queries(),
        )
        return benchmark

    @staticmethod
    def load_email_table(path: Path) -> Path:
        """Write the emails table, one row per email, in `EMAIL_LABELS`' order.

        The row ordinal is the labels' join key (see `EMAIL_LABELS`), so it is read from
        the file that holds the labels rather than reconstructed by listing the directory.

        Mismatches between the label file and the corpus on disk raise rather than
        truncate or reorder, since either would score emails against the wrong labels.
        """
        manifest = pd.read_csv(EMAIL_LABELS).sort_values("_index_emails")
        ordinals = manifest["_index_emails"].tolist()
        if ordinals != list(range(len(manifest))):
            raise ValueError(
                f"{EMAIL_LABELS} must key rows 0..{len(manifest) - 1} contiguously -- "
                "the table's rowid is what a label joins on, so a gap or a duplicate "
                "silently shifts every row after it."
            )

        names = manifest["filename"].tolist()
        on_disk = {p.name for p in path.glob("*.txt")}
        missing = [n for n in names if n not in on_disk]
        extra = sorted(on_disk - set(names))
        if missing or extra:
            raise ValueError(
                f"{path} does not hold exactly the {len(names)} emails "
                f"{EMAIL_LABELS} annotates: {len(missing)} missing "
                f"(e.g. {missing[:3]}), {len(extra)} unannotated (e.g. {extra[:3]})."
            )

        df = pd.DataFrame(
            [[i, (path / name).absolute()] for i, name in enumerate(names)],
            columns=["id", "text_path"],
        )
        path = path / "emails.csv"
        with open(path, "w") as f:
            df.to_csv(f, index=False)
        return path


class EnronEmailRandom(RandomBenchmark):
    @classmethod
    def name(cls) -> str:
        return "email_random"

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

    @classmethod
    def _load_database(cls, split: Literal["train", "dev", "test"]):
        email_table = EnronEmail.load_email_table(
            Path("palimpzest/testdata/enron-eval")
        )
        return ExperimentalDatabase.load_from_files(
            db_name=cls.name(),
            split=split,
            table_names=["emails"],
            paths=[email_table],
            text_columns=[RemoteColumn("emails.text_path", "emails.text")],
        )

    @classmethod
    def _get_query_shapes(cls) -> Sequence[QueryShape]:
        return EMAIL_QUERY_SHAPES

    @classmethod
    def _get_operator_options(cls) -> Sequence[OperatorOption]:
        return EMAIL_OPERATOR_OPTIONS

    @classmethod
    def _single_filter_shape(cls) -> Union[QueryShape, Dict[str, QueryShape]]:
        return SINGLE_FILTER_SHAPE

    @classmethod
    def get_join_queries(cls) -> "Queries":
        return EMAIL_JOIN_QUERIES


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
    # 10 join for emails (self-join: left uses {:emails:.text}, right uses {emails_other.text_other})
    OperatorOption(
        LogicalJoin,
        "{:emails:.text} and {emails_other.text_other} were sent on the exact same day",
    ),
    OperatorOption(
        LogicalJoin,
        "{:emails:.text} and {emails_other.text_other} mention the same person in the summary",
    ),
    OperatorOption(
        LogicalJoin,
        "{:emails:.text} and {emails_other.text_other} discuss the same specific project",
    ),
    OperatorOption(
        LogicalJoin,
        "{:emails:.text} and {emails_other.text_other} were sent by the same email address",
    ),
    OperatorOption(
        LogicalJoin,
        "{:emails:.text} and {emails_other.text_other} share the exact same subject line",
    ),
    OperatorOption(
        LogicalJoin,
        "{:emails:.text} and {emails_other.text_other} both contain a standard legal confidentiality notice",
    ),
    OperatorOption(
        LogicalJoin,
        "{:emails:.text} and {emails_other.text_other} were archived from the same organizational folder",
    ),
    OperatorOption(
        LogicalJoin,
        "{:emails:.text} and {emails_other.text_other} represent a back-and-forth conversation (where the sender of one is the recipient of the other)",
    ),
    OperatorOption(
        LogicalJoin,
        "{:emails:.text} and {emails_other.text_other} were both sent to the same primary recipient",
    ),
    OperatorOption(
        LogicalJoin,
        "{:emails:.text} and {emails_other.text_other} were sent within one hour of each other",
    ),
]


def _make_join_query(expression: str) -> "Query":
    expression = re.sub(
        r"{\:([a-z_][a-z0-9_]*)\:\.([a-z_][a-z0-9_]*)}",
        r"{\1.\2}",
        expression,
    )
    return Query(
        expression,
        _ground_truth_logical_plan=LogicalPlan(
            [
                LogicalRename(
                    explanation="Rename emails table for self-join.",
                    inputs=[VirtualTableIdentifier("emails")],
                    output=VirtualTableIdentifier("emails_other"),
                    expression="Rename {emails.text} to [text_other]",
                    labels=None,
                ),
                LogicalJoin(
                    explanation=expression,
                    inputs=[
                        VirtualTableIdentifier("emails"),
                        VirtualTableIdentifier("emails_other"),
                    ],
                    output=VirtualTableIdentifier("output"),
                    expression=expression,
                    labels=None,
                ),
            ]
        ),
    )


EMAIL_JOIN_QUERIES = Queries(
    *[
        _make_join_query(opt.expression)
        for opt in EMAIL_OPERATOR_OPTIONS
        if opt.operator_type == LogicalJoin
    ]
)

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
