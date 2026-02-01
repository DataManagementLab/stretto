import os
import logging
from pathlib import Path
from typing import Dict, Literal, Sequence, Union
from reasondb.database.database import ExperimentalDatabase
from reasondb.database.indentifier import (
    InPlaceColumn,
    RemoteColumn,
    VirtualTableIdentifier,
)
from reasondb.evaluation.benchmark import Benchmark, RandomBenchmark
from reasondb.query_plan.logical_plan import (
    LogicalFilter,
    LogicalPlan,
    LogicalExtract,
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


ECOMMERCE_QUERIES = Queries(
    Query(
        "Which images show items with football/soccer logos?",
        _ground_truth_logical_plan=LogicalPlan(
            [
                LogicalFilter(
                    explanation="Filter products with football/soccer logos.",
                    inputs=[VirtualTableIdentifier("products")],
                    output=VirtualTableIdentifier("football_products"),
                    expression="{products.product_image} shows a football/soccer logos",
                ),
            ]
        ),
    ),
)


class EcommerceLarge(Benchmark):
    @classmethod
    def name(cls) -> str:
        return "ecommerce_large"

    @property
    def has_ground_truth(self) -> bool:
        return False

    @staticmethod
    def get_csv():
        orig_csv = (
            Path(__file__).parent / "files" / "ecommerce_products_large.csv"
        )  # update
        return orig_csv

    @staticmethod
    def urls():
        return {}

    @staticmethod
    def download(split: Literal["train", "dev", "test"]) -> Benchmark:
        assert split == "dev"
        os.makedirs(EcommerceLarge.dir(), exist_ok=True)
        return EcommerceLarge.load_from_disk(split)

    @staticmethod
    def load_from_disk(split: Literal["train", "dev", "test"]) -> "Benchmark":
        assert split == "dev"
        os.makedirs(EcommerceLarge.dir(), exist_ok=True)
        benchmark = EcommerceLarge(
            split,
            ExperimentalDatabase.load_from_files(
                db_name=EcommerceLarge.name(),
                split=split,
                table_names=["products"],
                paths=[EcommerceLarge.get_csv()],
                text_columns=[
                    InPlaceColumn("products.description"),
                ],
                image_columns=[
                    RemoteColumn(
                        "products.product_image",
                        "products.product_image",
                    )
                ],
            ),
            ECOMMERCE_QUERIES,
        )
        return benchmark


class EcommerceRandom(RandomBenchmark):
    @classmethod
    def name(cls) -> str:
        return "ecommerce_random"

    @property
    def has_ground_truth(self) -> bool:
        return False

    @staticmethod
    def get_csv():
        orig_csv = Path(__file__).parent / "files" / "ecommerce_products.csv"  # update
        return orig_csv

    @staticmethod
    def urls():
        return {}

    @staticmethod
    def download(split: Literal["train", "dev", "test"]) -> Benchmark:
        assert split == "dev"
        os.makedirs(EcommerceRandom.dir(), exist_ok=True)
        return EcommerceRandom.load_from_disk(split)

    @staticmethod
    def load_from_disk(split: Literal["train", "dev", "test"]) -> "Benchmark":
        assert split == "dev"
        os.makedirs(EcommerceRandom.dir(), exist_ok=True)
        benchmark = EcommerceRandom(
            split,
            ExperimentalDatabase.load_from_files(
                db_name=EcommerceRandom.name(),
                split=split,
                table_names=["products"],
                paths=[EcommerceRandom.get_csv()],
                text_columns=[
                    InPlaceColumn("products.description"),
                ],
                image_columns=[
                    RemoteColumn("products.product_image", "products.product_image")
                ],
            ),
            EcommerceRandom.generate_random_queries(split),
        )
        return benchmark

    @classmethod
    def _get_query_shapes(cls) -> Sequence[QueryShape]:
        return ECOMMERCE_QUERY_SHAPES

    @classmethod
    def _get_operator_options(cls) -> Sequence[OperatorOption]:
        return ECOMMERCE_OPERATOR_OPTIONS

    @classmethod
    def _single_filter_shape(cls) -> Union[QueryShape, Dict[str, QueryShape]]:
        return SINGLE_FILTER_SHAPE


class EcommerceRandomLarge(RandomBenchmark):
    @classmethod
    def name(cls) -> str:
        return "ecommerce_random_large"

    @property
    def has_ground_truth(self) -> bool:
        return False

    @staticmethod
    def get_csv():
        orig_csv = (
            Path(__file__).parent / "files" / "ecommerce_products_large.csv"
        )  # update
        return orig_csv

    @staticmethod
    def urls():
        return {}

    @staticmethod
    def download(split: Literal["train", "dev", "test"]) -> Benchmark:
        assert split == "dev"
        os.makedirs(EcommerceRandomLarge.dir(), exist_ok=True)
        return EcommerceRandomLarge.load_from_disk(split)

    @staticmethod
    def load_from_disk(split: Literal["train", "dev", "test"]) -> "Benchmark":
        assert split == "dev"
        os.makedirs(EcommerceRandomLarge.dir(), exist_ok=True)
        benchmark = EcommerceRandomLarge(
            split,
            ExperimentalDatabase.load_from_files(
                db_name=EcommerceRandomLarge.name(),
                split=split,
                table_names=["products"],
                paths=[EcommerceRandomLarge.get_csv()],
                text_columns=[
                    InPlaceColumn("products.description"),
                ],
                image_columns=[
                    RemoteColumn("products.product_image", "products.product_image")
                ],
            ),
            EcommerceRandomLarge.generate_random_queries(split),
        )
        return benchmark

    @classmethod
    def _get_query_shapes(cls) -> Sequence[QueryShape]:
        return ECOMMERCE_QUERY_SHAPES

    @classmethod
    def _get_operator_options(cls) -> Sequence[OperatorOption]:
        return ECOMMERCE_OPERATOR_OPTIONS

    @classmethod
    def _single_filter_shape(cls) -> Union[QueryShape, Dict[str, QueryShape]]:
        return SINGLE_FILTER_SHAPE


SINGLE_FILTER_SHAPE = QueryShape(
    OperatorPlaceholder(
        LogicalFilter,
        inputs=[VirtualTableIdentifier("products")],
        output=VirtualTableIdentifier("output"),
    ),
)

ECOMMERCE_OPERATOR_OPTIONS = [
    # Filters on product_image
    OperatorOption(
        LogicalFilter,
        "{products.product_image} contains at least one human",
    ),
    OperatorOption(
        LogicalFilter,
        "{products.product_image} shows an item that is predominantly dark-colored",
    ),
    OperatorOption(
        LogicalFilter,
        "{products.product_image} shows a t-shirt",
    ),
    OperatorOption(
        LogicalFilter,
        "{products.product_image} shows an item intended for a male audience",
    ),
    OperatorOption(
        LogicalFilter,
        "{products.product_image} shows an item designed to carry or hold objects",
    ),
    OperatorOption(
        LogicalFilter,
        "{products.product_image} shows a football or soccer team jersey",
    ),
    OperatorOption(
        LogicalFilter,
        "{products.product_image} shows more than one item",
    ),
    OperatorOption(
        LogicalFilter,
        "{products.product_image} shows only the legs of the model",
    ),
    # Extract from product_image
    OperatorOption(
        LogicalExtract,
        "Extract the dominant [color] visible in {products.product_image}",
    ),
    OperatorOption(
        LogicalExtract,
        "Extract the [brand] visible in {products.product_image}; return null if no brand is visible",
    ),
    # Filters on product_description
    OperatorOption(
        LogicalFilter,
        "{products.description} mentions that the item is discounted",
    ),
    OperatorOption(
        LogicalFilter,
        "{products.description} contains warranty information",
    ),
    # Extract from product_description
    OperatorOption(
        LogicalExtract,
        "Extract the [discount_amount] mentioned in {products.description} (e.g. 20% discount); return 0 if not mentioned",
    ),
    OperatorOption(
        LogicalExtract,
        "Extract the recommended [wash_temperature] (e.g., cold, warm, 30 degrees, cold at 30 degrees) mentioned in {products.description}; return null if not specified",
    ),
    OperatorOption(
        LogicalExtract,
        "Extract the dominant [colour] from {products.description}",
    ),
    OperatorOption(
        LogicalExtract,
        "Extract the [fitting] or fit type (e.g., regular, comfort, athletic, fitted) mentioned in {products.description}; return null if not mentioned",
    ),
    OperatorOption(
        LogicalExtract,
        "Extract the [brand_name] mentioned in {products.description}; return null if no brand is explicitly mentioned",
    ),
    OperatorOption(
        LogicalExtract,
        "Extract the dominant [fabric_composition] (e.g., cotton, polyester, spandex, nylon) mentioned in {products.description}; return null if not specified",
    ),
    OperatorOption(
        LogicalExtract,
        "Extract the intended [activity] or sport (e.g., football, running, training, casual) mentioned in {products.description}; return null if not specified",
    ),
    OperatorOption(
        LogicalExtract,
        "Extract the intended [gender] or target audience (e.g., male, female, unisex) mentioned in {products.description}; return unisex if not specified",
    ),
    OperatorOption(
        LogicalExtract,
        "Extract the [closure_type] or fastening type mentioned in {products.description} (e.g., lace-up, velcro, buckle, zip, button, tang clasp); return null if not specified",
    ),
    OperatorOption(
        LogicalExtract,
        "Extract any [size_details] or measurement details mentioned in {products.description} (e.g., case diameter, case thickness, inseam length, shoe size); return null if not specified",
    ),
]

ECOMMERCE_QUERY_SHAPES = [
    # 2 operators: filter + extract
    QueryShape(
        RandomOrder(
            OperatorPlaceholder(
                LogicalFilter,
                inputs=[VirtualTableIdentifier("products")],
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
    # 2 operators: filter + filter
    QueryShape(
        RandomOrder(
            OperatorPlaceholder(
                LogicalFilter,
                inputs=[VirtualTableIdentifier("products")],
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
    # 2 operators: extract + extract
    QueryShape(
        RandomOrder(
            OperatorPlaceholder(
                LogicalExtract,
                inputs=[VirtualTableIdentifier("products")],
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
            "num_sem_filter": 0,
            "num_sem_extract": 2,
        },
    ),
    # 3 operators: filter + extract + extract
    QueryShape(
        RandomOrder(
            OperatorPlaceholder(
                LogicalFilter,
                inputs=[VirtualTableIdentifier("products")],
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
    # 3 operators: filter + filter + extract
    QueryShape(
        RandomOrder(
            OperatorPlaceholder(
                LogicalFilter,
                inputs=[VirtualTableIdentifier("products")],
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
    # 4 operators: filter + extract + extract + extract
    QueryShape(
        RandomOrder(
            OperatorPlaceholder(
                LogicalFilter,
                inputs=[VirtualTableIdentifier("products")],
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
    # 4 operators: filter + filter + extract + extract
    QueryShape(
        RandomOrder(
            OperatorPlaceholder(
                LogicalFilter,
                inputs=[VirtualTableIdentifier("products")],
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
