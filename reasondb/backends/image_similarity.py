import requests
import pandas as pd
from typing import TYPE_CHECKING, Tuple
from typing import Dict


from reasondb.backends.backend import Backend
from reasondb.database.indentifier import (
    ConcreteColumn,
    HiddenColumnIdentifier,
    RealColumnIdentifier,
)
from reasondb.utils.logging import FileLogger
from tqdm import tqdm

import numpy as np

if TYPE_CHECKING:
    from reasondb.database.database import Database


BATCH_SIZE = 100
PORT_IMAGE_SIM = 5005


class ImageSimilarityBackend(Backend):
    def __init__(self, model_name):
        self.check_model_name = model_name
        self.embed_cols: Dict[RealColumnIdentifier, HiddenColumnIdentifier] = {}

    def setup(
        self,
        logger: FileLogger,
    ):
        result = requests.get(f"http://localhost:{PORT_IMAGE_SIM}/status")
        assert result.status_code == 200
        json_response = result.json()
        assert json_response["status"] == "alive"
        assert json_response["model_name"] == self.check_model_name
        logger.info(
            "__name__", f"Image similarity model {self.check_model_name} is ready"
        )
        self.embed_dim = json_response["embed_dim"]

    async def wind_down(self):
        pass

    async def prepare(
        self,
        database: "Database",
        embed_cols: Dict["RealColumnIdentifier", "HiddenColumnIdentifier"],
        logger: FileLogger,
    ):
        collect_images = []
        num_images = 0
        self.embed_cols = embed_cols
        for img_col, embed_col in embed_cols.items():
            logger.info(
                __name__,
                f"Computing image embeddings for {img_col}. Storing in {embed_col}",
            )
            get_img_sql = f"SELECT _index_{img_col.table_name}, {img_col.column_name} FROM {img_col.table_name} WHERE {embed_col.column_name} IS NULL"
            for idx, img in database.sql(get_img_sql).fetchall():
                collect_images.append((embed_col, idx, img))
                num_images += 1

        # insert embeddings
        upper_pbar = tqdm(
            total=num_images,
            desc="Inserting embeddings",
            position=0,
            leave=True,
        )
        cols_to_build_indexes = set()
        for i in range(0, len(collect_images), BATCH_SIZE):
            batch = collect_images[i : i + BATCH_SIZE]
            image_paths = [
                {"index": idx, "path": img_path, "embed_col": str(embed_col)}
                for embed_col, idx, img_path in batch
            ]
            result = requests.post(
                f"http://localhost:{PORT_IMAGE_SIM}/image_embeddings",
                json={"image_paths": image_paths},
            )
            assert result.status_code == 200
            json_response = result.json()
            for embed_col, data in json_response.items():
                embed_col = HiddenColumnIdentifier(embed_col)

                database.use_table(embed_col.table_name)
                df = pd.DataFrame(data)
                with database.temp_register_df(df, "__tmp__"):
                    database.sql(
                        f"UPDATE {embed_col.table_name} SET {embed_col.column_name} = __tmp__.embeddings FROM __tmp__ WHERE {embed_col.table_name}._index_{embed_col.table_name} == __tmp__.indexes; "
                    )
                cols_to_build_indexes.add(embed_col)
            upper_pbar.update(BATCH_SIZE)

        for embed_col in self.embed_cols.values():
            database.use_table(embed_col.table_name)
            database.sql(
                f"UPDATE {embed_col.table_name} SET {embed_col.column_name} = ? WHERE {embed_col.table_name}.{embed_col.column_name} IS NULL; ",
                [[0.0] * self.embed_dim],
            )

        # build indexes
        for embed_col in cols_to_build_indexes:
            database.use_table(embed_col.table_name)
            database.sql(
                f"SET hnsw_enable_experimental_persistence=true; "
                f"CREATE INDEX {embed_col.table_name}_{embed_col.column_name}_cos_idx ON {embed_col.table_name} USING HNSW ({embed_col.column_name}) WITH (metric = 'cosine'); "
            )
        database.sql("USE memory;")
        logger.info(
            __name__,
            f"Image similarity indexes and model {self.check_model_name} are ready",
        )
        upper_pbar.close()

    def notify_materialization(
        self,
        column: RealColumnIdentifier,
        coupled_column: HiddenColumnIdentifier,
    ):
        self.embed_cols[column] = coupled_column  

    def shutdown(self, logger: FileLogger):
        pass

    async def run(
        self,
        concrete_image_column: "ConcreteColumn",
        description: str,
    ) -> Tuple["HiddenColumnIdentifier", np.ndarray]:
        assert isinstance(concrete_image_column, RealColumnIdentifier)
        embed_col = self.embed_cols[concrete_image_column]
        result = requests.post(
            f"http://localhost:{PORT_IMAGE_SIM}/text_embeddings",
            json={"description": description},
        )
        assert result.status_code == 200
        result_json = result.json()
        desc_emb = np.array(result_json["text_embedding"], dtype=np.float32)
        return embed_col, desc_emb

    def get_operation_identifier(self) -> str:
        return f"ImageSimilarityBackend-{self.check_model_name}"

    def setup_index(self, table, column):
        """Setup duckdb index."""
        pass
