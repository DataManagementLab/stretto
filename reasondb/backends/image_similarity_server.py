import logging
import argparse
from pathlib import Path
from dataclasses import dataclass
from flask import Flask, request
from flask_restful import Resource, Api
from typing import List, Optional
from typing import Dict

from torch.utils.data import Dataset

from reasondb.backends.image_similarity import PORT_IMAGE_SIM
from reasondb.database.indentifier import (
    HiddenColumnIdentifier,
    RealColumnIdentifier,
)
from PIL import Image
from transformers import AutoProcessor, BlipForImageTextRetrieval
import torch

import numpy as np
from PIL import ImageFile


MODEL_NAME = "Salesforce/blip-itm-base-coco"
BATCH_SIZE = 10
DATALOADER_NUM_WORKERS = 8
# Image.MAX_IMAGE_PIXELS = None
IMAGE_MAX_PIXELS = 1400 * 1400
ImageFile.LOAD_TRUNCATED_IMAGES = True


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
api = Api(app)


@dataclass
class ImageJob:
    index: int
    path: Path
    embed_col: str


class ImageSimilarityModelWrapper:
    def __init__(self, model_name, device_id):
        self.model_name = model_name
        self.embed_dim = 256
        self.embed_cols: Dict[RealColumnIdentifier, HiddenColumnIdentifier] = {}
        self.device_id = device_id
        self.init()

    def init(self):
        # load model first
        device = torch.device(
            f"cuda:{self.device_id}" if torch.cuda.is_available() else "cpu"
        )
        logger.info(f"Loading Image similarity model: {self.model_name}")
        self.model = BlipForImageTextRetrieval.from_pretrained(self.model_name)
        logger.info(f"Moving it to: {device}")
        self.model.to(device)  # type: ignore
        logger.info(f"Loading processor: {self.model_name}")
        self.processor = AutoProcessor.from_pretrained(self.model_name, use_fast=True)
        logger.info(f"Model loaded: {self.model_name}")

    def compute_image_embeddings(self, image_jobs: List[ImageJob]):
        device = torch.device(
            f"cuda:{self.device_id}" if torch.cuda.is_available() else "cpu"
        )
        dataset = ImageDataset(image_jobs)
        collator = ImageCollator(processor=self.processor)

        dataloader = torch.utils.data.DataLoader(
            batch_size=BATCH_SIZE,
            dataset=dataset,
            collate_fn=collator,
            num_workers=DATALOADER_NUM_WORKERS,
        )
        all_embeddings = []
        all_indexes = []
        all_embed_cols = []
        for data in dataloader:
            # try:
            if "pixel_values" not in data:
                continue  # Skip if no images are loaded

            embeddings = self.embed_images(data, device)
            all_embeddings.append(embeddings)
            all_indexes.extend(data["indexes"])
            all_embed_cols.extend(data["embed_col"])
        result_embeddings = np.vstack(all_embeddings)
        result_indexes = all_indexes
        result_embed_cols = all_embed_cols
        return result_embeddings, result_indexes, result_embed_cols
        # except Exception as e:
        #     print(f"Error processing images: {e}")  # Log the error

    def embed_images(self, inputs, device):
        with torch.no_grad():
            outputs = self.model.vision_model(
                pixel_values=inputs["pixel_values"].to(device)
            )
            pooled = outputs["pooler_output"]
            image_feat = self.model.vision_proj(pooled)
            return image_feat.cpu().numpy().astype(np.float32)

    def get_text_embeddings(self, query):
        """Return embeddings for text.

        Args:
            query (str): text query
        Returns:
            embeddings for text
        """
        device = self.model.device
        with torch.no_grad():
            inputs = self.processor(text=query, return_tensors="pt").to(device)
            question_embeds = self.model.text_encoder(
                input_ids=inputs.input_ids,
                attention_mask=None,
                return_dict=False,
            )
            text_feat = self.model.text_proj(question_embeds[0][:, 0, :])
            return text_feat[0].cpu().numpy()


class ImageDataset(Dataset):
    def __init__(self, image_jobs: List[ImageJob]):
        self.image_jobs = image_jobs

    def __len__(self):
        return len(self.image_jobs)

    def __getitem__(self, idx):
        return self.image_jobs[idx]


class ImageCollator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, batch_jobs: List[ImageJob]):
        """Collate images into a batch."""
        images = []
        indexes = []
        embed_cols = []
        for img_job in batch_jobs:
            img_path = img_job.path
            image = self.load_image(img_path)
            if image is not None:
                images.append(image)
                indexes.append(img_job.index)
                embed_cols.append(img_job.embed_col)
        if len(images) == 0:
            inputs = {}
        else:
            inputs = self.processor(
                images=images,
                return_tensors="pt",
                padding=True,
            )
        inputs["indexes"] = indexes
        inputs["embed_col"] = embed_cols
        return inputs

    def load_image(self, image_path) -> Optional[Image.Image]:
        """Loads an image from the database. If the image is too large, it is down-sampled."""
        try:
            with Image.open(image_path) as image:
                image = image.convert("RGB")  # Ensure image is in RGB format
                if int(np.prod(image.size)) > IMAGE_MAX_PIXELS:
                    ratio = np.sqrt(IMAGE_MAX_PIXELS / np.prod(image.size))
                    image.thumbnail(
                        (int(image.size[0] * ratio), int(image.size[1] * ratio))
                    )
                return image
        except Exception as e:
            print(str(e))
            return None


model_wrapper = None


class Status(Resource):
    def get(self):
        assert model_wrapper is not None
        return {
            "status": "alive",
            "model_name": model_wrapper.model_name,
            "embed_dim": model_wrapper.embed_dim,
        }, 200


class ImageEmbeddings(Resource):
    def post(self):
        """Expects JSON of the form:
        {
            "image_paths": [
                {
                    "index": 0,
                    "path": "/path/to/image1.jpg",
                    "embed_col": "image_embedding_col1"
                },
                {
                    "index": 1,
                    "path": "/path/to/image2.jpg",
                    "embed_col": "image_embedding_col1"
                },
                ...
            ]
        }
        """
        data = request.get_json(force=True)
        img_objs = data["image_paths"]
        image_jobs = [
            ImageJob(index=o["index"], path=o["path"], embed_col=o["embed_col"])
            for o in img_objs
        ]
        assert model_wrapper is not None
        img_emb, img_idx, img_embed_col = model_wrapper.compute_image_embeddings(
            image_jobs
        )
        grouped = {}
        for emb, idx, col in zip(img_emb, img_idx, img_embed_col):
            if col not in grouped:
                grouped[col] = {
                    "indexes": [],
                    "embeddings": [],
                }
            grouped[col]["indexes"].append(idx)
            grouped[col]["embeddings"].append(emb.tolist())
        return grouped, 200


class TextEmbeddings(Resource):
    def post(self):
        """
        Expects JSON of the form:
        {
            "description": "A image description"
        }
        """
        data = request.get_json(force=True)
        assert model_wrapper is not None
        desc_emb = model_wrapper.get_text_embeddings(data["description"])
        return {"text_embedding": desc_emb.tolist()}, 200


api.add_resource(Status, "/status")
api.add_resource(ImageEmbeddings, "/image_embeddings")
api.add_resource(TextEmbeddings, "/text_embeddings")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device-id",
        type=int,
        default=0,
        help="Device ID for GPU to use",
    )
    args = parser.parse_args()
    device_id = args.device_id
    model_wrapper = ImageSimilarityModelWrapper(MODEL_NAME, device_id)
    app.run(host="127.0.0.1", port=PORT_IMAGE_SIM, debug=False)
