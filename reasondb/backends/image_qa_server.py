import argparse
import logging
from tqdm import tqdm
from pathlib import Path
from flask import Flask, request
from flask_restful import Resource, Api
from typing import List
from typing import Dict

from torch.utils.data import Dataset

from reasondb.backends.vision_model import PORT_VISION
from reasondb.database.indentifier import (
    HiddenColumnIdentifier,
    RealColumnIdentifier,
)
from PIL import Image
from transformers import (
    Blip2ForConditionalGeneration,
    Blip2Processor,
)
import torch

from PIL import ImageFile


MODEL_NAME = "Salesforce/blip2-opt-2.7b"
BATCH_SIZE = 1
DATALOADER_NUM_WORKERS = 8
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
api = Api(app)


class ImageQAModelWrapper:
    def __init__(self, model_name, device_id):
        self.model_name = model_name
        self.device_id = device_id
        self.embed_dim = 256
        self.embed_cols: Dict[RealColumnIdentifier, HiddenColumnIdentifier] = {}
        self.init()

    def init(self):
        # load model first
        device = torch.device(
            f"cuda:{self.device_id}" if torch.cuda.is_available() else "cpu"
        )
        logger.info(f"Loading Image similarity model: {self.model_name}")
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            self.model_name, torch_dtype=torch.float16
        )
        logger.info(f"Moving it to: {device}")
        self.model.to(device)  # type: ignore
        logger.info(f"Loading processor: {self.model_name}")
        self.processor: Blip2Processor = Blip2Processor.from_pretrained(
            self.model_name, use_fast=True
        )  # type: ignore
        if isinstance(self.processor, tuple):
            self.processor = self.processor[0]
        logger.info(f"Model loaded: {self.model_name}")

    def compute_image_qa_response(self, image_paths: List[str], question: str):
        device = torch.device(
            f"cuda:{self.device_id}" if torch.cuda.is_available() else "cpu"
        )
        dataset = ImageDataset(image_paths)
        collator = ImageCollator(question=question, processor=self.processor)

        dataloader = torch.utils.data.DataLoader(
            batch_size=BATCH_SIZE,
            dataset=dataset,
            collate_fn=collator,
            num_workers=DATALOADER_NUM_WORKERS,
        )
        all_responses = {}
        for data in tqdm(dataloader):
            # try:
            if "pixel_values" not in data:
                continue  # Skip if no images are loaded

            responses = self.get_responses(data, device)
            for response, path in zip(responses, data["image_paths"]):
                all_responses[str(path)] = response
        return all_responses

        # except Exception as e:
        #     print(f"Error processing images: {e}")  # Log the error

    def get_responses(self, inputs, device):
        with torch.no_grad():
            # move to device
            inputs = {k: v.to(device) for k, v in inputs.items() if k != "image_paths"}
            generated_ids = self.model.generate(**inputs)  # type: ignore
            generated_texts = self.processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )
            result_texts = [
                generated_text.split("Answer:")[-1].strip()
                for generated_text in generated_texts
            ]
            return result_texts


class ImageDataset(Dataset):
    def __init__(self, image_paths: List[str]):
        self.image_paths = image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        return Path(self.image_paths[idx])


class ImageCollator:
    def __init__(self, question: str, processor):
        self.processor = processor
        self.question = question

    def __call__(self, batch_paths: List[Path]):
        """Collate images into a batch."""
        images = []
        paths = []
        prompt = f"Question: {self.question} Answer:"
        for img_path in batch_paths:
            image = Image.open(img_path)
            images.append(image.convert("RGB"))
            paths.append(img_path)

        if len(images) == 0:
            inputs = {}
        else:
            inputs = self.processor(images=images, text=prompt, return_tensors="pt").to(
                torch.float16,  # type: ignore
            )
        inputs["image_paths"] = paths
        return inputs


model_wrapper = None


class Status(Resource):
    def get(self):
        assert model_wrapper is not None
        return {
            "status": "alive",
            "model_name": model_wrapper.model_name,
            "embed_dim": model_wrapper.embed_dim,
        }, 200


class ImageQA(Resource):
    def post(self):
        """Expects JSON of the form:
        {
            "image_paths": [
                "/path/to/image1.jpg",
                "/path/to/image2.jpg",
            ],
            "question": "How many cats are there?"
        }
        """
        data = request.get_json(force=True)
        img_paths = data["image_paths"]
        question = data["question"]
        assert model_wrapper is not None
        responses = model_wrapper.compute_image_qa_response(
            image_paths=img_paths, question=question
        )
        return responses, 200


api.add_resource(Status, "/status")
api.add_resource(ImageQA, "/image_qa")

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
    model_wrapper = ImageQAModelWrapper(MODEL_NAME, device_id)
    app.run(host="127.0.0.1", port=PORT_VISION, debug=False)
