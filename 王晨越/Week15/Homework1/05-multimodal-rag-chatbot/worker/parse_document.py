"""
Kafka consumer: MinerU parse -> chunk -> embed -> Milvus.

Run: python -m worker.parse_document
Requires: Kafka, MinerU CLI, GPU models (see offline_precess_worker.py).
"""

import glob
import json
import os
import subprocess
import traceback

from kafka import KafkaConsumer
from pymilvus import MilvusClient

from app.config import get_settings
from app.ingestion.chunking import split_text2chunks

settings = get_settings()

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")


def _load_models():
    from sentence_transformers import SentenceTransformer

    bge = SentenceTransformer(settings.bge_model_path)
    clip = SentenceTransformer(
        settings.clip_model_path, trust_remote_code=True, truncate_dim=1024
    )
    return bge, clip


def encode_text_and_image(text: str, markdown_path: str, bge_model, clip_model):
    text_with_no_image = "\n".join(
        line for line in text.split("\n") if not line.startswith("![")
    )
    text_with_image = [line for line in text.split("\n") if line.startswith("![")]

    try:
        text_bge_embedding = list(
            bge_model.encode(text_with_no_image, normalize_embeddings=True)
        )
    except Exception:
        traceback.print_exc()
        text_bge_embedding = [0.0] * 512

    try:
        text_clip_embedding = list(
            clip_model.encode(text_with_no_image, normalize_embeddings=True)
        )
    except Exception:
        traceback.print_exc()
        text_clip_embedding = [0.0] * 1024

    if text_with_image:
        image_path = text_with_image[0].split("](")[1].split(")")[0]
        image_real_path = os.path.dirname(markdown_path) + image_path.split("/")[-1]
        try:
            image_clip_embedding = list(
                clip_model.encode(image_real_path, normalize_embeddings=True)
            )
        except Exception:
            traceback.print_exc()
            image_clip_embedding = [0.0] * 1024
    else:
        image_clip_embedding = [0.0] * 1024

    return text_bge_embedding, text_clip_embedding, image_clip_embedding


def encode_document(path: str, file_id: int, file_name: str, file_path: str, client, bge, clip):
    lines = open(path, encoding="utf-8").readlines()
    chunks = split_text2chunks(lines)
    for chunk in chunks:
        try:
            text_bge, text_clip, image_clip = encode_text_and_image(chunk, path, bge, clip)
            client.insert(
                collection_name=settings.milvus_collection,
                data=[
                    {
                        "text_vector": text_bge,
                        "clip_text_vector": text_clip,
                        "clip_image_vector": image_clip,
                        "text": chunk,
                        "db_id": file_id,
                        "file_name": file_name,
                        "file_path": file_path,
                    }
                ],
            )
        except Exception:
            traceback.print_exc()


def run_consumer():
    if not settings.milvus_uri:
        raise RuntimeError("MILVUS_URI is required for worker")

    bge, clip = _load_models()
    client = MilvusClient(uri=settings.milvus_uri, token=settings.milvus_token)
    consumer = KafkaConsumer(
        settings.kafka_topic,
        bootstrap_servers=settings.kafka_bootstrap,
        enable_auto_commit=True,
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
    )

    for msg in consumer:
        try:
            value = msg.value
            file_name = value["file_name"]
            file_path = value["file_path"]
            file_id = value["id"]
            if not os.path.exists(file_path):
                continue

            subprocess.check_output(
                f"mineru -p {file_path} -o ./processed -b vlm-http-client -u http://127.0.0.1:30000",
                shell=True,
                timeout=600,
            )

            markdown_paths = glob.glob(
                os.path.join(
                    "./processed",
                    os.path.basename(file_path).split(".")[0],
                )
                + "/**/**.md"
            )
            if not markdown_paths:
                print(f"Failed to find markdown for {file_name}")
                continue

            encode_document(markdown_paths[0], file_id, file_name, file_path, client, bge, clip)
        except Exception as e:
            print(e)


if __name__ == "__main__":
    run_consumer()
