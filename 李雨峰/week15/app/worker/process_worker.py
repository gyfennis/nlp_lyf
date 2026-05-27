"""
离线文档解析 Worker
从 Kafka 消费文档解析任务，调用 MinerU 解析 PDF，
将解析后的 markdown chunk 编码并存储到 Milvus 向量数据库。
"""
import os
import glob
import json
import subprocess
import traceback
from kafka import KafkaConsumer
from app.config import settings
from app.core.milvus_client import get_milvus_client
from app.services.embedding import (
    encode_text,
    encode_image,
    split_text2chunks,
    extract_image_from_markdown,
)


def process_document(markdown_path: str, file_id: int, file_name: str, file_path: str):
    """
    对解析后的 markdown 文档进行 chunk 划分、编码、存储到 Milvus
    """
    with open(markdown_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    chunks = split_text2chunks(lines)
    markdown_dir = os.path.dirname(markdown_path)
    client = get_milvus_client()

    for chunk in chunks:
        try:
            # 编码文本（不含图片行的纯文本）
            text_no_image = "\n".join(
                [line for line in chunk.split("\n") if not line.startswith("![")]
            )
            text_bge_emb, text_clip_emb = encode_text(text_no_image)

            # 编码图片（如果有）
            images = extract_image_from_markdown(chunk, markdown_dir)
            if images:
                image_clip_emb = encode_image(images[0])
            else:
                image_clip_emb = [0.0] * settings.CLIP_DIM

            # 插入 Milvus
            data = [
                {
                    "text_vector": text_bge_emb,
                    "clip_text_vector": text_clip_emb,
                    "clip_image_vector": image_clip_emb,
                    "text": chunk,
                    "db_id": file_id,
                    "file_name": file_name,
                    "file_path": file_path,
                }
            ]
            client.insert(
                collection_name=settings.MILVUS_COLLECTION,
                data=data,
            )
            print(f"[OK] Inserted chunk for file_id={file_id}, file={file_name}")

        except Exception:
            traceback.print_exc()


def main():
    """主循环：消费 Kafka 消息，解析文档"""
    consumer = KafkaConsumer(
        settings.KAFKA_TOPIC,
        bootstrap_servers=settings.KAFKA_BOOTSTRAP_SERVERS,
        enable_auto_commit=True,
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
    )

    os.makedirs(settings.PROCESSED_DIR, exist_ok=True)

    print(f"[Worker] Listening on Kafka topic: {settings.KAFKA_TOPIC}")
    print(f"[Worker] MinerU service: {settings.MINERU_SERVICE_URL}")

    for msg in consumer:
        try:
            file_name = msg.value["file_name"]
            file_path = msg.value["file_path"]
            file_id = msg.value["id"]

            if not os.path.exists(file_path):
                print(f"[WARN] File not found: {file_path}")
                continue

            # 步骤2: 调用 MinerU 解析文档
            output_base = settings.PROCESSED_DIR
            cmd = (
                f"mineru -p \"{file_path}\" -o \"{output_base}\" "
                f"-b vlm-http-client -u {settings.MINERU_SERVICE_URL}"
            )
            print(f"[Worker] Running: {cmd}")
            subprocess.check_output(cmd, shell=True, timeout=settings.MINERU_TIMEOUT)

            # 步骤3: 查找解析后的 markdown 文件
            dir_name = os.path.basename(file_path).split(".")[0]
            markdown_pattern = os.path.join(output_base, dir_name, "**", "**.md")
            markdown_files = glob.glob(markdown_pattern, recursive=True)

            if not markdown_files:
                print(f"[ERROR] No markdown found for {file_name}")
                continue

            # 步骤4: 编码并存储
            process_document(markdown_files[0], file_id, file_name, file_path)
            print(f"[DONE] Processed: {file_name}")

        except subprocess.TimeoutExpired:
            print(f"[ERROR] MinerU timeout for {msg.value.get('file_name')}")
        except Exception as e:
            print(f"[ERROR] {e}")
            traceback.print_exc()


if __name__ == "__main__":
    main()
