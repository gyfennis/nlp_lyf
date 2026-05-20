import os
import time
import json
from kafka import KafkaConsumer, KafkaProducer
from src.services.pdf_parser import pdf_parser
from src.services.embedder import text_embedder, image_embedder
from src.services.vector_store import vector_store
from src.models.database import get_session, Document
from src.config import config


class DocumentWorker:
    def __init__(self, kafka_brokers: list):
        self.consumer = KafkaConsumer(
            "pdf_parse",
            bootstrap_servers=kafka_brokers,
            value_deserializer=lambda m: json.loads(m.decode("utf-8")),
            auto_offset_reset="earliest"
        )
        self.producer = KafkaProducer(
            bootstrap_servers=kafka_brokers,
            value_deserializer=lambda m: json.dumps(m).encode("utf-8")
        )
        vector_store.create_collections()

    def process(self, message: dict):
        doc_id = message["doc_id"]
        file_path = message["file_path"]
        knowledge_base_id = message["knowledge_base_id"]

        session = get_session()
        try:
            doc = session.query(Document).filter(Document.id == doc_id).first()
            doc.status = "processing"
            session.commit()

            # 1. 解析PDF
            parse_result = pdf_parser.parse(file_path)
            markdown_content = parse_result.get("markdown", "")
            image_files = parse_result.get("images", [])

            # 2. 文本chunk切分
            chunks = pdf_parser.parse_chunk(markdown_content)

            # 3. 向量化并存储
            chunk_texts = [c["content"] for c in chunks]
            chunk_vectors = text_embedder.encode(chunk_texts)
            chunk_metadata = [
                {"doc_id": doc_id, "chunk_index": c["chunk_index"]}
                for c in chunks
            ]
            vector_store.insert_text(chunk_vectors, chunk_texts, chunk_metadata)

            # 4. 图像向量化并存储
            if image_files:
                image_vectors = image_embedder.encode(image_files)
                image_metadata = [
                    {"doc_id": doc_id, "image_index": i}
                    for i in range(len(image_files))
                ]
                vector_store.insert_image(image_vectors, image_files, image_metadata)

            # 5. 更新状态
            doc.status = "completed"
            session.commit()

            # 6. 发送完成通知
            self.producer.send("parse_complete", {
                "doc_id": doc_id,
                "status": "completed"
            })

        except Exception as e:
            doc.status = "failed"
            session.commit()
            self.producer.send("parse_complete", {
                "doc_id": doc_id,
                "status": "failed",
                "error": str(e)
            })
        finally:
            session.close()

    def run(self):
        print("Document worker started, waiting for messages...")
        for message in self.consumer:
            self.process(message.value)


if __name__ == "__main__":
    KAFKA_BROKERS = ["localhost:9092"]
    worker = DocumentWorker(KAFKA_BROKERS)
    worker.run()