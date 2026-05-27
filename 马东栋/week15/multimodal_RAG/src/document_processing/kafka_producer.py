import json
from datetime import datetime
from kafka import KafkaProducer
from kafka.errors import KafkaError
from src.config import get_kafka_config


class DocumentKafkaProducer:
    def __init__(self, bootstrap_servers: str = None, topic: str = None):
        cfg = get_kafka_config()
        servers = bootstrap_servers or cfg["bootstrap_servers"]
        self.topic = topic or cfg["topic"]
        self.producer = KafkaProducer(
            bootstrap_servers=servers,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            key_serializer=lambda k: k.encode("utf-8") if k else None,
        )

    def send_message(
        self, document_id: str, file_path: str, file_type: str
    ):
        message = {
            "event_type": "document_uploaded",
            "document_id": document_id,
            "file_path": file_path,
            "file_type": file_type,
            "upload_time": datetime.now().isoformat(),
            "retry_count": 0,
        }

        future = self.producer.send(self.topic, key=document_id, value=message)
        try:
            record_metadata = future.get(timeout=10)
            return record_metadata
        except KafkaError as e:
            raise Exception(f"Failed to send Kafka message: {e}")

    def close(self):
        self.producer.close()
