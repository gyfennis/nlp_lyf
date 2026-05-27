import json
from kafka import KafkaProducer
from app.config import settings

_producer = None


def get_kafka_producer() -> KafkaProducer:
    global _producer
    if _producer is None:
        _producer = KafkaProducer(
            bootstrap_servers=settings.KAFKA_BOOTSTRAP_SERVERS,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        )
    return _producer


def send_parse_task(file_name: str, file_path: str, file_id: int):
    """发送文档解析任务到 Kafka"""
    producer = get_kafka_producer()
    producer.send(
        settings.KAFKA_TOPIC,
        value={"file_name": file_name, "file_path": file_path, "id": file_id},
    )
    producer.flush()
