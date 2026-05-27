import json
from typing import Protocol

from app.config import Settings


class ParseQueue(Protocol):
    def enqueue_parse_job(self, *, file_id: int, file_name: str, file_path: str) -> None: ...


class KafkaParseQueue:
    def __init__(self, settings: Settings):
        self._settings = settings
        self._producer = None

    def _get_producer(self):
        if self._producer is None:
            from kafka import KafkaProducer

            self._producer = KafkaProducer(
                bootstrap_servers=self._settings.kafka_bootstrap,
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            )
        return self._producer

    def enqueue_parse_job(self, *, file_id: int, file_name: str, file_path: str) -> None:
        producer = self._get_producer()
        producer.send(
            self._settings.kafka_topic,
            value={"file_name": file_name, "file_path": file_path, "id": file_id},
        )
        producer.flush()


class InMemoryParseQueue:
    """Test double: records jobs without Kafka."""

    def __init__(self):
        self.jobs: list[dict] = []

    def enqueue_parse_job(self, *, file_id: int, file_name: str, file_path: str) -> None:
        self.jobs.append(
            {"id": file_id, "file_name": file_name, "file_path": file_path}
        )
