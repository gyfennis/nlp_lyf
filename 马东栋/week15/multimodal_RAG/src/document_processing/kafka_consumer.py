import json
import uuid
from kafka import KafkaConsumer
from src.config import get_kafka_config
from src.storage.mysql_client import MySQLClient
from src.storage.milvus_client import MilvusVectorStore
from src.document_processing.mineru_client import MinerUClient
from src.document_processing.chunker import OverlapChunker
from src.retrieval.bge_recall import EmbeddingClient


class DocumentKafkaConsumer:
    def __init__(
        self,
        bootstrap_servers: str = None,
        topic: str = None,
        group_id: str = None,
    ):
        cfg = get_kafka_config()
        servers = bootstrap_servers or cfg["bootstrap_servers"]
        self.topic = topic or cfg["topic"]
        self.group_id = group_id or cfg["consumer_group"]

        self.consumer = KafkaConsumer(
            self.topic,
            bootstrap_servers=servers,
            group_id=self.group_id,
            auto_offset_reset="earliest",
            enable_auto_commit=False,
            value_deserializer=lambda m: json.loads(m.decode("utf-8")),
            consumer_timeout_ms=1000,
        )

        self.mysql = MySQLClient()
        self.milvus = MilvusVectorStore()
        self.mineru = MinerUClient()
        self.chunker = OverlapChunker()
        self.embedding_client = EmbeddingClient()

    def consume(self):
        for message in self.consumer:
            try:
                document_data = message.value
                self.process_document(document_data)
                self.consumer.commit()
            except Exception as e:
                print(f"Error processing message: {e}")
                self._handle_retry(message.value)

    def process_document(self, data: dict):
        document_id = data["document_id"]
        file_path = data["file_path"]

        # Update status to processing
        self.mysql.execute(
            "UPDATE documents SET status='processing' WHERE id=%s",
            (document_id,),
        )

        # Parse with MinerU (synchronous call within async context)
        import asyncio
        parsed = asyncio.run(self.mineru.parse(file_path))

        # Extract text content from parsed result
        text_content = self._extract_text(parsed)

        # Chunk the text
        chunks = self.chunker.chunk_text(text_content, document_id)

        if not chunks:
            self.mysql.execute(
                "UPDATE documents SET status='completed', error_message='No content extracted' WHERE id=%s",
                (document_id,),
            )
            return

        # Generate embeddings
        chunk_texts = [c.content for c in chunks]
        embeddings = asyncio.run(
            self.embedding_client.embed(chunk_texts)
        )

        # Store chunks in MySQL and Milvus
        for chunk, embedding in zip(chunks, embeddings):
            chunk_id = chunk.id
            self.mysql.execute(
                """INSERT INTO chunks (id, document_id, content, content_type, page_number, position, token_count)
                   VALUES (%s, %s, %s, %s, %s, %s, %s)""",
                (chunk_id, document_id, chunk.content, chunk.content_type,
                 chunk.page_number, chunk.position, chunk.token_count),
            )

            self.milvus.insert([{
                "id": chunk_id,
                "chunk_id": chunk_id,
                "document_id": document_id,
                "content": chunk.content,
                "content_type": chunk.content_type,
                "page_number": chunk.page_number,
                "embedding": embedding,
            }])

        # Update status to completed
        self.mysql.execute(
            "UPDATE documents SET status='completed' WHERE id=%s",
            (document_id,),
        )
        print(f"Document {document_id} processed: {len(chunks)} chunks created")

    def _extract_text(self, parsed: dict) -> str:
        """Extract text from MinerU parsed result."""
        if "content" in parsed:
            return parsed["content"]
        if "text" in parsed:
            return parsed["text"]
        if "pages" in parsed:
            texts = []
            for page in parsed["pages"]:
                if isinstance(page, dict):
                    texts.append(page.get("text", "") or str(page))
                else:
                    texts.append(str(page))
            return "\n\n".join(texts)
        return str(parsed)

    def _handle_retry(self, data: dict):
        retry_count = data.get("retry_count", 0)
        if retry_count < 3:
            data["retry_count"] = retry_count + 1
            self.mysql.execute(
                "UPDATE documents SET error_message=%s WHERE id=%s",
                (f"Retrying (attempt {retry_count + 1})", data["document_id"]),
            )
        else:
            self.mysql.execute(
                "UPDATE documents SET status='failed', error_message=%s WHERE id=%s",
                ("Max retries exceeded", data["document_id"]),
            )

    def close(self):
        self.consumer.close()
