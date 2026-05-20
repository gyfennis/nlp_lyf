"""Kafka consumer worker for document parsing tasks."""
import time
import asyncio
from typing import Optional

from kafka import KafkaConsumer
from sqlalchemy.orm import Session

from app.core.config import settings
from app.models.database import Document, DocumentStatus, TextChunk, ImageEmbedding, engine
from app.services.parser import parser_service
from app.services.embedding import embedding_service
from app.services.retriever import retriever_service
from app.services.storage import storage_service
from loguru import logger


class DocumentWorker:
    """Worker that consumes document parsing tasks from Kafka."""

    def __init__(self):
        self.running = False
        self.max_retries = 3
        self.retry_delay = 30

    def get_db_session(self) -> Session:
        """Get database session."""
        return Session(bind=engine)

    async def process_document(self, doc_id: int, pdf_path: str, db: Session) -> None:
        """Process a single document: parse, chunk, embed, and store."""
        logger.info(f"Processing document {doc_id}: {pdf_path}")

        doc = db.query(Document).filter(Document.id == doc_id).first()
        if not doc:
            logger.error(f"Document {doc_id} not found in database")
            return

        doc.status = DocumentStatus.PROCESSING
        db.commit()

        try:
            # Parse PDF with MinerU
            markdown_content, images_list = await parser_service.parse_document(pdf_path)

            # Save parsed content
            parsed_md_path, image_paths = storage_service.save_parsed_content(
                doc_id, markdown_content, {img["name"]: img["data"] for img in images_list}
            )

            # Extract and chunk text
            chunks = embedding_service.chunk_text(markdown_content)

            # Extract page numbers from markdown (simplified)
            page_numbers = [1] * len(chunks)  # Placeholder - would need proper page tracking

            # Encode and store text embeddings
            if chunks:
                embeddings = embedding_service.encode_texts(chunks)
                vector_ids = retriever_service.insert_text_embeddings(
                    doc_id, chunks, embeddings, page_numbers
                )

                # Save text chunks to database
                for i, (chunk, vector_id, page) in enumerate(zip(chunks, vector_ids, page_numbers)):
                    text_chunk = TextChunk(
                        document_id=doc_id,
                        content=chunk,
                        chunk_index=i,
                        page_number=page,
                        vector_id=vector_id
                    )
                    db.add(text_chunk)

            # Encode and store image embeddings
            if image_paths:
                image_embeddings = embedding_service.encode_images(image_paths)
                captions = [img.get("caption", "") for img in images_list]
                img_pages = [img.get("page_number", 1) for img in images_list]

                vector_ids = retriever_service.insert_image_embeddings(
                    doc_id, image_paths, image_embeddings, captions, img_pages
                )

                # Save image embeddings to database
                for path, vector_id, caption, page in zip(image_paths, vector_ids, captions, img_pages):
                    img_emb = ImageEmbedding(
                        document_id=doc_id,
                        image_path=path,
                        caption=caption,
                        page_number=page,
                        vector_id=vector_id
                    )
                    db.add(img_emb)

            doc.status = DocumentStatus.COMPLETED
            doc.page_count = len(set(page_numbers))
            logger.info(f"Document {doc_id} processed successfully")

        except Exception as e:
            logger.error(f"Error processing document {doc_id}: {str(e)}")
            doc.status = DocumentStatus.FAILED
            doc.error_message = str(e)

        db.commit()

    async def process_with_retry(self, doc_id: int, pdf_path: str) -> None:
        """Process document with retry logic."""
        db = self.get_db_session()

        for attempt in range(self.max_retries):
            try:
                await self.process_document(doc_id, pdf_path, db)
                return
            except Exception as e:
                if attempt < self.max_retries - 1:
                    logger.warning(f"Retry {attempt + 1} for document {doc_id}: {str(e)}")
                    time.sleep(self.retry_delay)
                else:
                    doc = db.query(Document).filter(Document.id == doc_id).first()
                    if doc:
                        doc.status = DocumentStatus.FAILED
                        doc.error_message = f"Failed after {self.max_retries} attempts: {str(e)}"
                        db.commit()
        finally:
            db.close()

    def start(self) -> None:
        """Start consuming messages from Kafka."""
        self.running = True
        logger.info("Starting document worker...")

        try:
            consumer = KafkaConsumer(
                settings.kafka_topic,
                bootstrap_servers=settings.kafka_bootstrap_servers,
                group_id=settings.kafka_consumer_group,
                value_deserializer=lambda m: m.decode("utf-8"),
                auto_offset_reset="earliest",
                enable_auto_commit=True
            )

            logger.info(f"Connected to Kafka, listening on topic: {settings.kafka_topic}")

            for message in consumer:
                if not self.running:
                    break

                try:
                    doc_id, pdf_path = message.value.split("|")
                    doc_id = int(doc_id)
                    logger.info(f"Received task: doc_id={doc_id}, path={pdf_path}")

                    asyncio.run(self.process_with_retry(doc_id, pdf_path))

                except Exception as e:
                    logger.error(f"Error parsing message: {str(e)}")

        except Exception as e:
            logger.error(f"Kafka connection error: {str(e)}")
            logger.info("Worker stopping...")

    def stop(self) -> None:
        """Stop the worker."""
        self.running = False
        logger.info("Worker stop requested")


if __name__ == "__main__":
    worker = DocumentWorker()
    worker.start()