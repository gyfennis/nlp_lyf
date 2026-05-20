"""Embedding service using BGE for text and CLIP for images."""
from typing import List
import numpy as np

from app.core.config import settings


class EmbeddingService:
    """Service for encoding text and images into vectors."""

    def __init__(self):
        self._bge_model = None
        self._clip_model = None

    @property
    def bge_model(self):
        """Lazy load BGE model."""
        if self._bge_model is None:
            from sentence_transformers import SentenceTransformer
            self._bge_model = SentenceTransformer(settings.bge_model)
        return self._bge_model

    @property
    def clip_model(self):
        """Lazy load CLIP model."""
        if self._clip_model is None:
            import clip
            import torch
            model, preprocess = clip.load("ViT-B/32", device="cpu")
            self._clip_model = (model, preprocess)
        return self._clip_model

    def encode_texts(self, texts: List[str]) -> np.ndarray:
        """Encode texts using BGE model."""
        embeddings = self.bge_model.encode(texts, normalize_embeddings=True)
        return embeddings

    def encode_images(self, image_paths: List[str]) -> np.ndarray:
        """Encode images using CLIP model."""
        import clip
        import torch
        from PIL import Image

        model, preprocess = self.clip_model
        images = [Image.open(path).convert("RGB") for path in image_paths]

        image_tensors = torch.stack([preprocess(img) for img in images])
        with torch.no_grad():
            features = model.encode_image(image_tensors)
            embeddings = features.float().numpy()

        # Normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / norms

        return embeddings

    def chunk_text(self, text: str, chunk_size: int = None, overlap: int = None) -> List[str]:
        """Split text into overlapping chunks."""
        if chunk_size is None:
            chunk_size = settings.chunk_size
        if overlap is None:
            overlap = settings.chunk_overlap

        chars = list(text)
        chunks = []

        start = 0
        while start < len(chars):
            end = start + chunk_size
            chunk = "".join(chars[start:end])
            chunks.append(chunk)
            start += chunk_size - overlap

        return chunks


embedding_service = EmbeddingService()