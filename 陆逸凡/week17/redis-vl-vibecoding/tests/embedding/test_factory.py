"""Tests for EmbeddingProviderFactory."""

from unittest.mock import patch

import pytest

from vecstore.embedding.base import EmbeddingProvider
from vecstore.embedding.factory import EmbeddingProviderFactory
from vecstore.embedding.openai_provider import OpenAIEmbeddingProvider
from vecstore.embedding.sentence_provider import SentenceTransformerProvider


class TestEmbeddingProviderFactory:
    """Test suite for EmbeddingProviderFactory."""

    @patch("vecstore.embedding.openai_provider.AsyncOpenAI")
    def test_create_openai(self, mock_openai):
        """Factory should create an OpenAI provider."""
        provider = EmbeddingProviderFactory.create(provider="openai")
        assert isinstance(provider, OpenAIEmbeddingProvider)
        assert provider.model_name == "text-embedding-3-small"

    @patch("vecstore.embedding.openai_provider.AsyncOpenAI")
    def test_create_openai_with_custom_model(self, mock_openai):
        """Factory should respect model_name override."""
        provider = EmbeddingProviderFactory.create(
            provider="openai",
            model_name="text-embedding-3-large",
        )
        assert isinstance(provider, OpenAIEmbeddingProvider)
        assert provider.model_name == "text-embedding-3-large"

    def test_create_sentence_transformers(self):
        """Factory should create a SentenceTransformer provider."""
        provider = EmbeddingProviderFactory.create(provider="sentence_transformers")
        assert isinstance(provider, SentenceTransformerProvider)

    def test_create_unknown_provider_raises(self):
        """Unknown provider names should raise ValueError."""
        with pytest.raises(ValueError):
            EmbeddingProviderFactory.create(provider="nonexistent")

    def test_register_custom_provider(self):
        """Custom providers should be creatable after registration."""

        class FakeProvider(EmbeddingProvider):
            @property
            def dimensions(self) -> int:
                return 128

            @property
            def model_name(self) -> str:
                return "fake"

            async def embed(self, texts):
                import numpy as np
                return [np.zeros(self.dimensions, dtype=np.float32) for _ in texts]

        EmbeddingProviderFactory.register("fake", FakeProvider)
        provider = EmbeddingProviderFactory.create(provider="fake")
        assert isinstance(provider, FakeProvider)
