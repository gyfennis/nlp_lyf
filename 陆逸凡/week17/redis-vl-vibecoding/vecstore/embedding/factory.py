"""Factory and registry for embedding providers."""

from typing import Any, Dict, Optional, Type

from vecstore.embedding.base import EmbeddingProvider


class EmbeddingProviderFactory:
    """Registry and factory for creating :class:`EmbeddingProvider` instances.

    Usage::

        provider = EmbeddingProviderFactory.create(
            provider="openai",
            model_name="text-embedding-3-small",
        )

    Custom providers can be registered::

        EmbeddingProviderFactory.register("my_provider", MyProvider)
    """

    _registry: Dict[str, Type[EmbeddingProvider]] = {}

    @classmethod
    def register(cls, name: str, provider_cls: Type[EmbeddingProvider]) -> None:
        """Register a custom provider class.

        Args:
            name: Short name used to look up the provider.
            provider_cls: The provider class (must subclass :class:`EmbeddingProvider`).
        """
        cls._registry[name] = provider_cls

    @classmethod
    def create(
        cls,
        provider: str = "openai",
        model_name: Optional[str] = None,
        **kwargs: Any,
    ) -> EmbeddingProvider:
        """Create an embedding provider instance.

        Args:
            provider: Provider type name (``openai``, ``sentence_transformers``,
                or a custom registered name).
            model_name: Model name override. Falls back to the provider's default.
            **kwargs: Additional arguments passed to the provider constructor.

        Returns:
            An :class:`EmbeddingProvider` instance.

        Raises:
            ValueError: If the provider type is unknown.
            ImportError: If the required optional dependency is not installed.
        """
        if provider == "openai":
            try:
                from vecstore.embedding.openai_provider import OpenAIEmbeddingProvider
            except ImportError as err:
                raise ImportError(
                    "OpenAI provider requires 'openai' package. "
                    "Install it with: pip install vecstore[openai]"
                ) from err
            return OpenAIEmbeddingProvider(
                model=model_name or "text-embedding-3-small",
                **kwargs,
            )

        if provider == "sentence_transformers":
            try:
                from vecstore.embedding.sentence_provider import SentenceTransformerProvider
            except ImportError as err:
                raise ImportError(
                    "SentenceTransformers provider requires 'sentence-transformers' package. "
                    "Install it with: pip install vecstore[sentence]"
                ) from err
            return SentenceTransformerProvider(
                model_name=model_name or "all-MiniLM-L6-v2",
            )

        if provider in cls._registry:
            provider_cls = cls._registry[provider]
            resolved_kwargs: dict = {}
            if model_name is not None:
                resolved_kwargs["model_name"] = model_name
            resolved_kwargs.update(kwargs)
            return provider_cls(**resolved_kwargs)

        raise ValueError(
            f"Unknown embedding provider: '{provider}'. "
            f"Available: openai, sentence_transformers, {list(cls._registry.keys())}"
        )


# Register default providers — None-safe lazy registration
def _register_defaults() -> None:
    try:
        from vecstore.embedding.openai_provider import OpenAIEmbeddingProvider
        EmbeddingProviderFactory.register("openai", OpenAIEmbeddingProvider)
    except ImportError:
        pass

    try:
        from vecstore.embedding.sentence_provider import SentenceTransformerProvider
        EmbeddingProviderFactory.register("sentence_transformers", SentenceTransformerProvider)
    except ImportError:
        pass


_register_defaults()
