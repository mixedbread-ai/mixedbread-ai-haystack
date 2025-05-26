import os
from typing import Optional


class TestConfig:
    """Configuration class for integration tests."""

    API_KEY = os.environ.get("MXBAI_API_KEY")

    OFFICIAL_API_BASE_URL = "https://api.mixedbread.com"
    CUSTOM_BASE_URL = os.environ.get("MXBAI_CUSTOM_BASE_URL")

    TEST_TIMEOUT = 30.0
    TEST_MAX_RETRIES = 2

    DEFAULT_EMBEDDING_MODEL = "mixedbread-ai/mxbai-embed-large-v1"
    DEFAULT_RERANKING_MODEL = "mixedbread-ai/mxbai-rerank-large-v2"

    @classmethod
    def get_base_url(cls) -> Optional[str]:
        """Get the base URL to use for tests."""
        return cls.CUSTOM_BASE_URL or cls.OFFICIAL_API_BASE_URL

    @classmethod
    def has_api_key(cls) -> bool:
        """Check if API key is available."""
        return bool(cls.API_KEY)

    @classmethod
    def get_test_embedder_config(cls) -> dict:
        """Get default configuration for embedder tests."""
        config = {
            "timeout": cls.TEST_TIMEOUT,
            "max_retries": cls.TEST_MAX_RETRIES,
            "model": cls.DEFAULT_EMBEDDING_MODEL,
        }

        base_url = cls.get_base_url()
        if base_url:
            config["base_url"] = base_url

        return config

    @classmethod
    def get_test_reranker_config(cls) -> dict:
        """Get default configuration for reranker tests."""
        config = {
            "timeout": cls.TEST_TIMEOUT,
            "max_retries": cls.TEST_MAX_RETRIES,
            "model": cls.DEFAULT_RERANKING_MODEL,
        }

        base_url = cls.get_base_url()
        if base_url:
            config["base_url"] = base_url

        return config
