import pytest
import asyncio
from haystack.utils import Secret
from mixedbread_ai_haystack.embedders import MixedbreadTextEmbedder
from .test_config import TestConfig

DEFAULT_VALUES = {
    "base_url": None,
    "timeout": 60.0,
    "max_retries": 2,
    "model": "mixedbread-ai/mxbai-embed-large-v1",
    "normalized": True,
    "encoding_format": "float",
    "dimensions": None,
    "prompt": None,
}


class TestMixedbreadTextEmbedder:
    def test_init_default(self, monkeypatch):
        """
        Test default initialization parameters for MixedbreadTextEmbedder.
        """
        monkeypatch.setenv("MXBAI_API_KEY", "fake-api-key")
        embedder = MixedbreadTextEmbedder()

        assert embedder.api_key == Secret.from_env_var("MXBAI_API_KEY")
        assert embedder.base_url == DEFAULT_VALUES["base_url"]
        assert embedder.timeout == DEFAULT_VALUES["timeout"]
        assert embedder.max_retries == DEFAULT_VALUES["max_retries"]

        assert embedder.model == DEFAULT_VALUES["model"]
        assert embedder.normalized == DEFAULT_VALUES["normalized"]
        assert embedder.encoding_format.value == DEFAULT_VALUES["encoding_format"]

        assert embedder.dimensions == DEFAULT_VALUES["dimensions"]
        assert embedder.prompt == DEFAULT_VALUES["prompt"]

    def test_init_with_parameters(self):
        """
        Test custom initialization parameters for MixedbreadTextEmbedder.
        """
        embedder = MixedbreadTextEmbedder(
            api_key=Secret.from_token("test-api-key"),
            base_url="http://example.com",
            timeout=50.0,
            max_retries=10,
            model="model",
            normalized=False,
            encoding_format="binary",
            dimensions=500,
            prompt="prompt",
        )

        assert embedder.api_key == Secret.from_token("test-api-key")
        assert embedder.base_url == "http://example.com"
        assert embedder.timeout == 50.0
        assert embedder.max_retries == 10

        assert embedder.model == "model"
        assert not embedder.normalized
        assert embedder.encoding_format.value == "binary"
        assert embedder.dimensions == 500
        assert embedder.prompt == "prompt"

    def test_init_fail_wo_api_key(self, monkeypatch):
        """
        Test that initialization fails when no API key is provided.
        """
        monkeypatch.delenv("MXBAI_API_KEY", raising=False)
        with pytest.raises(
            ValueError,
            match="None of the following authentication environment variables are set",
        ):
            MixedbreadTextEmbedder()

    def test_to_dict(self, monkeypatch):
        """
        Test serialization of this component to a dictionary, using default initialization parameters.
        """
        monkeypatch.setenv("MXBAI_API_KEY", "fake-api-key")
        component = MixedbreadTextEmbedder()
        data = component.to_dict()
        assert data == {
            "type": "mixedbread_ai_haystack.embedders.text_embedder.MixedbreadTextEmbedder",
            "init_parameters": {
                **DEFAULT_VALUES,
                "api_key": Secret.from_env_var("MXBAI_API_KEY").to_dict(),
            },
        }

    def test_to_dict_with_custom_init_parameters(self, monkeypatch):
        """
        Test serialization of this component to a dictionary, using custom initialization parameters.
        """
        monkeypatch.setenv("MXBAI_API_KEY", "fake-api-key")
        component = MixedbreadTextEmbedder(
            base_url="http://example.com",
            timeout=50.0,
            max_retries=10,
            model="model",
            normalized=False,
            encoding_format="binary",
            dimensions=500,
            prompt="prompt",
        )
        data = component.to_dict()
        assert data == {
            "type": "mixedbread_ai_haystack.embedders.text_embedder.MixedbreadTextEmbedder",
            "init_parameters": {
                "api_key": Secret.from_env_var("MXBAI_API_KEY").to_dict(),
                "base_url": "http://example.com",
                "timeout": 50.0,
                "max_retries": 10,
                "model": "model",
                "normalized": False,
                "encoding_format": "binary",
                "dimensions": 500,
                "prompt": "prompt",
            },
        }

    @pytest.mark.skipif(
        not TestConfig.has_api_key(),
        reason="Export an env var called MXBAI_API_KEY containing the Mixedbread API key to run this test.",
    )
    @pytest.mark.integration
    def test_integration_basic_text_embedding(self):
        """
        Test basic text embedding with real API call.
        """
        embedder_config = TestConfig.get_test_embedder_config()
        embedder = MixedbreadTextEmbedder(**embedder_config)

        result = embedder.run(text="The food was delicious")

        assert isinstance(result["embedding"], list)
        assert all(isinstance(x, float) for x in result["embedding"])
        assert len(result["embedding"]) > 0
        assert "meta" in result

    # Async Tests
    
    @pytest.mark.asyncio
    async def test_run_async_empty_text(self, monkeypatch):
        """
        Test async run method with empty text returns empty response.
        """
        monkeypatch.setenv("MXBAI_API_KEY", "fake-api-key")
        embedder = MixedbreadTextEmbedder()
        
        result = await embedder.run_async(text="")
        
        assert "embedding" in result
        assert result["embedding"] == []
        assert "meta" in result
        assert result["meta"]["model"] == "mixedbread-ai/mxbai-embed-large-v1"

    @pytest.mark.asyncio
    async def test_run_async_whitespace_text(self, monkeypatch):
        """
        Test async run method with whitespace-only text returns empty response.
        """
        monkeypatch.setenv("MXBAI_API_KEY", "fake-api-key")
        embedder = MixedbreadTextEmbedder()
        
        result = await embedder.run_async(text="   \n\t  ")
        
        assert "embedding" in result
        assert result["embedding"] == []
        assert "meta" in result

    @pytest.mark.skipif(
        not TestConfig.has_api_key(),
        reason="Export an env var called MXBAI_API_KEY containing the Mixedbread API key to run this test.",
    )
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_integration_async_basic_text_embedding(self):
        """
        Test basic async text embedding with real API call.
        """
        embedder_config = TestConfig.get_test_embedder_config()
        embedder = MixedbreadTextEmbedder(**embedder_config)

        result = await embedder.run_async(text="The food was delicious")

        assert isinstance(result["embedding"], list)
        assert all(isinstance(x, float) for x in result["embedding"])
        assert len(result["embedding"]) > 0
        assert "meta" in result
        assert result["meta"]["model"] == embedder_config["model"]

    @pytest.mark.skipif(
        not TestConfig.has_api_key(),
        reason="Export an env var called MXBAI_API_KEY containing the Mixedbread API key to run this test.",
    )
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_integration_async_text_embedding_with_prompt(self):
        """
        Test async text embedding with custom prompt.
        """
        embedder_config = TestConfig.get_test_embedder_config()
        embedder = MixedbreadTextEmbedder(**embedder_config)

        result = await embedder.run_async(
            text="The food was delicious",
            prompt="Represent this text for semantic search:"
        )

        assert isinstance(result["embedding"], list)
        assert all(isinstance(x, float) for x in result["embedding"])
        assert len(result["embedding"]) > 0
        assert "meta" in result

    @pytest.mark.skipif(
        not TestConfig.has_api_key(),
        reason="Export an env var called MXBAI_API_KEY containing the Mixedbread API key to run this test.",
    )
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_integration_async_concurrent_text_embeddings(self):
        """
        Test concurrent async text embeddings for performance.
        """
        embedder_config = TestConfig.get_test_embedder_config()
        embedder = MixedbreadTextEmbedder(**embedder_config)

        texts = [
            "The weather is nice today",
            "I love machine learning",
            "Python is a great programming language",
            "Async processing is efficient"
        ]

        # Process all texts concurrently
        tasks = [embedder.run_async(text=text) for text in texts]
        results = await asyncio.gather(*tasks)

        assert len(results) == len(texts)
        for result in results:
            assert isinstance(result["embedding"], list)
            assert all(isinstance(x, float) for x in result["embedding"])
            assert len(result["embedding"]) > 0
            assert "meta" in result

    @pytest.mark.skipif(
        not TestConfig.has_api_key(),
        reason="Export an env var called MXBAI_API_KEY containing the Mixedbread API key to run this test.",
    )
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_integration_async_sync_consistency(self):
        """
        Test that async and sync methods return consistent results.
        """
        embedder_config = TestConfig.get_test_embedder_config()
        embedder = MixedbreadTextEmbedder(**embedder_config)

        text = "Test consistency between sync and async"

        # Get both sync and async results
        sync_result = embedder.run(text=text)
        async_result = await embedder.run_async(text=text)

        # Both should have the same structure
        assert "embedding" in sync_result
        assert "embedding" in async_result
        assert "meta" in sync_result
        assert "meta" in async_result
        
        # Embeddings should have same length
        assert len(sync_result["embedding"]) == len(async_result["embedding"])
        
        # Meta should have same model
        assert sync_result["meta"]["model"] == async_result["meta"]["model"]

    @pytest.mark.asyncio
    async def test_run_async_behavior_matches_sync(self, monkeypatch):
        """
        Test that async method behavior matches sync method for edge cases.
        """
        monkeypatch.setenv("MXBAI_API_KEY", "fake-api-key")
        embedder = MixedbreadTextEmbedder()
        
        # Test with empty string
        sync_result = embedder.run(text="")
        async_result = await embedder.run_async(text="")
        
        assert sync_result["embedding"] == async_result["embedding"]
        assert sync_result["meta"]["model"] == async_result["meta"]["model"]
        
        # Test with whitespace
        sync_result = embedder.run(text="   ")
        async_result = await embedder.run_async(text="   ")
        
        assert sync_result["embedding"] == async_result["embedding"]
