import pytest
import asyncio
from haystack import Document
from haystack.utils import Secret
from mixedbread_ai_haystack.embedders import MixedbreadDocumentEmbedder
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


class TestMixedbreadDocumentEmbedder:
    def test_init_default(self, monkeypatch):
        """
        Test default initialization parameters for MixedbreadDocumentEmbedder.
        """
        monkeypatch.setenv("MXBAI_API_KEY", "fake-api-key")
        embedder = MixedbreadDocumentEmbedder()

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
        Test custom initialization parameters for MixedbreadDocumentEmbedder.
        """
        embedder = MixedbreadDocumentEmbedder(
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
            MixedbreadDocumentEmbedder()

    def test_to_dict(self, monkeypatch):
        """
        Test serialization of this component to a dictionary, using default initialization parameters.
        """
        monkeypatch.setenv("MXBAI_API_KEY", "fake-api-key")
        component = MixedbreadDocumentEmbedder()
        data = component.to_dict()
        assert data == {
            "type": "mixedbread_ai_haystack.embedders.document_embedder.MixedbreadDocumentEmbedder",
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
        component = MixedbreadDocumentEmbedder(
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
            "type": "mixedbread_ai_haystack.embedders.document_embedder.MixedbreadDocumentEmbedder",
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

    def test_run_wrong_input_format(self):
        """
        Test for checking incorrect input when creating embeddings.
        """
        embedder = MixedbreadDocumentEmbedder(api_key=Secret.from_token("fake-api-key"))

        with pytest.raises(
            TypeError, match="Input must be a list of Haystack Documents"
        ):
            embedder.run(documents="not a list")

    def test_run_empty_documents(self):
        """
        Test embedding with empty document list.
        """
        embedder = MixedbreadDocumentEmbedder(api_key=Secret.from_token("fake-api-key"))
        result = embedder.run(documents=[])

        assert result["documents"] == []
        assert "meta" in result

    @pytest.mark.skipif(
        not TestConfig.has_api_key(),
        reason="Export an env var called MXBAI_API_KEY containing the Mixedbread API key to run this test.",
    )
    @pytest.mark.integration
    def test_integration_basic_embedding(self):
        """
        Test basic document embedding with real API call.
        """
        docs = [
            Document(content="The Eiffel Tower is in Paris", meta={"topic": "Travel"}),
            Document(
                content="Machine learning is transforming industries",
                meta={"topic": "AI"},
            ),
        ]

        embedder_config = TestConfig.get_test_embedder_config()
        embedder = MixedbreadDocumentEmbedder(**embedder_config)

        result = embedder.run(documents=docs)
        documents_with_embeddings = result["documents"]

        assert isinstance(documents_with_embeddings, list)
        assert len(documents_with_embeddings) == len(docs)
        for doc in documents_with_embeddings:
            assert isinstance(doc, Document)
            assert isinstance(doc.embedding, list)
            assert len(doc.embedding) > 0
            assert all(isinstance(x, float) for x in doc.embedding)

        assert "meta" in result

    # Async Tests
    
    @pytest.mark.asyncio
    async def test_run_async_empty_documents(self):
        """
        Test async run method with empty document list.
        """
        embedder = MixedbreadDocumentEmbedder(api_key=Secret.from_token("fake-api-key"))
        result = await embedder.run_async(documents=[])

        assert result["documents"] == []
        assert "meta" in result

    @pytest.mark.asyncio
    async def test_run_async_wrong_input_format(self):
        """
        Test async run method with incorrect input format.
        """
        embedder = MixedbreadDocumentEmbedder(api_key=Secret.from_token("fake-api-key"))

        with pytest.raises(
            TypeError, match="Input must be a list of Haystack Documents"
        ):
            await embedder.run_async(documents="not a list")

    @pytest.mark.skipif(
        not TestConfig.has_api_key(),
        reason="Export an env var called MXBAI_API_KEY containing the Mixedbread API key to run this test.",
    )
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_integration_async_basic_embedding(self):
        """
        Test basic async document embedding with real API call.
        """
        docs = [
            Document(content="The Eiffel Tower is in Paris", meta={"topic": "Travel"}),
            Document(
                content="Machine learning is transforming industries",
                meta={"topic": "AI"},
            ),
        ]

        embedder_config = TestConfig.get_test_embedder_config()
        embedder = MixedbreadDocumentEmbedder(**embedder_config)

        result = await embedder.run_async(documents=docs)
        documents_with_embeddings = result["documents"]

        assert isinstance(documents_with_embeddings, list)
        assert len(documents_with_embeddings) == len(docs)
        for doc in documents_with_embeddings:
            assert isinstance(doc, Document)
            assert isinstance(doc.embedding, list)
            assert len(doc.embedding) > 0
            assert all(isinstance(x, float) for x in doc.embedding)

        assert "meta" in result

    @pytest.mark.skipif(
        not TestConfig.has_api_key(),
        reason="Export an env var called MXBAI_API_KEY containing the Mixedbread API key to run this test.",
    )
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_integration_async_documents_with_prompt(self):
        """
        Test async document embedding with custom prompt.
        """
        docs = [
            Document(content="Python is a programming language"),
            Document(content="JavaScript is used for web development"),
        ]

        embedder_config = TestConfig.get_test_embedder_config()
        embedder = MixedbreadDocumentEmbedder(**embedder_config)

        result = await embedder.run_async(
            documents=docs,
            prompt="Represent this document for semantic search:"
        )

        documents_with_embeddings = result["documents"]
        assert len(documents_with_embeddings) == len(docs)
        for doc in documents_with_embeddings:
            assert isinstance(doc.embedding, list)
            assert len(doc.embedding) > 0

    @pytest.mark.skipif(
        not TestConfig.has_api_key(),
        reason="Export an env var called MXBAI_API_KEY containing the Mixedbread API key to run this test.",
    )
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_integration_async_concurrent_batch_processing(self):
        """
        Test concurrent async document embedding for performance.
        """
        embedder_config = TestConfig.get_test_embedder_config()
        embedder = MixedbreadDocumentEmbedder(**embedder_config)

        # Create multiple batches of documents
        batch1 = [
            Document(content="First batch document 1"),
            Document(content="First batch document 2"),
        ]
        batch2 = [
            Document(content="Second batch document 1"),
            Document(content="Second batch document 2"),
        ]
        batch3 = [
            Document(content="Third batch document 1"),
            Document(content="Third batch document 2"),
        ]

        # Process batches concurrently
        tasks = [
            embedder.run_async(documents=batch1),
            embedder.run_async(documents=batch2),
            embedder.run_async(documents=batch3),
        ]
        results = await asyncio.gather(*tasks)

        assert len(results) == 3
        for result in results:
            documents_with_embeddings = result["documents"]
            assert len(documents_with_embeddings) == 2
            for doc in documents_with_embeddings:
                assert isinstance(doc.embedding, list)
                assert len(doc.embedding) > 0

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
        docs = [
            Document(content="Test consistency between sync and async"),
            Document(content="Both methods should work identically"),
        ]

        embedder_config = TestConfig.get_test_embedder_config()
        embedder = MixedbreadDocumentEmbedder(**embedder_config)

        # Get both sync and async results
        sync_result = embedder.run(documents=docs)
        async_result = await embedder.run_async(documents=docs)

        # Both should have the same structure
        assert len(sync_result["documents"]) == len(async_result["documents"])
        assert "meta" in sync_result
        assert "meta" in async_result
        
        # Embeddings should have same length
        for sync_doc, async_doc in zip(sync_result["documents"], async_result["documents"]):
            assert len(sync_doc.embedding) == len(async_doc.embedding)
        
        # Meta should have same model
        assert sync_result["meta"]["model"] == async_result["meta"]["model"]

    @pytest.mark.asyncio
    async def test_run_async_behavior_matches_sync(self):
        """
        Test that async method behavior matches sync method for edge cases.
        """
        embedder = MixedbreadDocumentEmbedder(api_key=Secret.from_token("fake-api-key"))
        
        # Test with empty documents
        sync_result = embedder.run(documents=[])
        async_result = await embedder.run_async(documents=[])
        
        assert sync_result["documents"] == async_result["documents"]
        assert sync_result["meta"]["model"] == async_result["meta"]["model"]
