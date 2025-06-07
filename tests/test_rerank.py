import pytest
import asyncio
from haystack import Document
from haystack.utils.auth import Secret
from mixedbread_ai_haystack.rerankers import MixedbreadReranker
from .test_config import TestConfig

DEFAULT_VALUES = {
    "base_url": None,
    "timeout": 60.0,
    "max_retries": 2,
    "model": "mixedbread-ai/mxbai-rerank-large-v1",
    "top_k": 10,
    "return_input": False,
}


class TestMixedbreadReranker:
    def test_init_default(self, monkeypatch):
        """
        Test default initialization parameters for MixedbreadReranker.
        """
        monkeypatch.setenv("MXBAI_API_KEY", "test-api-key")
        component = MixedbreadReranker()
        assert component.model == DEFAULT_VALUES["model"]
        assert component.top_k == DEFAULT_VALUES["top_k"]
        assert component.api_key == Secret.from_env_var("MXBAI_API_KEY")
        assert component.return_input == DEFAULT_VALUES["return_input"]
        assert component.base_url == DEFAULT_VALUES["base_url"]
        assert component.timeout == DEFAULT_VALUES["timeout"]
        assert component.max_retries == DEFAULT_VALUES["max_retries"]

    def test_init_with_parameters(self, monkeypatch):
        """
        Test custom initialization parameters for MixedbreadReranker.
        """
        monkeypatch.setenv("MXBAI_API_KEY", "test-api-key")
        component = MixedbreadReranker(
            model="custom-model",
            top_k=5,
            return_input=True,
            base_url="http://custom-url.com",
            timeout=30.0,
            max_retries=5,
        )
        assert component.model == "custom-model"
        assert component.top_k == 5
        assert component.api_key == Secret.from_env_var("MXBAI_API_KEY")
        assert component.return_input is True
        assert component.base_url == "http://custom-url.com"
        assert component.timeout == 30.0
        assert component.max_retries == 5

    def test_init_fail_wo_api_key(self, monkeypatch):
        """
        Test that initialization fails when no API key is provided.
        """
        monkeypatch.delenv("MXBAI_API_KEY", raising=False)
        with pytest.raises(
            ValueError,
            match="None of the following authentication environment variables are set",
        ):
            MixedbreadReranker()

    def test_to_dict_default(self, monkeypatch):
        """
        Test serialization of this component to a dictionary, using default initialization parameters.
        """
        monkeypatch.setenv("MXBAI_API_KEY", "test-api-key")
        component = MixedbreadReranker()
        data = component.to_dict()
        assert data == {
            "type": "mixedbread_ai_haystack.rerankers.reranker.MixedbreadReranker",
            "init_parameters": {
                "model": DEFAULT_VALUES["model"],
                "api_key": Secret.from_env_var("MXBAI_API_KEY").to_dict(),
                "top_k": DEFAULT_VALUES["top_k"],
                "return_input": DEFAULT_VALUES["return_input"],
                "base_url": DEFAULT_VALUES["base_url"],
                "timeout": DEFAULT_VALUES["timeout"],
                "max_retries": DEFAULT_VALUES["max_retries"],
            },
        }

    def test_to_dict_with_custom_init_parameters(self, monkeypatch):
        """
        Test serialization of this component to a dictionary, using custom initialization parameters.
        """
        monkeypatch.setenv("MXBAI_API_KEY", "test-api-key")
        component = MixedbreadReranker(
            model="custom-model",
            top_k=2,
            return_input=True,
            base_url="http://custom-url.com",
            timeout=30.0,
            max_retries=5,
        )
        data = component.to_dict()
        assert data == {
            "type": "mixedbread_ai_haystack.rerankers.reranker.MixedbreadReranker",
            "init_parameters": {
                "model": "custom-model",
                "api_key": Secret.from_env_var("MXBAI_API_KEY").to_dict(),
                "top_k": 2,
                "return_input": True,
                "base_url": "http://custom-url.com",
                "timeout": 30.0,
                "max_retries": 5,
            },
        }

    def test_run_wrong_input_format(self):
        """
        Test for checking incorrect input when reranking documents.
        """
        ranker = MixedbreadReranker(api_key=Secret.from_token("fake-api-key"))

        with pytest.raises(
            TypeError, match="Input must be a list of Haystack Documents"
        ):
            ranker.run(documents="not a list", query="test query")

    def test_run_empty_documents(self):
        """
        Test reranking with empty document list.
        """
        ranker = MixedbreadReranker(api_key=Secret.from_token("fake-api-key"))
        result = ranker.run(documents=[], query="test query")

        assert result["documents"] == []
        assert "meta" in result
        assert result["meta"]["top_k"] == 0

    @pytest.mark.skipif(
        not TestConfig.has_api_key(),
        reason="Export an env var called MXBAI_API_KEY containing the Mixedbread API key to run this test.",
    )
    @pytest.mark.integration
    def test_integration_basic_reranking(self):
        """
        Test basic reranking with real API call.
        """
        reranker_config = TestConfig.get_test_reranker_config()
        reranker = MixedbreadReranker(top_k=2, **reranker_config)

        documents = [
            Document(id="doc1", content="Paris is the capital of France"),
            Document(id="doc2", content="Berlin is the capital of Germany"),
            Document(id="doc3", content="Madrid is the capital of Spain"),
        ]

        result = reranker.run(
            documents=documents, query="What is the capital of Germany?"
        )

        assert isinstance(result, dict)
        assert isinstance(result["documents"], list)
        assert len(result["documents"]) <= 2  # top_k = 2
        assert all(isinstance(doc, Document) for doc in result["documents"])
        assert all(
            hasattr(doc, "meta") and "rerank_score" in doc.meta
            for doc in result["documents"]
        )
        assert "meta" in result

    # Async Tests
    
    @pytest.mark.asyncio
    async def test_run_async_empty_documents(self):
        """
        Test async run method with empty document list.
        """
        reranker = MixedbreadReranker(api_key=Secret.from_token("fake-api-key"))
        result = await reranker.run_async(documents=[], query="test query")

        assert result["documents"] == []
        assert "meta" in result

    @pytest.mark.asyncio
    async def test_run_async_wrong_input_format(self):
        """
        Test async run method with incorrect input format.
        """
        reranker = MixedbreadReranker(api_key=Secret.from_token("fake-api-key"))

        with pytest.raises(
            TypeError, match="Input must be a list of Haystack Documents"
        ):
            await reranker.run_async(documents="not a list", query="test query")

    @pytest.mark.asyncio
    async def test_run_async_empty_query(self):
        """
        Test async run method with empty query.
        """
        reranker = MixedbreadReranker(api_key=Secret.from_token("fake-api-key"))
        documents = [
            Document(content="First document"),
            Document(content="Second document"),
        ]
        
        result = await reranker.run_async(documents=documents, query="")
        
        # Should return original documents when query is empty
        assert len(result["documents"]) == len(documents)
        assert "meta" in result

    @pytest.mark.skipif(
        not TestConfig.has_api_key(),
        reason="Export an env var called MXBAI_API_KEY containing the Mixedbread API key to run this test.",
    )
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_integration_async_basic_reranking(self):
        """
        Test basic async reranking with real API call.
        """
        reranker_config = TestConfig.get_test_reranker_config()
        reranker = MixedbreadReranker(top_k=2, **reranker_config)

        documents = [
            Document(id="doc1", content="Paris is the capital of France"),
            Document(id="doc2", content="Berlin is the capital of Germany"),
            Document(id="doc3", content="Madrid is the capital of Spain"),
        ]

        result = await reranker.run_async(
            documents=documents, query="What is the capital of Germany?"
        )

        assert isinstance(result, dict)
        assert isinstance(result["documents"], list)
        assert len(result["documents"]) <= 2  # top_k = 2
        assert all(isinstance(doc, Document) for doc in result["documents"])
        assert all(
            hasattr(doc, "meta") and "rerank_score" in doc.meta
            for doc in result["documents"]
        )
        assert "meta" in result

    @pytest.mark.skipif(
        not TestConfig.has_api_key(),
        reason="Export an env var called MXBAI_API_KEY containing the Mixedbread API key to run this test.",
    )
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_integration_async_concurrent_reranking(self):
        """
        Test concurrent async reranking for performance.
        """
        reranker_config = TestConfig.get_test_reranker_config()
        reranker = MixedbreadReranker(top_k=2, **reranker_config)

        documents = [
            Document(content="Python is a programming language"),
            Document(content="JavaScript is used for web development"),
            Document(content="Machine learning is transforming industries"),
            Document(content="Artificial intelligence has many applications"),
        ]

        queries = [
            "What programming languages are popular?",
            "Tell me about machine learning",
            "How is AI being used today?"
        ]

        # Process all queries concurrently
        tasks = [
            reranker.run_async(documents=documents, query=query) 
            for query in queries
        ]
        results = await asyncio.gather(*tasks)

        assert len(results) == len(queries)
        for result in results:
            assert isinstance(result["documents"], list)
            assert len(result["documents"]) <= 2  # top_k = 2
            assert all(
                "rerank_score" in doc.meta for doc in result["documents"]
            )

    @pytest.mark.skipif(
        not TestConfig.has_api_key(),
        reason="Export an env var called MXBAI_API_KEY containing the Mixedbread API key to run this test.",
    )
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_integration_async_different_top_k_values(self):
        """
        Test async reranking with different top_k values.
        """
        reranker_config = TestConfig.get_test_reranker_config()
        
        documents = [
            Document(content="Document about cats"),
            Document(content="Document about dogs"),
            Document(content="Document about birds"),
            Document(content="Document about fish"),
        ]
        
        query = "Tell me about pets"

        # Test different top_k values concurrently
        reranker_k1 = MixedbreadReranker(top_k=1, **reranker_config)
        reranker_k3 = MixedbreadReranker(top_k=3, **reranker_config)
        
        tasks = [
            reranker_k1.run_async(documents=documents, query=query),
            reranker_k3.run_async(documents=documents, query=query),
        ]
        results = await asyncio.gather(*tasks)

        result_k1, result_k3 = results
        assert len(result_k1["documents"]) == 1
        assert len(result_k3["documents"]) == 3

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
        reranker_config = TestConfig.get_test_reranker_config()
        reranker = MixedbreadReranker(top_k=2, **reranker_config)

        documents = [
            Document(content="Document about Python programming"),
            Document(content="Document about JavaScript development"),
            Document(content="Document about data science"),
        ]
        query = "Programming languages"

        # Get both sync and async results
        sync_result = reranker.run(documents=documents, query=query)
        async_result = await reranker.run_async(documents=documents, query=query)

        # Both should have the same structure
        assert len(sync_result["documents"]) == len(async_result["documents"])
        assert "meta" in sync_result
        assert "meta" in async_result
        
        # Meta should have same model
        assert sync_result["meta"]["model"] == async_result["meta"]["model"]
        
        # All documents should have rerank scores
        for sync_doc, async_doc in zip(sync_result["documents"], async_result["documents"]):
            assert "rerank_score" in sync_doc.meta
            assert "rerank_score" in async_doc.meta

    @pytest.mark.asyncio
    async def test_run_async_behavior_matches_sync(self):
        """
        Test that async method behavior matches sync method for edge cases.
        """
        reranker = MixedbreadReranker(api_key=Secret.from_token("fake-api-key"))
        
        # Test with empty documents
        sync_result = reranker.run(documents=[], query="test")
        async_result = await reranker.run_async(documents=[], query="test")
        
        assert sync_result["documents"] == async_result["documents"]
        assert sync_result["meta"]["model"] == async_result["meta"]["model"]
        
        # Test with empty query
        documents = [Document(content="test document")]
        sync_result = reranker.run(documents=documents, query="")
        async_result = await reranker.run_async(documents=documents, query="")
        
        assert len(sync_result["documents"]) == len(async_result["documents"])
