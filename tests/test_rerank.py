import pytest
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
    "rank_fields": [],
    "return_input": False,
    "rewrite_query": False,
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
        assert component.rank_fields == DEFAULT_VALUES["rank_fields"]
        assert component.return_input == DEFAULT_VALUES["return_input"]
        assert component.rewrite_query == DEFAULT_VALUES["rewrite_query"]
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
            rank_fields=["meta_field_1", "meta_field_2"],
            return_input=True,
            rewrite_query=True,
            base_url="http://custom-url.com",
            timeout=30.0,
            max_retries=5,
        )
        assert component.model == "custom-model"
        assert component.top_k == 5
        assert component.api_key == Secret.from_env_var("MXBAI_API_KEY")
        assert component.rank_fields == ["meta_field_1", "meta_field_2"]
        assert component.return_input is True
        assert component.rewrite_query is True
        assert component.base_url == "http://custom-url.com"
        assert component.timeout == 30.0
        assert component.max_retries == 5

    def test_init_fail_wo_api_key(self, monkeypatch):
        """
        Test that initialization fails when no API key is provided.
        """
        monkeypatch.delenv("MXBAI_API_KEY", raising=False)
        with pytest.raises(ValueError, match="None of the following authentication environment variables are set"):
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
                "rank_fields": DEFAULT_VALUES["rank_fields"],
                "return_input": DEFAULT_VALUES["return_input"],
                "rewrite_query": DEFAULT_VALUES["rewrite_query"],
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
            rank_fields=["meta_field_1", "meta_field_2"],
            return_input=True,
            rewrite_query=True,
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
                "rank_fields": ["meta_field_1", "meta_field_2"],
                "return_input": True,
                "rewrite_query": True,
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

        with pytest.raises(TypeError, match="Input must be a list of Haystack Documents"):
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

        result = reranker.run(documents=documents, query="What is the capital of Germany?")

        assert isinstance(result, dict)
        assert isinstance(result["documents"], list)
        assert len(result["documents"]) <= 2  # top_k = 2
        assert all(isinstance(doc, Document) for doc in result["documents"])
        assert all(hasattr(doc, "meta") and "rerank_score" in doc.meta for doc in result["documents"])
        assert "meta" in result
