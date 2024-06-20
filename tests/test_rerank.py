import os

import pytest
from haystack import Document
from haystack.utils.auth import Secret

from mixedbread_ai_haystack.rerankers import MixedbreadAIReranker
from tests.utils import mock_reranking_response

pytestmark = pytest.mark.ranker

DEFAULT_VALUES = {
    "base_url": None,
    "timeout": 60.0,
    "max_retries": 3,
    "use_async_client": False,
    "model": "mixedbread-ai/mxbai-rerank-large-v1",
    "top_k": 10,
    "meta_fields_to_rank": []
}


class TestMixedbreadAIReranker:
    def test_init_default(self, monkeypatch):
        monkeypatch.setenv("MXBAI_API_KEY", "test-api-key")
        component = MixedbreadAIReranker()
        assert component.model == DEFAULT_VALUES["model"]
        assert component.top_k == DEFAULT_VALUES["top_k"]
        assert component.api_key == Secret.from_env_var("MXBAI_API_KEY")
        assert component.meta_fields_to_rank == DEFAULT_VALUES["meta_fields_to_rank"]
        assert component.base_url == DEFAULT_VALUES["base_url"]
        assert component.timeout == DEFAULT_VALUES["timeout"]
        assert component.max_retries == DEFAULT_VALUES["max_retries"]
        assert component.use_async_client == DEFAULT_VALUES["use_async_client"]

    def test_init_fail_wo_api_key(self, monkeypatch):
        monkeypatch.delenv("MXBAI_API_KEY", raising=False)
        with pytest.raises(ValueError):
            MixedbreadAIReranker()

    def test_init_with_parameters(self, monkeypatch):
        monkeypatch.setenv("MXBAI_API_KEY", "test-api-key")
        component = MixedbreadAIReranker(
            model="custom-model",
            top_k=5,
            meta_fields_to_rank=["meta_field_1", "meta_field_2"],
        )
        assert component.model == "custom-model"
        assert component.top_k == 5
        assert component.api_key == Secret.from_env_var("MXBAI_API_KEY")
        assert component.meta_fields_to_rank == ["meta_field_1", "meta_field_2"]

    def test_to_dict_default(self, monkeypatch):
        monkeypatch.setenv("MXBAI_API_KEY", "test-api-key")
        component = MixedbreadAIReranker()
        data = component.to_dict()
        assert data == {
            "type": "mixedbread_ai_haystack.rerankers.reranker.MixedbreadAIReranker",
            "init_parameters": {
                "model": DEFAULT_VALUES["model"],
                "api_key": Secret.from_env_var("MXBAI_API_KEY").to_dict(),
                "top_k": DEFAULT_VALUES["top_k"],
                "meta_fields_to_rank": DEFAULT_VALUES["meta_fields_to_rank"],
                "base_url": DEFAULT_VALUES["base_url"],
                "timeout": DEFAULT_VALUES["timeout"],
                "max_retries": DEFAULT_VALUES["max_retries"],
                "use_async_client": DEFAULT_VALUES["use_async_client"],
            },
        }

    def test_to_dict_with_parameters(self, monkeypatch):
        monkeypatch.setenv("MXBAI_API_KEY", "test-api-key")
        component = MixedbreadAIReranker(
            model="custom-model",
            top_k=2,
            meta_fields_to_rank=["meta_field_1", "meta_field_2"],
        )
        data = component.to_dict()
        assert data == {
            "type": "mixedbread_ai_haystack.rerankers.reranker.MixedbreadAIReranker",
            "init_parameters": {
                "model": "custom-model",
                "api_key": Secret.from_env_var("MXBAI_API_KEY").to_dict(),
                "top_k": 2,
                "meta_fields_to_rank": ["meta_field_1", "meta_field_2"],
                "base_url": DEFAULT_VALUES["base_url"],
                "timeout": DEFAULT_VALUES["timeout"],
                "max_retries": DEFAULT_VALUES["max_retries"],
                "use_async_client": DEFAULT_VALUES["use_async_client"],
            },
        }

    def test_from_dict(self, monkeypatch):
        monkeypatch.setenv("MXBAI_API_KEY", "test-api-key")
        data = {
            "type": "mixedbread_ai_haystack.rerankers.reranker.MixedbreadAIReranker",
            "init_parameters": {
                "model": "custom-model",
                "api_key": Secret.from_env_var("MXBAI_API_KEY").to_dict(),
                "top_k": 2,
                "meta_fields_to_rank": ["meta_field_1", "meta_field_2"],
                "base_url": DEFAULT_VALUES["base_url"],
                "timeout": DEFAULT_VALUES["timeout"],
                "max_retries": DEFAULT_VALUES["max_retries"],
                "use_async_client": DEFAULT_VALUES["use_async_client"],
            },
        }
        component = MixedbreadAIReranker.from_dict(data)
        assert component.model == "custom-model"
        assert component.top_k == 2
        assert component.api_key == Secret.from_env_var("MXBAI_API_KEY")
        assert component.meta_fields_to_rank == ["meta_field_1", "meta_field_2"]

    def test_from_dict_fail_wo_env_var(self, monkeypatch):
        monkeypatch.delenv("MXBAI_API_KEY", raising=False)
        data = {
            "type": "mixedbread_ai_haystack.rerankers.reranker.MixedbreadAIReranker",
            "init_parameters": {
                "model": "custom-model",
                "top_k": 2,
                "meta_fields_to_rank": ["meta_field_1", "meta_field_2"],
            },
        }
        with pytest.raises(ValueError):
            MixedbreadAIReranker.from_dict(data)

    def test_run_documents_provided(self, monkeypatch, mock_reranking_response):  # noqa: ARG002
        monkeypatch.setenv("MXBAI_API_KEY", "test-api-key")
        ranker = MixedbreadAIReranker()
        query = "test query"
        documents = [
            Document(id="abcd", content="doc1", meta={"meta_field": "meta_value_1"}),
            Document(id="efgh", content="doc2", meta={"meta_field": "meta_value_2"}),
        ]
        ranker_results = ranker.run(query, documents, 2)

        assert isinstance(ranker_results, dict)
        reranked_docs = ranker_results["documents"]
        assert reranked_docs == [
            Document(id="abcd", content="doc1", meta={"meta_field": "meta_value_1"}, score=1.0),
            Document(id="efgh", content="doc2", meta={"meta_field": "meta_value_2"}, score=0.9),
        ]

    def test_run_topk_set_in_init(self, monkeypatch, mock_reranking_response):  # noqa: ARG002
        monkeypatch.setenv("MXBAI_API_KEY", "test-api-key")
        ranker = MixedbreadAIReranker(top_k=1)
        query = "test query"
        documents = [
            Document(id="abcd", content="doc1"),
            Document(id="efgh", content="doc2"),
        ]

        ranker_results = ranker.run(query, documents)

        assert isinstance(ranker_results, dict)
        reranked_docs = ranker_results["documents"]
        assert reranked_docs == [
            Document(id="abcd", content="doc1", score=1.0),
        ]

    @pytest.mark.skipif(
        not os.environ.get("MXBAI_API_KEY", None),
        reason="Export an env var called MXBAI_API_KEY containing the Mixedbread AI API key to run this test.",
    )
    @pytest.mark.integration
    def test_live_run(self):
        component = MixedbreadAIReranker()
        documents = [
            Document(id="abcd", content="Paris is in France"),
            Document(id="efgh", content="Berlin is in Germany"),
        ]

        ranker_result = component.run("What is the capital of Germany?", documents, 2)
        expected_documents = [documents[1], documents[0]]
        expected_documents_content = [doc.content for doc in expected_documents]
        result_documents_contents = [doc.content for doc in ranker_result["documents"]]

        assert isinstance(ranker_result, dict)
        assert isinstance(ranker_result["documents"], list)
        assert len(ranker_result["documents"]) == 2
        assert all(isinstance(doc, Document) for doc in ranker_result["documents"])
        assert set(result_documents_contents) == set(expected_documents_content)

    @pytest.mark.skipif(
        not os.environ.get("MXBAI_API_KEY", None),
        reason="Export an env var called MXBAI_API_KEY containing the Mixedbread AI API key to run this test.",
    )
    @pytest.mark.integration
    def test_live_run_topk_greater_than_docs(self):
        component = MixedbreadAIReranker(
            meta_fields_to_rank=["topic"]
        )
        documents = [
            Document(id="abcd", content="Paris is in France", meta={"topic": "France"}),
            Document(id="efgh", content="Berlin is in Germany", meta={"topic": "Germany"}),
        ]

        ranker_result = component.run("What is the capital of Germany?", documents, 5)
        expected_documents = [documents[1], documents[0]]
        expected_documents_content = [doc.content for doc in expected_documents]
        result_documents_contents = [doc.content for doc in ranker_result["documents"]]

        assert isinstance(ranker_result, dict)
        assert isinstance(ranker_result["documents"], list)
        assert len(ranker_result["documents"]) == 2
        assert all(isinstance(doc, Document) for doc in ranker_result["documents"])
        assert set(result_documents_contents) == set(expected_documents_content)



