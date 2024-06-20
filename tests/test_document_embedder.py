import os
from unittest.mock import patch
import pytest
from haystack import Document
from haystack.utils import Secret
from mixedbread_ai import EncodingFormat, TruncationStrategy, ObjectType

from mixedbread_ai.types import Usage
from mixedbread_ai_haystack.embedders import MixedbreadAIDocumentEmbedder
from .utils import mock_embeddings_response

DEFAULT_VALUES = {
    "base_url": None,
    "timeout": 60.0,
    "max_retries": 3,
    "use_async_client": False,
    "model": "mixedbread-ai/mxbai-embed-large-v1",
    "prefix": "",
    "suffix": "",
    "normalized": True,
    "encoding_format": EncodingFormat.FLOAT,
    "truncation_strategy": TruncationStrategy.START,
    "dimensions": None,
    "prompt": None,
    "batch_size": 128,
    "show_progress_bar": True,
    "embedding_separator": "\n",
    "meta_fields_to_embed": [],
}


class TestMixedbreadAIDocumentEmbedder:
    def test_init_default(self, monkeypatch):
        monkeypatch.setenv("MXBAI_API_KEY", "fake-api-key")
        embedder = MixedbreadAIDocumentEmbedder()

        assert embedder.api_key == Secret.from_env_var("MXBAI_API_KEY")
        assert embedder.base_url == DEFAULT_VALUES["base_url"]
        assert embedder.timeout == DEFAULT_VALUES["timeout"]
        assert embedder.max_retries == DEFAULT_VALUES["max_retries"]
        assert embedder.use_async_client == DEFAULT_VALUES["use_async_client"]

        assert embedder.model == DEFAULT_VALUES["model"]
        assert embedder.prefix == DEFAULT_VALUES["prefix"]
        assert embedder.suffix == DEFAULT_VALUES["suffix"]
        assert embedder.normalized == DEFAULT_VALUES["normalized"]
        assert embedder.encoding_format == DEFAULT_VALUES["encoding_format"]
        assert embedder.truncation_strategy == DEFAULT_VALUES["truncation_strategy"]
        assert embedder.dimensions == DEFAULT_VALUES["dimensions"]
        assert embedder.prompt == DEFAULT_VALUES["prompt"]

        assert embedder.batch_size == DEFAULT_VALUES["batch_size"]
        assert embedder.show_progress_bar == DEFAULT_VALUES["show_progress_bar"]
        assert embedder.embedding_separator == DEFAULT_VALUES["embedding_separator"]
        assert embedder.meta_fields_to_embed == DEFAULT_VALUES["meta_fields_to_embed"]

    def test_init_with_parameters(self):
        embedder = MixedbreadAIDocumentEmbedder(
            api_key=Secret.from_token("test-api-key"),
            base_url="http://example.com",
            timeout=50.0,
            max_retries=10,
            use_async_client=True,

            model="model",
            prefix="prefix",
            suffix="suffix",
            normalized=False,
            encoding_format=EncodingFormat.BINARY,
            truncation_strategy=TruncationStrategy.END,
            dimensions=500,
            prompt="prompt",

            batch_size=64,
            show_progress_bar=False,
            embedding_separator=" | ",
            meta_fields_to_embed=["test_field"],
        )

        assert embedder.api_key == Secret.from_token("test-api-key")
        assert embedder.base_url == "http://example.com"
        assert embedder.timeout == 50.0
        assert embedder.max_retries == 10
        assert embedder.use_async_client

        assert embedder.model == "model"
        assert embedder.prefix == "prefix"
        assert embedder.suffix == "suffix"
        assert not embedder.normalized
        assert embedder.encoding_format == EncodingFormat.BINARY
        assert embedder.truncation_strategy == TruncationStrategy.END
        assert embedder.dimensions == 500
        assert embedder.prompt == "prompt"

        assert embedder.batch_size == 64
        assert not embedder.show_progress_bar
        assert embedder.embedding_separator == " | "
        assert embedder.meta_fields_to_embed == ["test_field"]

    def test_init_fail_wo_api_key(self, monkeypatch):
        monkeypatch.delenv("MXBAI_API_KEY", raising=False)
        with pytest.raises(ValueError):
            MixedbreadAIDocumentEmbedder()

    def test_to_dict(self, monkeypatch):
        monkeypatch.setenv("MXBAI_API_KEY", "fake-api-key")
        component = MixedbreadAIDocumentEmbedder()
        data = component.to_dict()
        assert data == {
            "type": "mixedbread_ai_haystack.embedders.document_embedder.MixedbreadAIDocumentEmbedder",
            "init_parameters": {
                **DEFAULT_VALUES,
                "api_key": Secret.from_env_var("MXBAI_API_KEY").to_dict()
            }
        }

    def test_to_dict_with_custom_init_parameters(self, monkeypatch):
        monkeypatch.setenv("MXBAI_API_KEY", "fake-api-key")
        component = MixedbreadAIDocumentEmbedder(
            base_url="http://example.com",
            timeout=50.0,
            max_retries=10,
            use_async_client=True,

            model="model",
            prefix="prefix",
            suffix="suffix",
            normalized=False,
            encoding_format=EncodingFormat.BINARY,
            truncation_strategy=TruncationStrategy.END,
            dimensions=500,
            prompt="prompt",

            batch_size=64,
            show_progress_bar=False,
            meta_fields_to_embed=["test_field"],
            embedding_separator=" | ",
        )
        data = component.to_dict()
        assert data == {
            "type": "mixedbread_ai_haystack.embedders.document_embedder.MixedbreadAIDocumentEmbedder",
            "init_parameters": {
                "api_key": Secret.from_env_var("MXBAI_API_KEY").to_dict(),
                "base_url": "http://example.com",
                "timeout": 50.0,
                "max_retries": 10,
                "use_async_client": True,
                "model": "model",
                "prefix": "prefix",
                "suffix": "suffix",
                "normalized": False,
                "encoding_format": EncodingFormat.BINARY,
                "truncation_strategy": TruncationStrategy.END,
                "dimensions": 500,
                "prompt": "prompt",

                "batch_size": 64,
                "show_progress_bar": False,
                "meta_fields_to_embed": ["test_field"],
                "embedding_separator": " | ",
            },
        }

    def test_prepare_texts_to_embed_w_metadata(self):
        documents = [
            Document(content=f"document number {i}:\ncontent", meta={"meta_field": f"meta_value {i}"}) for i in range(5)
        ]

        embedder = MixedbreadAIDocumentEmbedder(
            api_key=Secret.from_token("fake-api-key"), meta_fields_to_embed=["meta_field"], embedding_separator=" | "
        )

        prepared_texts = embedder.from_docs_to_texts(documents)

        assert prepared_texts == [
            "meta_value 0 | document number 0:\ncontent",
            "meta_value 1 | document number 1:\ncontent",
            "meta_value 2 | document number 2:\ncontent",
            "meta_value 3 | document number 3:\ncontent",
            "meta_value 4 | document number 4:\ncontent",
        ]

    def test_prepare_texts_to_embed_w_suffix(self):
        documents = [Document(content=f"document number {i}") for i in range(5)]

        embedder = MixedbreadAIDocumentEmbedder(api_key=Secret.from_token("fake-api-key"), prefix="my_prefix ", suffix=" my_suffix")

        prepared_texts = embedder.from_docs_to_texts(documents)

        assert prepared_texts == [
            "my_prefix document number 0 my_suffix",
            "my_prefix document number 1 my_suffix",
            "my_prefix document number 2 my_suffix",
            "my_prefix document number 3 my_suffix",
            "my_prefix document number 4 my_suffix",
        ]

    def test_run(self, mock_embeddings_response):
        docs = [
            Document(content="I love cheese", meta={"topic": "Cuisine"}),
            Document(content="A transformer is a deep learning architecture", meta={"topic": "ML"}),
            Document(content="Mixedbread AI is building yummy models", meta={"topic": "AI"})
        ]

        model = DEFAULT_VALUES["model"]
        embedder = MixedbreadAIDocumentEmbedder(
            api_key=Secret.from_token("fake-api-key"),
            model=model,
            prefix="prefix ",
            suffix=" suffix",
            batch_size=2,
            meta_fields_to_embed=["topic"],
            embedding_separator=" | ",
        )

        result = embedder.run(documents=docs)

        documents_with_embeddings = result["documents"]

        assert isinstance(documents_with_embeddings, list)
        assert len(documents_with_embeddings) == len(docs)
        for doc in documents_with_embeddings:
            assert isinstance(doc, Document)
            assert isinstance(doc.embedding, list)
            assert len(doc.embedding) == 3
            assert all(isinstance(x, float) for x in doc.embedding)

        assert result["meta"] == {
            "model": model,
            "usage": Usage(prompt_tokens=8, total_tokens=8).dict(),
            "object": ObjectType.LIST,
            "normalized": True,
            "encoding_format": EncodingFormat.FLOAT,
            "dimensions": 3
        }

    @pytest.mark.skipif(
        not os.environ.get("MXBAI_API_KEY", None),
        reason="Export an env var called MXBAI_API_KEY containing the Mixedbread AI API key to run this test.",
    )
    @pytest.mark.integration
    def test_live_run_with_real_documents(self):
        docs = [
            Document(content="The Eiffel Tower is in Paris", meta={"topic": "Travel"}),
            Document(content="Quantum mechanics is a fundamental theory in physics", meta={"topic": "Science"}),
            Document(content="Baking bread requires yeast", meta={"topic": "Cooking"})
        ]

        embedder = MixedbreadAIDocumentEmbedder(
            batch_size=2,
            meta_fields_to_embed=["topic"],
            embedding_separator=" | ",
        )

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
