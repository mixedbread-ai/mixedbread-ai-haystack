import os
import pytest
from haystack import Document
from haystack.utils import Secret

from mixedbread.types.embedding_create_response import Usage
from mixedbread_ai_haystack.embedders import MixedbreadDocumentEmbedder
from mixedbread_ai_haystack.embedders.text_embedder import TextEmbedderMeta
from .test_config import TestConfig

DEFAULT_VALUES = {
    "base_url": None,
    "timeout": 60.0,
    "max_retries": 2,
    "model": "mixedbread-ai/mxbai-embed-large-v1",
    "prefix": "",
    "suffix": "",
    "normalized": True,
    "encoding_format": "float",
    "dimensions": None,
    "prompt": None,
    "batch_size": 128,
    "progress_bar": True,
    "embedding_separator": "\n",
    "meta_fields_to_embed": [],
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
        assert embedder.prefix == DEFAULT_VALUES["prefix"]
        assert embedder.suffix == DEFAULT_VALUES["suffix"]
        assert embedder.normalized == DEFAULT_VALUES["normalized"]
        assert embedder.encoding_format.value == DEFAULT_VALUES["encoding_format"]
        assert embedder.dimensions == DEFAULT_VALUES["dimensions"]
        assert embedder.prompt == DEFAULT_VALUES["prompt"]

        assert embedder.batch_size == DEFAULT_VALUES["batch_size"]
        assert embedder.progress_bar == DEFAULT_VALUES["progress_bar"]
        assert embedder.embedding_separator == DEFAULT_VALUES["embedding_separator"]
        assert embedder.meta_fields_to_embed == DEFAULT_VALUES["meta_fields_to_embed"]

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
            prefix="prefix",
            suffix="suffix",
            normalized=False,
            encoding_format="binary",
            dimensions=500,
            prompt="prompt",
            batch_size=64,
            progress_bar=False,
            embedding_separator=" | ",
            meta_fields_to_embed=["test_field"],
        )

        assert embedder.api_key == Secret.from_token("test-api-key")
        assert embedder.base_url == "http://example.com"
        assert embedder.timeout == 50.0
        assert embedder.max_retries == 10

        assert embedder.model == "model"
        assert embedder.prefix == "prefix"
        assert embedder.suffix == "suffix"
        assert not embedder.normalized
        assert embedder.encoding_format.value == "binary"
        assert embedder.dimensions == 500
        assert embedder.prompt == "prompt"

        assert embedder.batch_size == 64
        assert not embedder.progress_bar
        assert embedder.embedding_separator == " | "
        assert embedder.meta_fields_to_embed == ["test_field"]

    def test_init_fail_wo_api_key(self, monkeypatch):
        """
        Test that initialization fails when no API key is provided.
        """
        monkeypatch.delenv("MXBAI_API_KEY", raising=False)
        with pytest.raises(ValueError, match="None of the following authentication environment variables are set"):
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
            "init_parameters": {**DEFAULT_VALUES, "api_key": Secret.from_env_var("MXBAI_API_KEY").to_dict()},
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
            prefix="prefix",
            suffix="suffix",
            normalized=False,
            encoding_format="binary",
            dimensions=500,
            prompt="prompt",
            batch_size=64,
            progress_bar=False,
            meta_fields_to_embed=["test_field"],
            embedding_separator=" | ",
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
                "prefix": "prefix",
                "suffix": "suffix",
                "normalized": False,
                "encoding_format": "binary",
                "dimensions": 500,
                "prompt": "prompt",
                "batch_size": 64,
                "progress_bar": False,
                "meta_fields_to_embed": ["test_field"],
                "embedding_separator": " | ",
            },
        }

    def test_prepare_texts_to_embed_w_metadata(self):
        """
        Test document preparation with metadata fields.
        """
        documents = [
            Document(
                content=f"document number {i}:\ncontent",
                meta={"meta_field": f"meta_value {i}"},
            )
            for i in range(5)
        ]

        embedder = MixedbreadDocumentEmbedder(
            api_key=Secret.from_token("fake-api-key"), meta_fields_to_embed=["meta_field"], embedding_separator=" | "
        )

        prepared_texts = embedder._prepare_texts_to_embed(documents)

        assert prepared_texts == [
            "meta_value 0 | document number 0:\ncontent",
            "meta_value 1 | document number 1:\ncontent",
            "meta_value 2 | document number 2:\ncontent",
            "meta_value 3 | document number 3:\ncontent",
            "meta_value 4 | document number 4:\ncontent",
        ]

    def test_prepare_texts_to_embed_w_suffix(self):
        """
        Test document preparation with prefix and suffix.
        """
        documents = [Document(content=f"document number {i}") for i in range(5)]

        embedder = MixedbreadDocumentEmbedder(
            api_key=Secret.from_token("fake-api-key"), prefix="my_prefix ", suffix=" my_suffix"
        )

        prepared_texts = embedder._prepare_texts_to_embed(documents)

        assert prepared_texts == [
            "my_prefix document number 0 my_suffix",
            "my_prefix document number 1 my_suffix",
            "my_prefix document number 2 my_suffix",
            "my_prefix document number 3 my_suffix",
            "my_prefix document number 4 my_suffix",
        ]

    def test_run_wrong_input_format(self):
        """
        Test for checking incorrect input when creating embeddings.
        """
        embedder = MixedbreadDocumentEmbedder(api_key=Secret.from_token("fake-api-key"))

        with pytest.raises(TypeError, match="Input must be a list of Haystack Documents"):
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
            Document(content="Machine learning is transforming industries", meta={"topic": "AI"}),
        ]

        embedder_config = TestConfig.get_test_embedder_config()
        embedder = MixedbreadDocumentEmbedder(
            batch_size=2, meta_fields_to_embed=["topic"], embedding_separator=" | ", **embedder_config
        )

        result = embedder.run(documents=docs)
        documents_with_embeddings = result["documents"]

        # Basic validation
        assert isinstance(documents_with_embeddings, list)
        assert len(documents_with_embeddings) == len(docs)
        for doc in documents_with_embeddings:
            assert isinstance(doc, Document)
            assert isinstance(doc.embedding, list)
            assert len(doc.embedding) > 0
            assert all(isinstance(x, float) for x in doc.embedding)

        assert "meta" in result
