import os

import pytest
from haystack.utils import Secret
from mixedbread.types.embedding_create_response import Usage as EmbeddingUsage

from mixedbread_ai_haystack.embedders import MixedbreadTextEmbedder
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
        assert embedder.prefix == DEFAULT_VALUES["prefix"]
        assert embedder.suffix == DEFAULT_VALUES["suffix"]
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
            prefix="prefix",
            suffix="suffix",
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
        assert embedder.prefix == "prefix"
        assert embedder.suffix == "suffix"
        assert not embedder.normalized
        assert embedder.encoding_format.value == "binary"
        assert embedder.dimensions == 500
        assert embedder.prompt == "prompt"

    def test_init_fail_wo_api_key(self, monkeypatch):
        """
        Test that initialization fails when no API key is provided.
        """
        monkeypatch.delenv("MXBAI_API_KEY", raising=False)
        with pytest.raises(ValueError, match="None of the following authentication environment variables are set"):
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
            "init_parameters": {**DEFAULT_VALUES, "api_key": Secret.from_env_var("MXBAI_API_KEY").to_dict()},
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
            prefix="prefix",
            suffix="suffix",
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
                "prefix": "prefix",
                "suffix": "suffix",
                "normalized": False,
                "encoding_format": "binary",
                "dimensions": 500,
                "prompt": "prompt",
            },
        }

    def test_run_wrong_input_format(self):
        """
        Test for checking incorrect input when creating embedding.
        """
        embedder = MixedbreadTextEmbedder(api_key=Secret.from_token("fake-api-key"))

        list_integers_input = [1, 2, 3]

        with pytest.raises(TypeError, match="MixedbreadTextEmbedder expects a string as input"):
            embedder.run(text=list_integers_input)

    def test_run_with_list_input(self):
        """
        Test for checking that list input raises appropriate error with helpful message.
        """
        embedder = MixedbreadTextEmbedder(api_key=Secret.from_token("fake-api-key"))

        list_input = ["text1", "text2"]

        with pytest.raises(TypeError, match="MixedbreadTextEmbedder expects a string as input"):
            embedder.run(text=list_input)

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
