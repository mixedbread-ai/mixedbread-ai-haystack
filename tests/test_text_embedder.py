import os

import pytest
from haystack.utils import Secret
from typing import Literal
from mixedbread.types.embedding_create_response import Usage as EmbeddingUsage

from mixedbread_ai_haystack.embedders import MixedbreadTextEmbedder
from mixedbread_ai_haystack.embedders.text_embedder import EmbedderMeta
from .utils import mock_embeddings_response

DEFAULT_VALUES = {
    "base_url": None,
    "timeout": 60.0,
    "max_retries": 3,
    "model": "mixedbread-ai/mxbai-embed-large-v1",
    "prefix": "",
    "suffix": "",
    "normalized": True,
    "encoding_format": "float",
    "truncation_strategy": "start",
    "dimensions": None,
    "prompt": None,
}


class TestMixedbreadTextEmbedder:
    def test_init_default(self, monkeypatch):
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
        assert embedder.encoding_format == DEFAULT_VALUES["encoding_format"]
        assert embedder.truncation_strategy == DEFAULT_VALUES["truncation_strategy"]
        assert embedder.dimensions == DEFAULT_VALUES["dimensions"]
        assert embedder.prompt == DEFAULT_VALUES["prompt"]

    def test_init_with_parameters(self):
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
            truncation_strategy="end",
            dimensions=500,
            prompt="prompt"
        )

        assert embedder.api_key == Secret.from_token("test-api-key")
        assert embedder.base_url == "http://example.com"
        assert embedder.timeout == 50.0
        assert embedder.max_retries == 10

        assert embedder.model == "model"
        assert embedder.prefix == "prefix"
        assert embedder.suffix == "suffix"
        assert not embedder.normalized
        assert embedder.encoding_format == "binary"
        assert embedder.truncation_strategy == "end"
        assert embedder.dimensions == 500
        assert embedder.prompt == "prompt"

    def test_init_fail_wo_api_key(self, monkeypatch):
        monkeypatch.delenv("MXBAI_API_KEY", raising=False)
        with pytest.raises(Exception):
            MixedbreadTextEmbedder()

    def test_to_dict(self, monkeypatch):
        monkeypatch.setenv("MXBAI_API_KEY", "fake-api-key")
        component = MixedbreadTextEmbedder()
        data = component.to_dict()
        assert data == {
            "type": "mixedbread_ai_haystack.embedders.text_embedder.MixedbreadTextEmbedder",
            "init_parameters":
                {
                    **DEFAULT_VALUES,
                    "api_key": Secret.from_env_var("MXBAI_API_KEY").to_dict()
                }
        }

    def test_to_dict_with_custom_init_parameters(self, monkeypatch):
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
            truncation_strategy="end",
            dimensions=500,
            prompt="prompt"
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
                "truncation_strategy": "end",
                "dimensions": 500,
                "prompt": "prompt",
            },
        }

    def test_run(self, mock_embeddings_response):
        model = "mixedbread/mxbai-embed-large-v1"
        embedder = MixedbreadTextEmbedder(
            api_key=Secret.from_token("fake-api-key"),
            model=model,
            prefix="prefix ",
            suffix=" suffix"
        )
        result = embedder.run(text="The food was delicious")

        assert len(result["embedding"]) == 3
        assert all(isinstance(x, float) for x in result["embedding"])
        assert result["meta"] == EmbedderMeta(
            usage=EmbeddingUsage(prompt_tokens=4, total_tokens=4),
            model=model,
            object="list",
            normalized=True,
            encoding_format="float",
            dimensions=3
        )

    def test_run_wrong_input_format(self):
        embedder = MixedbreadTextEmbedder(api_key=Secret.from_token("fake-api-key"))

        list_integers_input = [1, 2, 3]

        with pytest.raises(TypeError, match="MixedbreadTextEmbedder expects a string as an input. "
                                            "In case you want to embed a list of Documents, "
                                            "please use the MixedbreadDocumentEmbedder."):
            embedder.run(text=list_integers_input)

    @pytest.mark.skipif(
        not os.environ.get("MXBAI_API_KEY", None),
        reason="Export an env var called MXBAI_API_KEY containing the Mixedbread API key to run this test.",
    )
    def test_live_run_with_real_text(self):
        embedder = MixedbreadTextEmbedder()
        result = embedder.run(text="This is a live test with real text input.")

        assert isinstance(result["embedding"], list)
        assert all(isinstance(x, float) for x in result["embedding"])
        assert "meta" in result
