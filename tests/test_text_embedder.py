import os

import pytest
from haystack.utils import Secret
from mixedbread_ai import EncodingFormat, TruncationStrategy, ObjectType
from mixedbread_ai.types import Usage

from mixedbread_ai_haystack.embedders import MixedbreadAITextEmbedder
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
}


class TestMixedbreadAITextEmbedder:
    def test_init_default(self, monkeypatch):
        monkeypatch.setenv("MXBAI_API_KEY", "fake-api-key")
        embedder = MixedbreadAITextEmbedder()

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

    def test_init_with_parameters(self):
        embedder = MixedbreadAITextEmbedder(
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
            prompt="prompt"
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

    def test_init_fail_wo_api_key(self, monkeypatch):
        monkeypatch.delenv("MXBAI_API_KEY", raising=False)
        with pytest.raises(Exception):
            MixedbreadAITextEmbedder()

    def test_to_dict(self, monkeypatch):
        monkeypatch.setenv("MXBAI_API_KEY", "fake-api-key")
        component = MixedbreadAITextEmbedder()
        data = component.to_dict()
        assert data == {
            "type": "mixedbread_ai_haystack.embedders.text_embedder.MixedbreadAITextEmbedder",
            "init_parameters":
                {
                    **DEFAULT_VALUES,
                    "api_key": Secret.from_env_var("MXBAI_API_KEY").to_dict()
                }
        }

    def test_to_dict_with_custom_init_parameters(self, monkeypatch):
        monkeypatch.setenv("MXBAI_API_KEY", "fake-api-key")
        component = MixedbreadAITextEmbedder(
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
            prompt="prompt"
        )
        data = component.to_dict()
        assert data == {
            "type": "mixedbread_ai_haystack.embedders.text_embedder.MixedbreadAITextEmbedder",
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
            },
        }

    def test_run(self, mock_embeddings_response):
        model = "mixedbread-ai/mxbai-embed-large-v1"
        embedder = MixedbreadAITextEmbedder(
            api_key=Secret.from_token("fake-api-key"),
            model=model,
            prefix="prefix ",
            suffix=" suffix"
        )
        result = embedder.run(text="The food was delicious")

        assert len(result["embedding"]) == 3
        assert all(isinstance(x, float) for x in result["embedding"])
        assert result["meta"] == {
            "model": model,
            "usage": Usage(prompt_tokens=4, total_tokens=4).dict(),
            "object": ObjectType.LIST,
            "normalized": True,
            "encoding_format": EncodingFormat.FLOAT,
            "dimensions": 3
        }

    def test_run_wrong_input_format(self):
        embedder = MixedbreadAITextEmbedder(api_key=Secret.from_token("fake-api-key"))

        list_integers_input = [1, 2, 3]

        with pytest.raises(TypeError, match="MixedbreadAITextEmbedder expects a string as an input. "
                                            "In case you want to embed a list of Documents, "
                                            "please use the MixedbreadAIDocumentEmbedder."):
            embedder.run(text=list_integers_input)

    @pytest.mark.skipif(
        not os.environ.get("MXBAI_API_KEY", None),
        reason="Export an env var called MXBAI_API_KEY containing the Mixedbread AI API key to run this test.",
    )
    @pytest.mark.integration
    def test_live_run_with_real_text(self):
        embedder = MixedbreadAITextEmbedder()
        result = embedder.run(text="This is a live test with real text input.")

        assert isinstance(result["embedding"], list)
        assert all(isinstance(x, float) for x in result["embedding"])
        assert "meta" in result
