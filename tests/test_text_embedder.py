from unittest.mock import patch

import pytest
from mixedbread_ai_haystack.embedders import MixedbreadAiTextEmbedder
from mixedbread_ai import models
from .utils import mock_embeddings_response


class TestMixedbreadAiTextEmbedder:
    def test_init_default(self, monkeypatch):
        monkeypatch.setenv("MIXEDBREAD_API_KEY", "fake-api-key")
        embedder = MixedbreadAiTextEmbedder()

        assert embedder.model_name == "UAE-Large-V1"
        assert embedder.prefix == ""
        assert embedder.suffix == ""

    def test_init_with_parameters(self):
        embedder = MixedbreadAiTextEmbedder(
            api_key="fake-api-key",
            model="model",
            prefix="prefix",
            suffix="suffix",
        )
        assert embedder.model_name == "model"
        assert embedder.prefix == "prefix"
        assert embedder.suffix == "suffix"

    def test_init_fail_wo_api_key(self, monkeypatch):
        monkeypatch.delenv("MIXEDBREAD_API_KEY", raising=False)
        with pytest.raises(Exception):
            MixedbreadAiTextEmbedder()

    def test_to_dict(self):
        component = MixedbreadAiTextEmbedder(api_key="fake-api-key")
        data = component.to_dict()
        assert data == {
            "type": "mixedbread_ai_haystack.embedders.text_embedder.MixedbreadAiTextEmbedder",
            "init_parameters": {
                "model": "UAE-Large-V1",
                "prefix": "",
                "suffix": "",
                "instruction": None,
                "normalized": True,
            },
        }

    def test_to_dict_with_custom_init_parameters(self):
        component = MixedbreadAiTextEmbedder(
            api_key="fake-api-key",
            model="model",
            prefix="prefix",
            suffix="suffix",
            instruction="instruction",
            normalized=False,
        )
        data = component.to_dict()
        assert data == {
            "type": "mixedbread_ai_haystack.embedders.text_embedder.MixedbreadAiTextEmbedder",
            "init_parameters": {
                "model": "model",
                "prefix": "prefix",
                "suffix": "suffix",
                "instruction": "instruction",
                "normalized": False,
            },
        }

    def test_run(self):
        model = "UAE-Large-V1"
        with patch("mixedbread_ai.MixedbreadAi.embeddings", side_effect=mock_embeddings_response):
            embedder = MixedbreadAiTextEmbedder(api_key="fake-api-key", model=model, prefix="prefix ", suffix=" suffix")
            result = embedder.run(text="The food was delicious")

        assert len(result["embedding"]) == 3
        assert all(isinstance(x, float) for x in result["embedding"])
        assert result["meta"] == {
            "model": model,
            "usage": models.ModelUsage(prompt_tokens=4, total_tokens=4),
            "normalized": True,
            "truncated": result["meta"]["truncated"],
        }

    def test_run_wrong_input_format(self):
        embedder = MixedbreadAiTextEmbedder(api_key="fake-api-key")

        list_integers_input = [1, 2, 3]

        with pytest.raises(TypeError, match="MixedbreadAiTextEmbedder expects a string as an input"):
            embedder.run(text=list_integers_input)
