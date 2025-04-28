from unittest.mock import MagicMock, patch

import pytest
import torch
from transformers.modeling_outputs import CausalLMOutput

from haystack import Document
from haystack.utils.device import ComponentDevice

from mxbai_rerank.base import RankResult

from mixedbread_ai_haystack.rerankers.local_reranker_v2 import LocalMixedbreadAIRerankerV2


class TestSimilarityRanker:
    def test_to_dict(self):
        component = LocalMixedbreadAIRerankerV2()
        data = component.to_dict()
        assert data == {
            "type": "mixedbread_ai_haystack.rerankers.local_reranker_v2.LocalMixedbreadAIRerankerV2",
            "init_parameters": {
                "device": None,
                "top_k": 10,
                "model": "mixedbread-ai/mxbai-rerank-base-v2",
                "meta_fields_to_rank": [],
                "ranking_separator": "\n",
                "score_threshold": None,
                "model_kwargs": {},
                "tokenizer_kwargs": None,
                "batch_size": 16,
            },
        }

    def test_to_dict_with_custom_init_parameters(self):
        component = LocalMixedbreadAIRerankerV2(
            model="my_model",
            device=ComponentDevice.from_str("cuda:0"),
            top_k=5,
            score_threshold=0.01,
            model_kwargs={"torch_dtype": torch.float16},
            tokenizer_kwargs={"model_max_length": 512},
            batch_size=32,
        )
        data = component.to_dict()
        assert data == {
            "type": "mixedbread_ai_haystack.rerankers.local_reranker_v2.LocalMixedbreadAIRerankerV2",
            "init_parameters": {
                "device": ComponentDevice.from_str("cuda:0").to_dict(),
                "model": "my_model",
                "top_k": 5,
                "meta_fields_to_rank": [],
                "ranking_separator": "\n",
                "score_threshold": 0.01,
                "model_kwargs": {
                    "torch_dtype": "torch.float16",
                },
                "tokenizer_kwargs": {"model_max_length": 512},
                "batch_size": 32,
            },
        }

    def test_from_dict(self):
        data = {
            "type": "mixedbread_ai_haystack.rerankers.local_reranker_v2.LocalMixedbreadAIRerankerV2",
            "init_parameters": {
                "device": None,
                "model": "my_model",
                "top_k": 5,
                "meta_fields_to_rank": [],
                "ranking_separator": "\n",
                "score_threshold": 0.01,
                "model_kwargs": {"torch_dtype": "torch.float16"},
                "tokenizer_kwargs": {"model_max_length": 512},
                "batch_size": 32,
            },
        }

        component = LocalMixedbreadAIRerankerV2.from_dict(data)
        assert component.device is None
        assert component.model == "my_model"
        assert component.top_k == 5
        assert component.meta_fields_to_rank == []
        assert component.ranking_separator == "\n"
        assert component.score_threshold == 0.01
        assert component.model_kwargs == {"torch_dtype": torch.float16}
        assert component.tokenizer_kwargs == {"model_max_length": 512}
        assert component.batch_size == 32

    def test_meta_fields_to_rank(self):
        ranker = LocalMixedbreadAIRerankerV2(
            model="model", meta_fields_to_rank=["meta_field"], ranking_separator="\n"
        )
        ranker._torch_model = MagicMock()
        ranker._torch_model.rank.return_value = []

        documents = [Document(content=f"document number {i}", meta={"meta_field": f"meta_value {i}"}) for i in range(5)]
        ranker.run(query="test", documents=documents)
        ranker._torch_model.rank(
            query="test",
            documents=[
                "meta_value 0\ndocument number 0",
                "meta_value 1\ndocument number 1",
                "meta_value 2\ndocument number 2",
                "meta_value 3\ndocument number 3",
                "meta_value 4\ndocument number 4",
            ],
            top_k=10,
            return_documents=True,
            sort=True,
            batch_size=16,
        )

    def test_score_threshold(self):
        ranker = LocalMixedbreadAIRerankerV2(score_threshold=0.1)
        ranker._torch_model = MagicMock()
        ranker._torch_model.rank.return_value = [
            RankResult(index=2, score=4.54, document="Sarajevo"),
            RankResult(index=1, score=4.41, document="Belgrade"),
            RankResult(index=0, score=-1.0, document="Berlin")
        ]

        docs_before_texts = ["Berlin", "Belgrade", "Sarajevo"]
        documents = [Document(content=text) for text in docs_before_texts]
        out = ranker.run(query="test", documents=documents)
        assert len(out["documents"]) == 2
        assert out["documents"][0].content == "Sarajevo"
        assert out["documents"][0].score == 4.54

    def test_returns_empty_list_if_no_documents_are_provided(self):
        sampler = LocalMixedbreadAIRerankerV2()
        sampler._torch_model = MagicMock()
        output = sampler.run(query="City in Germany", documents=[])
        assert not output["documents"]

    def test_raises_component_error_if_model_not_warmed_up(self):
        sampler = LocalMixedbreadAIRerankerV2()
        with pytest.raises(RuntimeError):
            sampler.run(query="query", documents=[Document(content="document")])

    @pytest.mark.integration
    def test_run(self):
        """
        Test if the component ranks documents correctly.
        """
        ranker = LocalMixedbreadAIRerankerV2()
        ranker.warm_up()

        query = "City in Bosnia and Herzegovina"
        docs_before_texts = ["Berlin", "Belgrade", "Sarajevo"]
        expected_first_text = "Sarajevo"
        expected_scores = [4.543488502502441, 4.413060188293457, -0.9995536804199219]

        docs_before = [Document(content=text) for text in docs_before_texts]
        output = ranker.run(query=query, documents=docs_before)
        docs_after = output["documents"]

        assert len(docs_after) == 3
        assert docs_after[0].content == expected_first_text

        sorted_scores = sorted(expected_scores, reverse=True)
        assert docs_after[0].score == pytest.approx(sorted_scores[0], abs=1e-6)
        assert docs_after[1].score == pytest.approx(sorted_scores[1], abs=1e-6)
        assert docs_after[2].score == pytest.approx(sorted_scores[2], abs=1e-6)
