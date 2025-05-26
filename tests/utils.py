from unittest.mock import patch

import pytest
from typing import Literal
from mixedbread.types import EmbeddingCreateResponse
from mixedbread.types import RerankingCreateResponse
from mixedbread.types.embedding_create_response import DataUnionMember0 as Embedding, Usage as EmbeddingUsage
from mixedbread.types.reranking_create_response import Data as RankedDocument, Usage as RerankingUsage


@pytest.fixture
def mock_embeddings_response():
    def mocking(*args, **kwargs):  # noqa: ARG001
        model = kwargs["model"]
        inputs = kwargs["input"]
        normalized = kwargs.get("normalized", None)
        encoding_format = kwargs.get("encoding_format", None)

        mock_response = EmbeddingCreateResponse(
            usage=EmbeddingUsage(total_tokens=4, prompt_tokens=4, completion_tokens=None),
            model=model,
            object="list",
            normalized=normalized,
            encoding_format=encoding_format,
            dimensions=3,
            data=[Embedding(index=i, embedding=[0.1, 0.2, 0.3], object="embedding") for i in range(len(inputs))],
        )
        return mock_response

    with patch("mixedbread_ai_haystack.common.client.Mixedbread") as MockClient:
        mock_instance = MockClient.return_value
        mock_instance.embeddings.create.side_effect = mocking
        yield mock_instance.embeddings.create


@pytest.fixture
def mock_reranking_response():
    def mocking(*args, **kwargs):  # noqa: ARG001
        model = kwargs["model"]
        inputs = kwargs["input"]
        top_k = kwargs["top_k"]

        mock_response = RerankingCreateResponse(
            usage=RerankingUsage(total_tokens=4, prompt_tokens=4, completion_tokens=None),
            model=model,
            data=[
                RankedDocument(index=i, score=1.0 - 0.1 * i, input=None)
                for i in range(min(top_k, len(inputs)))
            ],
            object=None,
            top_k=top_k,
            return_input=False,
        )
        return mock_response

    with patch("mixedbread_ai_haystack.common.client.Mixedbread") as MockClient:
        mock_instance = MockClient.return_value
        mock_instance.reranking.create.side_effect = mocking
        yield mock_instance.reranking.create

