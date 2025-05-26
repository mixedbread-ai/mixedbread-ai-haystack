from unittest.mock import patch

import pytest
from mixedbread_ai import Embedding, RerankingResponse, RankedDocument, ObjectType
from mixedbread_ai.types import EmbeddingsResponse, Usage


@pytest.fixture
def mock_embeddings_response():
    def mocking(*args, **kwargs):  # noqa: ARG001
        model = kwargs["model"]
        inputs = kwargs["input"]
        normalized = kwargs.get("normalized", None)
        encoding_format = kwargs.get("encoding_format", None)

        mock_response = EmbeddingsResponse(
            usage=Usage(total_tokens=4, prompt_tokens=4),
            model=model,
            object=ObjectType.LIST,
            normalized=normalized,
            encoding_format=encoding_format,
            dimensions=3,
            data=[
                Embedding(index=i, embedding=[0.1, 0.2, 0.3], object="embedding")
                for i in range(len(inputs))
            ],
        )
        return mock_response

    with patch(
        "mixedbread_ai.client.MixedbreadAI.embeddings", side_effect=mocking
    ) as mock_embeddings_response:
        yield mock_embeddings_response


@pytest.fixture
def mock_reranking_response():
    def mocking(*args, **kwargs):  # noqa: ARG001
        model = kwargs["model"]
        inputs = kwargs["input"]
        top_k = kwargs["top_k"]

        mock_response = RerankingResponse(
            usage=Usage(total_tokens=4, prompt_tokens=4),
            model=model,
            data=[
                RankedDocument(index=i, score=1.0 - 0.1 * i)
                for i in range(min(top_k, len(inputs)))
            ],
            object=None,
            top_k=top_k,
            return_input=False,
        )
        return mock_response

    with patch(
        "mixedbread_ai.client.MixedbreadAI.reranking", side_effect=mocking
    ) as mock_reranking_response:
        yield mock_reranking_response
