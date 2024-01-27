from mixedbread_ai.mixedbread_ai_client import models


def mock_embeddings_response(*args, **kwargs):  # noqa: ARG001
    inputs = kwargs["input"]
    model = kwargs["model"]
    normalized = kwargs["normalized"]
    mock_response = models.EmbeddingsResponse(
        model=model,
        object_="list",
        normalized=normalized,
        usage=models.ModelUsage(total_tokens=4, prompt_tokens=4),
        data=[models.Embedding(index=i, embedding=[0.1, 0.2, 0.3], object_="embedding") for i in range(len(inputs))],
    )
    return mock_response
