from typing import Any, Dict, List, Tuple, Optional, Union

from mixedbread import Mixedbread as MixedbreadSDKClient, AsyncMixedbread as AsyncMixedbreadSDKClient
from mixedbread.types import (
    EmbeddingCreateResponse,
    Embedding as SDKEmbedding,
    MultiEncodingEmbedding as SDKMultiEncodingEmbedding,
)
from mixedbread_ai_haystack.embedders.embedding_types import MixedbreadEmbeddingType


def _extract_embedding_from_item(
    item: Union[SDKEmbedding, SDKMultiEncodingEmbedding],
) -> Union[List[float], List[int], str]:
    """Extracts a single embedding vector from an SDK response item."""
    if isinstance(item, SDKEmbedding):
        return item.embedding
    if isinstance(item, SDKMultiEncodingEmbedding):
        if item.embedding.float:
            return item.embedding.float
        if item.embedding.int8:
            return item.embedding.int8
        if item.embedding.base64:
            return item.embedding.base64
        for val in vars(item.embedding).values():
            if val is not None:
                return val
    return []


def get_embedding_response(
    client: MixedbreadSDKClient,
    texts: List[str],
    model: str,
    normalized: bool,
    encoding_format: MixedbreadEmbeddingType,
    dimensions: Optional[int],
    prompt: Optional[str],
) -> Tuple[List[Union[List[float], List[int], str]], Dict[str, Any]]:
    """Gets embeddings from Mixedbread API synchronously."""
    response: EmbeddingCreateResponse = client.embed(
        model=model,
        input=texts,
        normalized=normalized,
        encoding_format=encoding_format.value,
        dimensions=dimensions,
        prompt=prompt,
    )

    embeddings = [_extract_embedding_from_item(item) for item in response.data] if response.data else []

    metadata = {
        "model": response.model,
        "usage": response.usage.model_dump(),
        "normalized": response.normalized,
        "encoding_format": response.encoding_format,
        "dimensions": response.dimensions,
        "object": response.object,
    }
    return embeddings, metadata


async def get_async_embedding_response(
    async_client: AsyncMixedbreadSDKClient,
    texts: List[str],
    model: str,
    normalized: bool,
    encoding_format: MixedbreadEmbeddingType,
    dimensions: Optional[int],
    prompt: Optional[str],
) -> Tuple[List[Union[List[float], List[int], str]], Dict[str, Any]]:
    """Gets embeddings from Mixedbread API asynchronously."""
    response: EmbeddingCreateResponse = await async_client.embed(
        model=model,
        input=texts,
        normalized=normalized,
        encoding_format=encoding_format.value,
        dimensions=dimensions,
        prompt=prompt,
    )

    embeddings = [_extract_embedding_from_item(item) for item in response.data] if response.data else []

    metadata = {
        "model": response.model,
        "usage": response.usage.model_dump(),
        "normalized": response.normalized,
        "encoding_format": response.encoding_format,
        "dimensions": response.dimensions,
        "object": response.object,
    }
    return embeddings, metadata
