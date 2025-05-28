from typing import Any, Dict, List, Optional, TypedDict, Union
from haystack import component, default_to_dict, default_from_dict
from haystack.utils import Secret, deserialize_secrets_inplace
from mixedbread_ai_haystack.common.client import MixedbreadClient
from mixedbread_ai_haystack.embedders.embedding_types import MixedbreadEmbeddingType


class TextEmbedderMeta(TypedDict):
    model: str
    usage: Dict[str, int]
    normalized: bool
    encoding_format: Union[str, List[str]]
    dimensions: Optional[int]
    object: Optional[str]


@component
class MixedbreadTextEmbedder(MixedbreadClient):
    """
    Embed a single string using the Mixedbread Embeddings API.
    """

    def __init__(
        self,
        api_key: Secret = Secret.from_env_var("MXBAI_API_KEY"),
        model: str = "mixedbread-ai/mxbai-embed-large-v1",
        normalized: bool = True,
        encoding_format: Union[
            str, MixedbreadEmbeddingType
        ] = MixedbreadEmbeddingType.FLOAT,
        dimensions: Optional[int] = None,
        prompt: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: Optional[float] = 60.0,
        max_retries: Optional[int] = 2,
    ):
        super(MixedbreadTextEmbedder, self).__init__(
            api_key=api_key, base_url=base_url, timeout=timeout, max_retries=max_retries
        )
        self.model = model
        self.normalized = normalized
        if isinstance(encoding_format, str):
            self.encoding_format = MixedbreadEmbeddingType.from_str(encoding_format)
        else:
            self.encoding_format = encoding_format
        self.dimensions = dimensions
        self.prompt = prompt

    def to_dict(self) -> Dict[str, Any]:
        client_params = MixedbreadClient.to_dict(self)["init_parameters"]
        return default_to_dict(
            self,
            **client_params,
            model=self.model,
            normalized=self.normalized,
            encoding_format=self.encoding_format.value,
            dimensions=self.dimensions,
            prompt=self.prompt,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MixedbreadTextEmbedder":
        deserialize_secrets_inplace(data["init_parameters"], keys=["api_key"])
        ef_val = data["init_parameters"].get("encoding_format")
        if isinstance(ef_val, str):
            data["init_parameters"]["encoding_format"] = (
                MixedbreadEmbeddingType.from_str(ef_val)
            )
        return default_from_dict(cls, data)

    @component.output_types(embedding=List[float], meta=TextEmbedderMeta)
    def run(self, text: str, prompt: Optional[str] = None) -> Dict[str, Any]:
        response = self.client.embed(
            model=self.model,
            input=[text],
            normalized=self.normalized,
            encoding_format=self.encoding_format.value,
            dimensions=self.dimensions,
            prompt=prompt or self.prompt,
        )

        embedding = response.data[0].embedding if response.data else []

        meta = {
            "model": response.model,
            "usage": response.usage.model_dump(),
            "normalized": response.normalized,
            "encoding_format": response.encoding_format,
            "dimensions": response.dimensions,
            "object": response.object,
        }

        return {"embedding": embedding, "meta": meta}

    @component.output_types(embedding=List[float], meta=TextEmbedderMeta)
    async def run_async(
        self, text: str, prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        response = await self.async_client.embed(
            model=self.model,
            input=[text],
            normalized=self.normalized,
            encoding_format=self.encoding_format.value,
            dimensions=self.dimensions,
            prompt=prompt or self.prompt,
        )

        embedding = response.data[0].embedding if response.data else []

        meta = {
            "model": response.model,
            "usage": response.usage.model_dump(),
            "normalized": response.normalized,
            "encoding_format": response.encoding_format,
            "dimensions": response.dimensions,
            "object": response.object,
        }

        return {"embedding": embedding, "meta": meta}
