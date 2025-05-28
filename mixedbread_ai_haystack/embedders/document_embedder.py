from typing import Optional, List, Dict, Any, Union

from haystack import component, Document, default_to_dict, default_from_dict
from haystack.utils import Secret, deserialize_secrets_inplace
from mixedbread.types.shared import Usage as MixedUsage

from mixedbread_ai_haystack.common.client import MixedbreadClient
from mixedbread_ai_haystack.embedders.embedding_types import MixedbreadEmbeddingType
from mixedbread_ai_haystack.embedders.utils import (
    get_embedding_response,
    get_async_embedding_response,
)
from mixedbread_ai_haystack.embedders.text_embedder import TextEmbedderMeta


@component
class MixedbreadDocumentEmbedder(MixedbreadClient):
    """
    Embed multiple documents using the Mixedbread Embeddings API.
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
        super(MixedbreadDocumentEmbedder, self).__init__(
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
    def from_dict(cls, data: Dict[str, Any]) -> "MixedbreadDocumentEmbedder":
        deserialize_secrets_inplace(data["init_parameters"], keys=["api_key"])
        ef_val = data["init_parameters"].get("encoding_format")
        if isinstance(ef_val, str):
            data["init_parameters"]["encoding_format"] = (
                MixedbreadEmbeddingType.from_str(ef_val)
            )
        return default_from_dict(cls, data)

    def _prepare_texts_to_embed(self, documents: List[Document]) -> List[str]:
        texts = []
        for doc in documents:
            content_to_embed = doc.content or ""
            texts.append(content_to_embed)
        return texts

    @component.output_types(documents=List[Document], meta=TextEmbedderMeta)
    def run(
        self, documents: List[Document], prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        if not isinstance(documents, list) or (
            documents and not isinstance(documents[0], Document)
        ):
            raise TypeError("Input must be a list of Haystack Documents.")
        if not documents:
            usage = MixedUsage(prompt_tokens=0, total_tokens=0)
            return {
                "documents": [],
                "meta": TextEmbedderMeta(
                    usage=usage.model_dump(),
                    model=self.model,
                    normalized=self.normalized,
                    encoding_format=self.encoding_format.value,
                    dimensions=self.dimensions,
                    object="list",
                ),
            }

        texts_to_embed = self._prepare_texts_to_embed(documents)

        embeddings, meta = get_embedding_response(
            client=self.client,
            texts=texts_to_embed,
            model=self.model,
            normalized=self.normalized,
            encoding_format=self.encoding_format,
            dimensions=self.dimensions,
            prompt=prompt or self.prompt,
        )

        for doc, embedding in zip(documents, embeddings):
            doc.embedding = embedding

        return {"documents": documents, "meta": meta}

    @component.output_types(documents=List[Document], meta=TextEmbedderMeta)
    async def run_async(
        self, documents: List[Document], prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        if not isinstance(documents, list) or (
            documents and not isinstance(documents[0], Document)
        ):
            raise TypeError("Input must be a list of Haystack Documents.")
        if not documents:
            usage = MixedUsage(prompt_tokens=0, total_tokens=0)
            return {
                "documents": [],
                "meta": TextEmbedderMeta(
                    usage=usage.model_dump(),
                    model=self.model,
                    normalized=self.normalized,
                    encoding_format=self.encoding_format.value,
                    dimensions=self.dimensions,
                    object="list",
                ),
            }

        texts_to_embed = self._prepare_texts_to_embed(documents)

        embeddings, meta = await get_async_embedding_response(
            async_client=self.async_client,
            texts=texts_to_embed,
            model=self.model,
            normalized=self.normalized,
            encoding_format=self.encoding_format,
            dimensions=self.dimensions,
            prompt=prompt or self.prompt,
        )

        for doc, embedding in zip(documents, embeddings):
            doc.embedding = embedding

        return {"documents": documents, "meta": meta}
