import asyncio
from typing import Optional, List, Dict, Any, Literal, Union

from haystack import component, Document, default_to_dict, default_from_dict
from haystack.utils import Secret, deserialize_secrets_inplace
from mixedbread.types.shared import Usage as MixedUsage

from mixedbread_ai_haystack.common.client import MixedbreadClient
from mixedbread_ai_haystack.embedders.embedding_types import MixedbreadEmbeddingType
from mixedbread_ai_haystack.embedders.utils import get_embedding_response, get_async_embedding_response
from mixedbread_ai_haystack.embedders.text_embedder import TextEmbedderMeta

try:
    from tqdm import tqdm

    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

    def tqdm(iterable, *args, **kwargs):
        return iterable


@component
class MixedbreadDocumentEmbedder(MixedbreadClient):
    """
    Embeds Haystack Documents using Mixedbread AI.
    """

    def __init__(
        self,
        api_key: Secret = Secret.from_env_var("MXBAI_API_KEY"),
        model: str = "mixedbread-ai/mxbai-embed-large-v1",
        prefix: str = "",
        suffix: str = "",
        normalized: bool = True,
        encoding_format: Union[str, MixedbreadEmbeddingType] = MixedbreadEmbeddingType.FLOAT,
        dimensions: Optional[int] = None,
        prompt: Optional[str] = None,
        batch_size: int = 128,
        progress_bar: bool = True,
        embedding_separator: str = "\n",
        meta_fields_to_embed: Optional[List[str]] = None,
        base_url: Optional[str] = None,
        timeout: Optional[float] = 60.0,
        max_retries: Optional[int] = 2,
    ):
        super(MixedbreadDocumentEmbedder, self).__init__(
            api_key=api_key, base_url=base_url, timeout=timeout, max_retries=max_retries
        )

        if not (1 <= batch_size <= 256):  # Check SDK for actual limits if any
            raise ValueError("batch_size must be between 1 and 256.")
        if progress_bar and not HAS_TQDM:
            raise ImportError("tqdm is not installed. Please install it to use the progress bar.")

        self.model = model
        self.prefix = prefix
        self.suffix = suffix
        self.normalized = normalized
        if isinstance(encoding_format, str):
            self.encoding_format = MixedbreadEmbeddingType.from_str(encoding_format)
        else:
            self.encoding_format = encoding_format
        self.dimensions = dimensions
        self.prompt = prompt
        self.batch_size = batch_size
        self.progress_bar = progress_bar
        self.embedding_separator = embedding_separator
        self.meta_fields_to_embed = meta_fields_to_embed or []

    def to_dict(self) -> Dict[str, Any]:
        client_params = MixedbreadClient.to_dict(self)["init_parameters"]
        return default_to_dict(
            self,
            **client_params,
            model=self.model,
            prefix=self.prefix,
            suffix=self.suffix,
            normalized=self.normalized,
            encoding_format=self.encoding_format.value,
            dimensions=self.dimensions,
            prompt=self.prompt,
            batch_size=self.batch_size,
            progress_bar=self.progress_bar,
            embedding_separator=self.embedding_separator,
            meta_fields_to_embed=self.meta_fields_to_embed,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MixedbreadDocumentEmbedder":
        deserialize_secrets_inplace(data["init_parameters"], keys=["api_key"])
        ef_val = data["init_parameters"].get("encoding_format")
        if isinstance(ef_val, str):
            data["init_parameters"]["encoding_format"] = MixedbreadEmbeddingType.from_str(ef_val)
        return default_from_dict(cls, data)

    def _prepare_texts_to_embed(self, documents: List[Document]) -> List[str]:
        texts = []
        for doc in documents:
            meta_content = self.embedding_separator.join(
                str(doc.meta[key]) for key in self.meta_fields_to_embed if doc.meta.get(key) is not None
            )
            content_to_embed = doc.content or ""
            if meta_content and content_to_embed:
                full_text = meta_content + self.embedding_separator + content_to_embed
            elif meta_content:
                full_text = meta_content
            else:
                full_text = content_to_embed
            texts.append(f"{self.prefix}{full_text}{self.suffix}")
        return texts

    @component.output_types(documents=List[Document], meta=TextEmbedderMeta)
    def run(self, documents: List[Document], prompt: Optional[str] = None) -> Dict[str, Any]:
        if not isinstance(documents, list) or (documents and not isinstance(documents[0], Document)):
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

        all_embeddings: List[Union[List[float], List[int], str]] = []
        final_meta: Optional[Dict[str, Any]] = None

        for i in tqdm(
            range(0, len(texts_to_embed), self.batch_size), disable=not self.progress_bar, desc="Embedding documents"
        ):
            batch_texts = texts_to_embed[i : i + self.batch_size]
            embeddings_batch, meta_batch = get_embedding_response(
                client=self.client,
                texts=batch_texts,
                model=self.model,
                normalized=self.normalized,
                encoding_format=self.encoding_format,
                dimensions=self.dimensions,
                prompt=prompt or self.prompt,
            )
            all_embeddings.extend(embeddings_batch)
            if final_meta is None:
                final_meta = meta_batch
                final_meta["usage"] = meta_batch["usage"].copy()
            else:
                final_meta["usage"]["prompt_tokens"] += meta_batch["usage"]["prompt_tokens"]
                final_meta["usage"]["total_tokens"] += meta_batch["usage"]["total_tokens"]

        for doc, embedding in zip(documents, all_embeddings):
            doc.embedding = embedding

        return {"documents": documents, "meta": final_meta or {}}

    @component.output_types(documents=List[Document], meta=TextEmbedderMeta)
    async def run_async(self, documents: List[Document], prompt: Optional[str] = None) -> Dict[str, Any]:
        if not isinstance(documents, list) or (documents and not isinstance(documents[0], Document)):
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

        all_embeddings: List[Union[List[float], List[int], str]] = []
        final_meta: Optional[Dict[str, Any]] = None

        tasks = []
        for i in range(0, len(texts_to_embed), self.batch_size):
            batch_texts = texts_to_embed[i : i + self.batch_size]
            tasks.append(
                get_async_embedding_response(
                    async_client=self.async_client,
                    texts=batch_texts,
                    model=self.model,
                    normalized=self.normalized,
                    encoding_format=self.encoding_format,
                    dimensions=self.dimensions,
                    prompt=prompt or self.prompt,
                )
            )

        results = await asyncio.gather(*tasks)

        for embeddings_batch, meta_batch in results:
            all_embeddings.extend(embeddings_batch)
            if final_meta is None:
                final_meta = meta_batch
                final_meta["usage"] = meta_batch["usage"].copy()
            else:
                final_meta["usage"]["prompt_tokens"] += meta_batch["usage"]["prompt_tokens"]
                final_meta["usage"]["total_tokens"] += meta_batch["usage"]["total_tokens"]

        for doc, embedding in zip(documents, all_embeddings):
            doc.embedding = embedding

        return {"documents": documents, "meta": final_meta or {}}
