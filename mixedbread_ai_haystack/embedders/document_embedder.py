import os
from typing import Any, Dict, List, Optional, Tuple

from haystack import Document, component, default_to_dict
from tqdm import tqdm

from mixedbread_ai import MixedbreadAi, models


class MixedbreadAiDocumentEmbedderMeta:
    """
    TypedDict for the meta attribute in MixedbreadAiDocumentEmbedder response.

    Attributes:
        model: The name of the model used for the embedding.
        usage: Detailed information about the usage of tokens.
        truncated: Whether the input text was truncated or not (if the text was too long for the model).
        document_meta: Information for each embedding, if it was truncated or not.
    """

    model: str
    usage: models.ModelUsage
    truncated: bool
    normalized: bool
    document_meta: List[Dict[str, Any]]


@component
class MixedbreadAiDocumentEmbedder:
    """
    A component for computing Document embeddings using Mixedbread AI models.
    The embedding of each Document is stored in the `embedding` field of the Document.

    Usage example:
    ```python
    from haystack import Document
    from mixedbread_haystack import MixedbreadAiDocumentEmbedder

    doc = Document(content="The quick brown fox jumps over the lazy dog")

    document_embedder = MixedbreadAiDocumentEmbedder()

    result = document_embedder.run([doc])
    print(result['documents'][0].embedding)

    # [0.015468, -0.028194, ...]
    ```
    """

    def __init__(
            self,
            model: str = "UAE-Large-V1",
            prefix: str = "",
            suffix: str = "",
            normalized: bool = False,
            batch_size: int = 128,
            progress_bar: bool = True,
            embedding_separator: str = "\n",
            instruction: Optional[str] = None,
            meta_fields_to_embed: Optional[List[str]] = None,
            api_key: Optional[str] = None,
            base_url: Optional[str] = "https://api.mixedbread.ai",
            custom_headers: Optional[Dict[str, str]] = None,
            timeout: Optional[float] = None,
            verify_ssl: Optional[bool] = None,
    ):
        """
        Create a MixedbreadAiDocumentEmbedder component.
        :param api_key: The mixedbread.ai API key. It can be explicitly provided or automatically read from the
            environment variable MIXEDBREAD_API_KEY.
        :param model: The name of the mixedbread.ai model to use. Check the list of available models on `https://mixedbread.ai/docs/models/embeddings/`
        :param prefix: A string to add to the beginning of each text.
        :param suffix: A string to add to the end of each text.
        :param normalized: Whether to normalize the embeddings or not.
        :param instruction: Instruction to show to the user when using the model.
        :param custom_headers: Custom headers to add to the requests sent to the mixedbread.ai API.
        :param batch_size: Number of Documents to encode at once.
        :param progress_bar: Whether to show a progress bar or not.
        :param meta_fields_to_embed: List of meta fields that should be embedded along with the Document text.
        :param embedding_separator: Separator used to concatenate the meta fields to the Document text.
        """

        api_key = api_key or os.environ.get("MIXEDBREAD_API_KEY")
        if not api_key:
            raise ValueError(
                "MixedbreadAiDocumentEmbedder requires an API key to be provided. "
                "Set the MIXEDBREAD_API_KEY environment variable (recommended) or pass it explicitly."
            )
        self.model_name = model
        self.prefix = prefix
        self.suffix = suffix
        self.batch_size = batch_size
        self.progress_bar = progress_bar
        self.meta_fields_to_embed = meta_fields_to_embed or []
        self.embedding_separator = embedding_separator
        self.instruction = instruction
        self.normalized = normalized
        self._client = MixedbreadAi(
            api_key=api_key,
            base_url=base_url,
            verify_ssl=verify_ssl,
            timeout=timeout,
            raise_for_status=True,
            headers={
                "User-Agent": "@mixedbread-ai/integrations-haystack",
                **(custom_headers or {}),
            },
        )

        # Other necessary imports remain the same

    def _get_telemetry_data(self) -> Dict[str, Any]:
        """
        Data that is sent for usage analytics, if applicable.
        """
        return {"model": self.model_name}

    def to_dict(self) -> Dict[str, Any]:
        """
        This method overrides the default serializer to avoid leaking the `api_key`.
        """
        return default_to_dict(
            self,
            model=self.model_name,
            prefix=self.prefix,
            suffix=self.suffix,
            batch_size=self.batch_size,
            progress_bar=self.progress_bar,
            meta_fields_to_embed=self.meta_fields_to_embed,
            embedding_separator=self.embedding_separator,
            instruction=self.instruction,
            normalized=self.normalized,
        )

    def _prepare_texts_to_embed(self, documents: List[Document]) -> List[str]:
        """
        Prepare texts to embed by concatenating Document text with metadata fields.
        """
        texts_to_embed = []
        for doc in documents:
            meta_values_to_embed = [
                str(doc.meta[key]) for key in self.meta_fields_to_embed if key in doc.meta and doc.meta[key] is not None
            ]
            text_to_embed = (
                    self.prefix + self.embedding_separator.join(
                [*meta_values_to_embed, doc.content or ""]) + self.suffix
            )
            texts_to_embed.append(text_to_embed)
        return texts_to_embed

    def _embed_batch(self, texts_to_embed: List[str], batch_size: int) -> Tuple[List[List[float]], Dict[str, Any]]:
        """
        Embed a list of texts in batches.
        """
        all_embeddings = []
        metadata = {}
        if not texts_to_embed:
            return all_embeddings, metadata

        for i in tqdm(
                range(0, len(texts_to_embed), batch_size), disable=not self.progress_bar, desc="Calculating embeddings"
        ):
            batch = texts_to_embed[i: i + batch_size]
            res = self._client.embeddings(
                model=self.model_name,
                input=batch,
                instruction=self.instruction,
                normalized=self.normalized,
            )
            if res is None:
                raise ValueError("MixedbreadAiDocumentEmbedder received an empty response.")
            if "message" in res:
                raise ValueError(
                    f"MixedbreadAiDocumentEmbedder recieved an unexpected response. Code: {res['code']} Message: {res['message']}")

            sorted_embeddings = sorted(res.data, key=lambda e: e.index)

            metadata = {
                "model": self.model_name,
                "usage": res.usage,
                "normalized": self.normalized,
                "document_meta": [],
            }

            embeddings = []
            for embedding in sorted_embeddings:
                metadata["document_meta"].append({
                    "truncated": embedding.truncated,
                    "index": embedding.index,
                })
                embeddings.append(embedding.embedding)
            all_embeddings.extend(embeddings)

        return all_embeddings, metadata

    @component.output_types(documents=List[Document], meta=Dict[str, Any])
    def run(self, documents: List[Document]):
        """
        Embed a list of Documents.
        """
        if not isinstance(documents, list) or (documents and not isinstance(documents[0], Document)):
            raise TypeError(
                "MixedbreadAiDocumentEmbedder expects a list of Documents as input. "
                "Please provide a valid list of Documents."
            )

        texts_to_embed = self._prepare_texts_to_embed(documents=documents)
        embeddings, metadata = self._embed_batch(texts_to_embed=texts_to_embed, batch_size=self.batch_size)

        for doc, emb in zip(documents, embeddings):
            doc.embedding = emb

        return {"documents": documents, "meta": metadata}
