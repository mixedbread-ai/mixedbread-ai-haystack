from typing import Optional, List, Dict, Any

from haystack import component, Document, default_to_dict
from mixedbread_ai import Usage
from tqdm.auto import tqdm

from mixedbread_ai_haystack.embedders import MixedbreadAITextEmbedder


@component
class MixedbreadAIDocumentEmbedder(MixedbreadAITextEmbedder):
    """
    A component for generating document embeddings using mixedbread ai's embedding API.
    The embedding of each Document is stored in the `embedding` field of the Document.

    Find out more at https://mixedbread.ai/docs

    This implementation uses the embeddings API.

    To use this you'll need a mixedbread ai API key - either pass it to
    the api_key parameter or set the MXBAI_API_KEY environment variable.

    API keys are available on https://mixedbread.ai - it's free to sign up and trial API
    keys work with this implementation.

    Usage example:
    ```python
    from haystack import Document
    from mixedbread_ai_haystack import MixedbreadAIDocumentEmbedder

    doc = Document(content="The quick brown fox jumps over the lazy dog")

    document_embedder = MixedbreadAIDocumentEmbedder()

    result = document_embedder.run([doc])
    print(result['documents'][0].embedding)
    ```

    Attributes:
        batch_size (int): The size of batches for processing documents.
        show_progress_bar (bool): Whether to show a progress bar during embedding computation.
        embedding_separator (str): The separator to use between different parts of the document when embedding.
        meta_fields_to_embed (Optional[List[str]]): List of metadata fields to include in the embedding process.
    """

    def __init__(
            self,
            batch_size: int = 128,
            show_progress_bar: bool = False,
            embedding_separator: str = "\n",
            meta_fields_to_embed: Optional[List[str]] = None,
            **kwargs
    ):
        super(MixedbreadAIDocumentEmbedder, self).__init__(**kwargs)

        self.batch_size = batch_size
        self.show_progress_bar = show_progress_bar
        self.embedding_separator = embedding_separator
        self.meta_fields_to_embed = meta_fields_to_embed or []

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes the component to a dictionary.

        Returns:
            Dict[str, Any]: The serialized component data.
        """
        parent_params = super(MixedbreadAIDocumentEmbedder, self).to_dict()["init_parameters"]

        return default_to_dict(
            self,
            **parent_params,
            batch_size=self.batch_size,
            show_progress_bar=self.show_progress_bar,
            embedding_separator=self.embedding_separator,
            meta_fields_to_embed=self.meta_fields_to_embed
        )

    def from_docs_to_texts(self, docs: List[Document]) -> List[str]:
        """
        Converts a list of documents to a list of strings to be embedded.

        Parameters:
            docs (List[Document]): The documents to convert.

        Returns:
            List[str]: The converted list of strings.
        """
        def prepare_doc(doc: Document) -> str:
            meta_values = [
                str(doc.meta[key])
                for key in self.meta_fields_to_embed
                if key in doc.meta and doc.meta[key] is not None
            ]

            return self.embedding_separator.join([*meta_values, doc.content or ""])

        return [self.prefix + prepare_doc(doc) + self.suffix for doc in docs]

    @component.output_types(documents=List[Document], meta=Dict[str, Any])
    def run(self, documents: List[Document], prompt: Optional[str] = None) -> Dict[str, Any]:
        """
        Computes embeddings for a list of documents.

        Parameters:
            documents (List[Document]): The list of documents to embed.
            prompt (Optional[str]): An optional prompt to use with the embedding model.

        Returns:
            Dict[str, Any]: A dictionary containing the embeddings and metadata.

        Raises:
            TypeError: If the input is not a list of Documents.
        """
        if not isinstance(documents, list) or documents and not isinstance(documents[0], Document):
            raise TypeError(
                "MixedbreadAIDocumentEmbedder expects a list of Documents as input."
                " In case you want to embed a string, please use the MixedbreadAITextEmbedder."
            )

        if len(documents) == 0:
            return {}

        texts_to_embed = self.from_docs_to_texts(documents)

        batch_iter = tqdm(
            range(0, len(texts_to_embed), self.batch_size),
            disable=not self.show_progress_bar,
            desc="Calculating embeddings"
        )

        responses = [
            self._client.embeddings(
                model=self.model,
                input=texts_to_embed[i:i + self.batch_size],
                normalized=self.normalized,
                encoding_format=self.encoding_format,
                truncation_strategy=self.truncation_strategy,
                dimensions=self.dimensions,
                prompt=prompt or self.prompt,
                request_options=self._request_options
            ) for i in batch_iter
        ]

        embeddings = [item.embedding for response in responses for item in response.data]

        for doc, embedding in zip(documents, embeddings):
            doc.embedding = embedding

        meta = responses[0].dict(exclude={"data", "usage"})
        meta["usage"] = Usage(
            prompt_tokens=sum(response.usage.prompt_tokens for response in responses),
            total_tokens=sum(response.usage.total_tokens for response in responses),
        ).dict()

        return {"documents": documents, "meta": meta}
