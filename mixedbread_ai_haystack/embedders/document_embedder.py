from typing import Optional, List, Dict, Any
import concurrent

from haystack import component, Document, default_to_dict
from mixedbread_ai import EmbeddingsResponse, ObjectType, Usage

from mixedbread_ai_haystack.embedders import MixedbreadAITextEmbedder
from mixedbread_ai_haystack.embedders.text_embedder import EmbedderMeta

try:
    from tqdm import tqdm
    has_progress_bar = True
except ImportError:
    has_progress_bar = False
    def tqdm(iterable, *args, **kwargs):
        return iterable

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
            max_concurrency: int = 1,
            meta_fields_to_embed: Optional[List[str]] = None,
            **kwargs
    ):
        super(MixedbreadAIDocumentEmbedder, self).__init__(**kwargs)

        if max_concurrency < 1:
            raise ValueError("The max_concurrency parameter must be greater than 0.")
        if batch_size < 1 or batch_size > 256:
            raise ValueError("The batch_size parameter must be between 1 and 256.")
        if not has_progress_bar and show_progress_bar:
            raise ImportError(
                "The package 'tqdm' must be installed to use the progress bar. "
                "You can install it via 'pip install tqdm'."
            )

        self.batch_size = batch_size
        self.show_progress_bar = show_progress_bar
        self.embedding_separator = embedding_separator
        self.meta_fields_to_embed = meta_fields_to_embed or []
        self.max_concurrency = max_concurrency

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

    def _documents_to_texts(self, docs: List[Document]) -> List[str]:
        """
        Converts a list of documents to a list of strings to be embedded.

        Parameters:
            docs (List[Document]): The documents to convert.

        Returns:
            List[str]: The converted list of strings.
        """
        def prepare_doc(doc: Document) -> str:
            values_to_embed = [
                str(doc.meta[key])
                for key in self.meta_fields_to_embed
                if key in doc.meta and doc.meta[key] is not None
            ]
            if doc.content:
                values_to_embed.append(doc.content)

            return self.embedding_separator.join(values_to_embed)

        return [self.prefix + prepare_doc(doc) + self.suffix for doc in docs]

    def _calculate_embeddings_single_thread(self, texts_to_embed: List[str]) -> List[EmbeddingsResponse]:
        responses = []
        batch_iter = tqdm(
            range(0, len(texts_to_embed), self.batch_size),
            disable=not self.show_progress_bar,
            desc="MixedbreadAIDocumentEmbedder - Calculating embedding batches"
        )

        for i in batch_iter:
            batch = texts_to_embed[i:i + self.batch_size]
            response = self._client.embeddings(
                model=self.model,
                input=batch,
                normalized=self.normalized,
                encoding_format=self.encoding_format,
                truncation_strategy=self.truncation_strategy,
                dimensions=self.dimensions,
                prompt=self.prompt,
                request_options=self._request_options
            )
            responses.append(response)
        return responses

    def _calculate_embeddings_multi_thread(self, texts_to_embed: List[str]):
        def process_batch(batch):
            return self._client.embeddings(
                model=self.model,
                input=batch,
                normalized=self.normalized,
                encoding_format=self.encoding_format,
                truncation_strategy=self.truncation_strategy,
                dimensions=self.dimensions,
                prompt=self.prompt,
                request_options=self._request_options
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_concurrency) as executor:
            futures = [executor.submit(process_batch, texts_to_embed[i:i + self.batch_size]) 
                       for i in range(0, len(texts_to_embed), self.batch_size)]
            futures_iterator = tqdm(
                concurrent.futures.as_completed(futures),
                total=len(futures),
                desc="MixedbreadAIDocumentEmbedder - Calculating embedding batches",
                disable=not self.show_progress_bar
            )
            responses = [future.result() for future in futures_iterator]

        return responses

    @component.output_types(documents=List[Document], meta=EmbedderMeta)
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
            return {
                "documents": [],
                "meta": EmbedderMeta(
                    model=self.model,
                    object=ObjectType.LIST,
                    normalized=self.normalized,
                    encoding_format=self.encoding_format,
                    dimensions=self.dimensions,
                    usage=Usage(
                        prompt_tokens=0,
                        total_tokens=0
                    )
                )
            }

        texts_to_embed = self._documents_to_texts(documents)

        if self.max_concurrency == 1:
            responses = self._calculate_embeddings_single_thread(texts_to_embed)
        else:
            responses = self._calculate_embeddings_multi_thread(texts_to_embed)

        embeddings = [item.embedding for response in responses for item in response.data]
        for doc, embedding in zip(documents, embeddings):
            doc.embedding = embedding

        meta = responses[0].dict(exclude={"data", "usage"})
        meta["usage"] = Usage(
            prompt_tokens=sum(response.usage.prompt_tokens for response in responses),
            total_tokens=sum(response.usage.total_tokens for response in responses),
        )

        return {"documents": documents, "meta": EmbedderMeta(**meta)}
