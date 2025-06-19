from typing import Any, Dict, List, Optional

from haystack import Document, component, default_from_dict, default_to_dict
from haystack.utils import Secret, deserialize_secrets_inplace
from mixedbread import AsyncMixedbread, Mixedbread


@component
class MixedbreadReranker:
    """
    Rerank documents using the Mixedbread Reranking API.

    Supports both synchronous and asynchronous reranking operations.
    """

    def __init__(
        self,
        api_key: Secret = Secret.from_env_var("MXBAI_API_KEY"),
        model: str = "mixedbread-ai/mxbai-rerank-large-v2",
        top_k: int = 10,
        return_input: Optional[bool] = False,
        base_url: Optional[str] = None,
        timeout: Optional[float] = 60.0,
        max_retries: Optional[int] = 2,
    ):
        """
        Initialize the MixedbreadReranker.

        Args:
            api_key: Mixedbread API key.
            model: Model name for document reranking.
            top_k: Maximum number of documents to return.
            return_input: Whether to return input documents in response.
            base_url: Optional custom API base URL.
            timeout: Request timeout in seconds.
            max_retries: Maximum retry attempts.
        """
        resolved_api_key = api_key.resolve_value()
        if not resolved_api_key:
            raise ValueError(
                "Mixedbread API key not found. Set MXBAI_API_KEY environment variable."
            )

        self.api_key = api_key
        self.model = model
        self.top_k = top_k
        self.return_input = return_input
        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries

        # Initialize clients
        self.client = Mixedbread(
            api_key=resolved_api_key,
            base_url=self.base_url,
            timeout=self.timeout,
            max_retries=self.max_retries,
        )

        self.aclient = AsyncMixedbread(
            api_key=resolved_api_key,
            base_url=self.base_url,
            timeout=self.timeout,
            max_retries=self.max_retries,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the reranker configuration."""
        return default_to_dict(
            self,
            api_key=self.api_key.to_dict(),
            model=self.model,
            top_k=self.top_k,
            return_input=self.return_input,
            base_url=self.base_url,
            timeout=self.timeout,
            max_retries=self.max_retries,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MixedbreadReranker":
        """Create reranker from dictionary."""
        deserialize_secrets_inplace(data["init_parameters"], keys=["api_key"])
        return default_from_dict(cls, data)

    def _create_metadata(self, response: Any, num_results: int) -> Dict[str, Any]:
        """Create metadata from API response."""
        return {
            "model": response.model,
            "usage": response.usage.model_dump(),
            "top_k": num_results,
        }

    def _prepare_texts(self, documents: List[Document]) -> List[str]:
        """Extract text content from documents."""
        return [doc.content or "" for doc in documents]

    @component.output_types(documents=List[Document], meta=Dict[str, Any])
    def run(self, documents: List[Document], query: str) -> Dict[str, Any]:
        """
        Rerank documents based on relevance to a query.

        Args:
            documents: List of documents to rerank.
            query: Query to rank documents against.

        Returns:
            Dictionary containing reranked documents and metadata.
        """
        if not documents:
            return {
                "documents": [],
                "meta": {
                    "model": self.model,
                    "usage": {"prompt_tokens": 0, "total_tokens": 0},
                    "top_k": 0,
                },
            }

        # Validate documents
        if not isinstance(documents, list) or (
            documents and not isinstance(documents[0], Document)
        ):
            raise TypeError("Input must be a list of Haystack Documents.")

        if not query.strip():
            return {
                "documents": documents,
                "meta": {
                    "model": self.model,
                    "usage": {"prompt_tokens": 0, "total_tokens": 0},
                    "top_k": len(documents),
                },
            }

        texts = self._prepare_texts(documents)

        response = self.client.rerank(
            model=self.model,
            query=query,
            input=texts,
            top_k=self.top_k,
            return_input=self.return_input,
        )

        # Create reranked documents with scores
        reranked_documents = []
        for result in response.data:
            original_doc = documents[result.index]
            # Add rerank score to document metadata
            original_doc.meta["rerank_score"] = result.score
            reranked_documents.append(original_doc)

        meta = self._create_metadata(response, len(reranked_documents))
        return {"documents": reranked_documents, "meta": meta}

    @component.output_types(documents=List[Document], meta=Dict[str, Any])
    async def run_async(self, documents: List[Document], query: str) -> Dict[str, Any]:
        """
        Asynchronously rerank documents based on relevance to a query.

        Args:
            documents: List of documents to rerank.
            query: Query to rank documents against.

        Returns:
            Dictionary containing reranked documents and metadata.
        """
        if not documents:
            return {
                "documents": [],
                "meta": {
                    "model": self.model,
                    "usage": {"prompt_tokens": 0, "total_tokens": 0},
                    "top_k": 0,
                },
            }

        # Validate documents
        if not isinstance(documents, list) or (
            documents and not isinstance(documents[0], Document)
        ):
            raise TypeError("Input must be a list of Haystack Documents.")

        if not query.strip():
            return {
                "documents": documents,
                "meta": {
                    "model": self.model,
                    "usage": {"prompt_tokens": 0, "total_tokens": 0},
                    "top_k": len(documents),
                },
            }

        texts = self._prepare_texts(documents)

        response = await self.aclient.rerank(
            model=self.model,
            query=query,
            input=texts,
            top_k=self.top_k,
            return_input=self.return_input,
        )

        # Create reranked documents with scores
        reranked_documents = []
        for result in response.data:
            original_doc = documents[result.index]
            # Add rerank score to document metadata
            original_doc.meta["rerank_score"] = result.score
            reranked_documents.append(original_doc)

        meta = self._create_metadata(response, len(reranked_documents))
        return {"documents": reranked_documents, "meta": meta}
