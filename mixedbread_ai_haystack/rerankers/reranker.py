from typing import Any, Dict, List, Optional
from haystack import Document, component
from haystack.utils import Secret

from mixedbread_ai_haystack.common.client import MixedbreadClient
from mixedbread_ai_haystack.common.mixins import SerializationMixin
from mixedbread_ai_haystack.common.utils import (
    validate_documents,
    create_response_meta,
    create_empty_reranking_response
)
from mixedbread_ai_haystack.common.logging import get_logger

logger = get_logger(__name__)


@component
class MixedbreadReranker(SerializationMixin, MixedbreadClient):
    """
    Rerank documents using the Mixedbread Reranking API.
    """

    def __init__(
        self,
        api_key: Secret = Secret.from_env_var("MXBAI_API_KEY"),
        model: str = "mixedbread-ai/mxbai-rerank-large-v1",
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
        super(MixedbreadReranker, self).__init__(
            api_key=api_key, base_url=base_url, timeout=timeout, max_retries=max_retries
        )
        self.model = model
        self.top_k = top_k
        self.return_input = return_input

    def to_dict(self) -> Dict[str, Any]:
        from haystack import default_to_dict
        
        client_params = MixedbreadClient.to_dict(self)["init_parameters"]
        return default_to_dict(
            self,
            **client_params,
            model=self.model,
            top_k=self.top_k,
            return_input=self.return_input,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MixedbreadReranker":
        from haystack import default_from_dict
        from haystack.utils import deserialize_secrets_inplace
        
        deserialize_secrets_inplace(data["init_parameters"], keys=["api_key"])
        return default_from_dict(cls, data)

    @component.output_types(documents=List[Document], meta=Dict[str, Any])
    def run(self, documents: List[Document], query: str) -> Dict[str, Any]:
        """
        Rerank documents based on relevance to a query.
        
        Args:
            documents: List of documents to rerank.
            query: Query to rank documents against.
            
        Returns:
            Dictionary containing reranked documents and metadata.
            
        Raises:
            Exception: If the reranking request fails.
        """
        validate_documents(documents)
        
        if not documents:
            logger.info("Empty document list provided for reranking")
            return create_empty_reranking_response(self.model)
            
        if not query.strip():
            logger.warning("Empty query provided for reranking")
            return {"documents": documents, "meta": create_empty_reranking_response(self.model)["meta"]}

        try:
            # Prepare texts for reranking
            texts_to_rerank = [doc.content or "" for doc in documents]

            response = self.client.rerank(
                model=self.model,
                query=query,
                input=texts_to_rerank,
                top_k=self.top_k,
                return_input=self.return_input,
            )

            reranked_documents = []
            for result in response.data:
                original_doc = documents[result.index]
                original_doc.meta["rerank_score"] = result.score
                reranked_documents.append(original_doc)

            meta = create_response_meta(response, include_reranker_fields=True)
            meta["top_k"] = len(reranked_documents)

            return {"documents": reranked_documents, "meta": meta}
            
        except Exception as e:
            logger.error(f"Error during document reranking: {str(e)}")
            raise

    @component.output_types(documents=List[Document], meta=Dict[str, Any])
    async def run_async(self, documents: List[Document], query: str) -> Dict[str, Any]:
        """
        Asynchronously rerank documents based on relevance to a query.
        
        Args:
            documents: List of documents to rerank.
            query: Query to rank documents against.
            
        Returns:
            Dictionary containing reranked documents and metadata.
            
        Raises:
            Exception: If the reranking request fails.
        """
        validate_documents(documents)
        
        if not documents:
            logger.info("Empty document list provided for reranking")
            return create_empty_reranking_response(self.model)
            
        if not query.strip():
            logger.warning("Empty query provided for reranking")
            return {"documents": documents, "meta": create_empty_reranking_response(self.model)["meta"]}

        try:
            # Prepare texts for reranking
            texts_to_rerank = [doc.content or "" for doc in documents]

            response = await self.async_client.rerank(
                model=self.model,
                query=query,
                input=texts_to_rerank,
                top_k=self.top_k,
                return_input=self.return_input,
            )

            reranked_documents = []
            for result in response.data:
                original_doc = documents[result.index]
                original_doc.meta["rerank_score"] = result.score
                reranked_documents.append(original_doc)

            meta = create_response_meta(response, include_reranker_fields=True)
            meta["top_k"] = len(reranked_documents)

            return {"documents": reranked_documents, "meta": meta}
            
        except Exception as e:
            logger.error(f"Error during async document reranking: {str(e)}")
            raise
