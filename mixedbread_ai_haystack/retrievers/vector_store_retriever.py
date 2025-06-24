from typing import Any, Dict, List, Literal, Optional

from haystack import Document, component, default_from_dict, default_to_dict
from haystack.utils import Secret, deserialize_secrets_inplace
from mixedbread import AsyncMixedbread, Mixedbread


@component
class MixedbreadVectorStoreRetriever:
    """
    Retrieve documents from Mixedbread vector stores.

    Supports both chunk-level and file-level search across multiple vector stores.
    """

    def __init__(
        self,
        vector_store_identifiers: List[str],
        api_key: Secret = Secret.from_env_var("MXBAI_API_KEY"),
        search_type: Literal["chunk", "file"] = "chunk",
        top_k: int = 10,
        score_threshold: Optional[float] = None,
        base_url: Optional[str] = None,
        timeout: Optional[float] = 60.0,
        max_retries: Optional[int] = 2,
    ):
        """
        Initialize the MixedbreadVectorStoreRetriever.

        Args:
            vector_store_identifiers: List of vector store identifiers to search.
            api_key: Mixedbread API key.
            search_type: Type of search - "chunk" for individual chunks, "file" for complete files.
            top_k: Maximum number of results to return.
            score_threshold: Minimum relevance score threshold.
            base_url: Optional custom API base URL.
            timeout: Request timeout in seconds.
            max_retries: Maximum retry attempts.
        """
        resolved_api_key = api_key.resolve_value()
        if not resolved_api_key:
            raise ValueError(
                "Mixedbread API key not found. Set MXBAI_API_KEY environment variable."
            )

        if not vector_store_identifiers:
            raise ValueError("At least one vector store identifier must be provided.")

        self.vector_store_identifiers = vector_store_identifiers
        self.api_key = api_key
        self.search_type = search_type
        self.top_k = top_k
        self.score_threshold = score_threshold
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
        """Serialize the retriever configuration."""
        return default_to_dict(
            self,
            vector_store_identifiers=self.vector_store_identifiers,
            api_key=self.api_key.to_dict(),
            search_type=self.search_type,
            top_k=self.top_k,
            score_threshold=self.score_threshold,
            base_url=self.base_url,
            timeout=self.timeout,
            max_retries=self.max_retries,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MixedbreadVectorStoreRetriever":
        """Create retriever from dictionary."""
        deserialize_secrets_inplace(data["init_parameters"], keys=["api_key"])
        return default_from_dict(cls, data)

    def _extract_content(self, item: Any) -> str:
        """Extract content from search result item."""
        # Handle chunk objects with different content types
        if hasattr(item, "type"):
            chunk_type = item.type
            if chunk_type == "text" and hasattr(item, "text"):
                return item.text
            elif chunk_type == "image_url" and hasattr(item, "ocr_text"):
                return item.ocr_text if item.ocr_text else getattr(item, "summary", "")
            elif chunk_type == "audio_url" and hasattr(item, "transcription"):
                return item.transcription if item.transcription else getattr(item, "summary", "")
            elif hasattr(item, "summary"):
                return item.summary
        
        # Fallback for direct text content
        if hasattr(item, "text"):
            return item.text
        
        return ""

    def _create_metadata(self, item: Any, search_type: str) -> Dict[str, Any]:
        """Create metadata from search result item."""
        metadata = {
            "retrieval_score": getattr(item, "score", 0.0),
            "search_type": search_type,
        }

        # Add chunk-specific metadata
        if hasattr(item, "chunk_index"):
            metadata["chunk_index"] = item.chunk_index
        
        # Add file metadata  
        if hasattr(item, "filename"):
            metadata["filename"] = item.filename
        if hasattr(item, "file_id"):
            metadata["file_id"] = item.file_id
        if hasattr(item, "id"):
            metadata["file_id"] = item.id

        # Add custom metadata if present
        if hasattr(item, "metadata") and item.metadata:
            metadata.update(item.metadata)

        return metadata


    def _convert_results_to_documents(self, results: List[Any]) -> List[Document]:
        """Convert API results to Haystack Documents."""
        documents = []

        for item in results:
            if self.search_type == "chunk":
                # Direct chunk content
                content = self._extract_content(item)
                metadata = self._create_metadata(item, self.search_type)
                documents.append(Document(content=content, meta=metadata))
            else:
                # File search - combine chunk content
                if hasattr(item, "chunks") and item.chunks:
                    chunk_texts = []
                    for chunk in item.chunks:
                        chunk_content = self._extract_content(chunk)
                        if chunk_content.strip():
                            chunk_texts.append(chunk_content)
                    
                    content = "\n\n".join(chunk_texts) if chunk_texts else f"[File: {getattr(item, 'filename', 'Unknown file')} - No extractable content]"
                else:
                    content = f"[File: {getattr(item, 'filename', 'Unknown file')} - No chunks returned]"
                
                metadata = self._create_metadata(item, self.search_type)
                documents.append(Document(content=content, meta=metadata))

        return documents

    @component.output_types(documents=List[Document])
    def run(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        top_k: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Retrieve documents from vector stores.

        Args:
            query: Search query string.
            filters: Optional runtime filters (not used by Mixedbread API).
            top_k: Override default top_k for this search.

        Returns:
            Dictionary containing retrieved documents and metadata.
        """
        if not query.strip():
            return {"documents": []}

        effective_top_k = top_k or self.top_k

        try:
            search_request = {
                "query": query,
                "vector_store_identifiers": self.vector_store_identifiers,
                "top_k": effective_top_k,
                "search_options": {"return_metadata": True},
            }

            if self.search_type == "chunk":
                response = self.client.vector_stores.search(**search_request)
            else:  # file search
                search_request["search_options"]["return_chunks"] = True
                response = self.client.vector_stores.files.search(**search_request)

            # Extract results
            results = response.data if hasattr(response, "data") else []
            
            # Apply score threshold filtering client-side if specified
            if self.score_threshold is not None:
                results = [r for r in results if getattr(r, "score", 0.0) >= self.score_threshold]

            # Convert to documents
            documents = self._convert_results_to_documents(results)

            return {
                "documents": documents,
                "meta": {
                    "query": query,
                    "search_type": self.search_type,
                    "vector_stores": self.vector_store_identifiers,
                    "total_results": len(documents),
                    "top_k": effective_top_k,
                },
            }

        except Exception as e:
            # Return empty results on error
            return {
                "documents": [],
                "meta": {
                    "query": query,
                    "search_type": self.search_type,
                    "vector_stores": self.vector_store_identifiers,
                    "error": str(e),
                    "total_results": 0,
                },
            }

    @component.output_types(documents=List[Document])
    async def run_async(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        top_k: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Asynchronously retrieve documents from vector stores.

        Args:
            query: Search query string.
            filters: Optional runtime filters (not used by Mixedbread API).
            top_k: Override default top_k for this search.

        Returns:
            Dictionary containing retrieved documents and metadata.
        """
        if not query.strip():
            return {"documents": []}

        effective_top_k = top_k or self.top_k

        try:
            search_request = {
                "query": query,
                "vector_store_identifiers": self.vector_store_identifiers,
                "top_k": effective_top_k,
                "search_options": {"return_metadata": True},
            }

            if self.search_type == "chunk":
                response = await self.aclient.vector_stores.search(**search_request)
            else:  # file search
                search_request["search_options"]["return_chunks"] = True
                response = await self.aclient.vector_stores.files.search(**search_request)

            # Extract results
            results = response.data if hasattr(response, "data") else []
            
            # Apply score threshold filtering client-side if specified
            if self.score_threshold is not None:
                results = [r for r in results if getattr(r, "score", 0.0) >= self.score_threshold]

            # Convert to documents
            documents = self._convert_results_to_documents(results)

            return {
                "documents": documents,
                "meta": {
                    "query": query,
                    "search_type": self.search_type,
                    "vector_stores": self.vector_store_identifiers,
                    "total_results": len(documents),
                    "top_k": effective_top_k,
                },
            }

        except Exception as e:
            # Return empty results on error
            return {
                "documents": [],
                "meta": {
                    "query": query,
                    "search_type": self.search_type,
                    "vector_stores": self.vector_store_identifiers,
                    "error": str(e),
                    "total_results": 0,
                },
            }
