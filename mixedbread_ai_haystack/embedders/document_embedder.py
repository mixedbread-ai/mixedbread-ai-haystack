from typing import Any, Dict, List, Optional

from haystack import Document, component, default_from_dict, default_to_dict
from haystack.utils import Secret, deserialize_secrets_inplace
from mixedbread import AsyncMixedbread, Mixedbread


@component
class MixedbreadDocumentEmbedder:
    """
    Embed multiple documents using the Mixedbread Embeddings API.
    
    Supports both synchronous and asynchronous embedding operations.
    """

    def __init__(
        self,
        api_key: Secret = Secret.from_env_var("MXBAI_API_KEY"),
        model: str = "mixedbread-ai/mxbai-embed-large-v1",
        normalized: bool = True,
        encoding_format: str = "float",
        dimensions: Optional[int] = None,
        prompt: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: Optional[float] = 60.0,
        max_retries: Optional[int] = 2,
    ):
        """
        Initialize the MixedbreadDocumentEmbedder.
        
        Args:
            api_key: Mixedbread API key.
            model: Model name for document embedding.
            normalized: Whether to normalize embeddings.
            encoding_format: Format for returned embeddings.
            dimensions: Target embedding dimensions.
            prompt: Optional prompt to customize embedding behavior.
            base_url: Optional custom API base URL.
            timeout: Request timeout in seconds.
            max_retries: Maximum retry attempts.
        """
        resolved_api_key = api_key.resolve_value()
        if not resolved_api_key:
            raise ValueError("Mixedbread API key not found. Set MXBAI_API_KEY environment variable.")

        self.api_key = api_key
        self.model = model
        self.normalized = normalized
        self.encoding_format = encoding_format
        self.dimensions = dimensions
        self.prompt = prompt
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
        """Serialize the embedder configuration."""
        return default_to_dict(
            self,
            api_key=self.api_key.to_dict(),
            model=self.model,
            normalized=self.normalized,
            encoding_format=self.encoding_format,
            dimensions=self.dimensions,
            prompt=self.prompt,
            base_url=self.base_url,
            timeout=self.timeout,
            max_retries=self.max_retries,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MixedbreadDocumentEmbedder":
        """Create embedder from dictionary."""
        deserialize_secrets_inplace(data["init_parameters"], keys=["api_key"])
        return default_from_dict(cls, data)

    def _create_metadata(self, response: Any) -> Dict[str, Any]:
        """Create metadata from API response."""
        return {
            "model": response.model,
            "usage": response.usage.model_dump(),
            "normalized": response.normalized,
            "encoding_format": response.encoding_format,
            "dimensions": response.dimensions,
        }

    def _prepare_texts(self, documents: List[Document]) -> List[str]:
        """Extract text content from documents."""
        return [doc.content or "" for doc in documents]

    @component.output_types(documents=List[Document], meta=Dict[str, Any])
    def run(self, documents: List[Document], prompt: Optional[str] = None) -> Dict[str, Any]:
        """
        Embed multiple documents and attach embeddings to them.
        
        Args:
            documents: List of Haystack documents to embed.
            prompt: Optional prompt to override the default.
            
        Returns:
            Dictionary containing documents with embeddings and metadata.
        """
        if not documents:
            return {
                "documents": [],
                "meta": {
                    "model": self.model,
                    "usage": {"prompt_tokens": 0, "total_tokens": 0},
                    "normalized": self.normalized,
                    "encoding_format": self.encoding_format,
                    "dimensions": 0,
                }
            }

        # Validate documents
        if not isinstance(documents, list) or (documents and not isinstance(documents[0], Document)):
            raise TypeError("Input must be a list of Haystack Documents.")

        texts = self._prepare_texts(documents)
        
        response = self.client.embed(
            model=self.model,
            input=texts,
            normalized=self.normalized,
            encoding_format=self.encoding_format,
            dimensions=self.dimensions,
            prompt=prompt or self.prompt,
        )

        # Attach embeddings to documents
        embeddings = [item.embedding for item in response.data] if response.data else []
        for doc, embedding in zip(documents, embeddings):
            doc.embedding = embedding

        meta = self._create_metadata(response)
        return {"documents": documents, "meta": meta}

    @component.output_types(documents=List[Document], meta=Dict[str, Any])
    async def run_async(self, documents: List[Document], prompt: Optional[str] = None) -> Dict[str, Any]:
        """
        Asynchronously embed multiple documents and attach embeddings to them.
        
        Args:
            documents: List of Haystack documents to embed.
            prompt: Optional prompt to override the default.
            
        Returns:
            Dictionary containing documents with embeddings and metadata.
        """
        if not documents:
            return {
                "documents": [],
                "meta": {
                    "model": self.model,
                    "usage": {"prompt_tokens": 0, "total_tokens": 0},
                    "normalized": self.normalized,
                    "encoding_format": self.encoding_format,
                    "dimensions": 0,
                }
            }

        # Validate documents
        if not isinstance(documents, list) or (documents and not isinstance(documents[0], Document)):
            raise TypeError("Input must be a list of Haystack Documents.")

        texts = self._prepare_texts(documents)
        
        response = await self.aclient.embed(
            model=self.model,
            input=texts,
            normalized=self.normalized,
            encoding_format=self.encoding_format,
            dimensions=self.dimensions,
            prompt=prompt or self.prompt,
        )

        # Attach embeddings to documents
        embeddings = [item.embedding for item in response.data] if response.data else []
        for doc, embedding in zip(documents, embeddings):
            doc.embedding = embedding

        meta = self._create_metadata(response)
        return {"documents": documents, "meta": meta}