from typing import Any, Dict, List, Optional

from haystack import component, default_from_dict, default_to_dict
from haystack.utils import Secret, deserialize_secrets_inplace
from mixedbread import AsyncMixedbread, Mixedbread


@component
class MixedbreadTextEmbedder:
    """
    Embed a single string using the Mixedbread Embeddings API.

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
        Initialize the MixedbreadTextEmbedder.

        Args:
            api_key: Mixedbread API key.
            model: Model name for text embedding.
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
            raise ValueError(
                "Mixedbread API key not found. Set MXBAI_API_KEY environment variable."
            )

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
    def from_dict(cls, data: Dict[str, Any]) -> "MixedbreadTextEmbedder":
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

    @component.output_types(embedding=List[float], meta=Dict[str, Any])
    def run(self, text: str, prompt: Optional[str] = None) -> Dict[str, Any]:
        """
        Embed a single text string.

        Args:
            text: Text to embed.
            prompt: Optional prompt to override the default.

        Returns:
            Dictionary containing embedding vector and metadata.
        """
        if not text.strip():
            return {
                "embedding": [],
                "meta": {
                    "model": self.model,
                    "usage": {"prompt_tokens": 0, "total_tokens": 0},
                    "normalized": self.normalized,
                    "encoding_format": self.encoding_format,
                    "dimensions": 0,
                },
            }

        response = self.client.embed(
            model=self.model,
            input=[text],
            normalized=self.normalized,
            encoding_format=self.encoding_format,
            dimensions=self.dimensions,
            prompt=prompt or self.prompt,
        )

        embedding = response.data[0].embedding if response.data else []
        meta = self._create_metadata(response)

        return {"embedding": embedding, "meta": meta}

    @component.output_types(embedding=List[float], meta=Dict[str, Any])
    async def run_async(
        self, text: str, prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Asynchronously embed a single text string.

        Args:
            text: Text to embed.
            prompt: Optional prompt to override the default.

        Returns:
            Dictionary containing embedding vector and metadata.
        """
        if not text.strip():
            return {
                "embedding": [],
                "meta": {
                    "model": self.model,
                    "usage": {"prompt_tokens": 0, "total_tokens": 0},
                    "normalized": self.normalized,
                    "encoding_format": self.encoding_format,
                    "dimensions": 0,
                },
            }

        response = await self.aclient.embed(
            model=self.model,
            input=[text],
            normalized=self.normalized,
            encoding_format=self.encoding_format,
            dimensions=self.dimensions,
            prompt=prompt or self.prompt,
        )

        embedding = response.data[0].embedding if response.data else []
        meta = self._create_metadata(response)

        return {"embedding": embedding, "meta": meta}
