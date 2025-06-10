from typing import Optional, List, Dict, Any, Union

from haystack import component, Document
from haystack.utils import Secret

from mixedbread_ai_haystack.common.client import MixedbreadClient
from mixedbread_ai_haystack.common.mixins import SerializationMixin
from mixedbread_ai_haystack.common.utils import (
    validate_documents, 
    create_response_meta, 
    create_empty_documents_response
)
from mixedbread_ai_haystack.common.logging import get_logger
from mixedbread_ai_haystack.embedders.embedding_types import MixedbreadEmbeddingType
from mixedbread_ai_haystack.embedders.text_embedder import TextEmbedderMeta

logger = get_logger(__name__)


@component
class MixedbreadDocumentEmbedder(SerializationMixin, MixedbreadClient):
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
        from haystack import default_to_dict
        
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
        from haystack import default_from_dict
        from haystack.utils import deserialize_secrets_inplace
        
        deserialize_secrets_inplace(data["init_parameters"], keys=["api_key"])
        ef_val = data["init_parameters"].get("encoding_format")
        if isinstance(ef_val, str):
            data["init_parameters"]["encoding_format"] = (
                MixedbreadEmbeddingType.from_str(ef_val)
            )
        return default_from_dict(cls, data)

    def _prepare_texts_to_embed(self, documents: List[Document]) -> List[str]:
        """
        Extract text content from documents for embedding.
        
        Args:
            documents: List of Haystack documents.
            
        Returns:
            List of text strings to embed.
        """
        texts = []
        for doc in documents:
            content_to_embed = doc.content or ""
            texts.append(content_to_embed)
        return texts

    def _run_impl(self, documents: List[Document], prompt: Optional[str] = None, use_async: bool = False):
        """
        Internal implementation for both sync and async document embedding.
        
        Args:
            documents: List of Haystack documents to embed.
            prompt: Optional prompt to override the default.
            use_async: Whether to use async client.
            
        Returns:
            Dictionary containing documents with embeddings and metadata, or awaitable if async.
            
        Raises:
            Exception: If the embedding request fails.
        """
        validate_documents(documents)
        
        if not documents:
            logger.info("Empty document list provided")
            result = create_empty_documents_response(self.model)
            if use_async:
                async def _return_empty():
                    return result
                return _return_empty()
            return result

        try:
            texts_to_embed = self._prepare_texts_to_embed(documents)
            client = self.async_client if use_async else self.client
            
            embed_call = client.embed(
                model=self.model,
                input=texts_to_embed,
                normalized=self.normalized,
                encoding_format=self.encoding_format.value,
                dimensions=self.dimensions,
                prompt=prompt or self.prompt,
            )
            
            if use_async:
                # Return the awaitable coroutine for async processing
                async def _process_async_response():
                    response = await embed_call
                    embeddings = [item.embedding for item in response.data] if response.data else []
                    meta = create_response_meta(response, include_embedder_fields=True)
                    
                    for doc, embedding in zip(documents, embeddings):
                        doc.embedding = embedding
                    
                    return {"documents": documents, "meta": meta}
                return _process_async_response()
            else:
                # Process sync response immediately
                response = embed_call
                embeddings = [item.embedding for item in response.data] if response.data else []
                meta = create_response_meta(response, include_embedder_fields=True)
                
                for doc, embedding in zip(documents, embeddings):
                    doc.embedding = embedding
                
                return {"documents": documents, "meta": meta}
            
        except Exception as e:
            error_msg = f"Error during {'async ' if use_async else ''}document embedding: {str(e)}"
            logger.error(error_msg)
            raise

    @component.output_types(documents=List[Document], meta=TextEmbedderMeta)
    def run(
        self, documents: List[Document], prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Embed multiple documents and attach embeddings to them.
        
        Args:
            documents: List of Haystack documents to embed.
            prompt: Optional prompt to override the default.
            
        Returns:
            Dictionary containing documents with embeddings and metadata.
            
        Raises:
            Exception: If the embedding request fails.
        """
        return self._run_impl(documents, prompt, use_async=False)

    @component.output_types(documents=List[Document], meta=TextEmbedderMeta)
    async def run_async(
        self, documents: List[Document], prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Asynchronously embed multiple documents and attach embeddings to them.
        
        Args:
            documents: List of Haystack documents to embed.
            prompt: Optional prompt to override the default.
            
        Returns:
            Dictionary containing documents with embeddings and metadata.
            
        Raises:
            Exception: If the embedding request fails.
        """
        return await self._run_impl(documents, prompt, use_async=True)
