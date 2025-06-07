"""Shared utilities for mixedbread-ai-haystack components."""

from typing import Dict, List, Any, Optional
from haystack import Document


def validate_documents(documents: List[Document]) -> None:
    """
    Validate that input is a list of Haystack Documents.
    
    Args:
        documents: List of documents to validate.
        
    Raises:
        TypeError: If input is not a list of Haystack Documents.
    """
    if not isinstance(documents, list) or (
        documents and not isinstance(documents[0], Document)
    ):
        raise TypeError("Input must be a list of Haystack Documents.")


def create_response_meta(
    response: Any,
    include_embedder_fields: bool = False,
    include_reranker_fields: bool = False
) -> Dict[str, Any]:
    """
    Create standardized response metadata from API response.
    
    Args:
        response: API response object with model, usage, etc.
        include_embedder_fields: Include embedder-specific fields (normalized, encoding_format, dimensions).
        include_reranker_fields: Include reranker-specific fields if any.
        
    Returns:
        Standardized metadata dictionary.
    """
    meta = {
        "model": response.model,
        "usage": response.usage.model_dump(),
        "object": response.object,
    }
    
    if include_embedder_fields:
        meta.update({
            "normalized": response.normalized,
            "encoding_format": response.encoding_format,
            "dimensions": response.dimensions,
        })
    
    return meta


def create_empty_response(
    response_type: str,
    model: str = "unknown"
) -> Dict[str, Any]:
    """
    Create standardized empty response for different component types.
    
    Args:
        response_type: Type of response ("embedding", "documents", "reranking").
        model: Model name to include in metadata.
        
    Returns:
        Empty response with appropriate structure.
    """
    empty_usage = {
        "prompt_tokens": 0,
        "total_tokens": 0,
    }
    
    base_meta = {
        "model": model,
        "usage": empty_usage,
        "object": "list",
    }
    
    if response_type == "embedding":
        return {
            "embedding": [],
            "meta": {
                **base_meta,
                "normalized": True,
                "encoding_format": "float",
                "dimensions": 0,
            }
        }
    elif response_type == "documents":
        return {
            "documents": [],
            "meta": {
                **base_meta,
                "normalized": True,
                "encoding_format": "float", 
                "dimensions": 0,
            }
        }
    elif response_type == "reranking":
        return {
            "documents": [],
            "meta": {
                **base_meta,
                "top_k": 0,
            }
        }
    else:
        raise ValueError(f"Unknown response type: {response_type}")


def create_empty_documents_response(model: str = "unknown") -> Dict[str, Any]:
    """Create empty response for document-based operations."""
    return create_empty_response("documents", model)


def create_empty_embedding_response(model: str = "unknown") -> Dict[str, Any]:
    """Create empty response for embedding operations."""
    return create_empty_response("embedding", model)


def create_empty_reranking_response(model: str = "unknown") -> Dict[str, Any]:
    """Create empty response for reranking operations."""
    return create_empty_response("reranking", model)