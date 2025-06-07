"""
Common utilities and shared components for mixedbread-ai-haystack.

This module provides shared functionality including:
- HTTP client for Mixedbread AI API
- Logging utilities
- Serialization mixins
- Response utilities and validation helpers
"""

from .client import MixedbreadClient
from .logging import get_logger
from .mixins import SerializationMixin
from .utils import (
    validate_documents,
    create_response_meta,
    create_empty_response,
    create_empty_documents_response,
    create_empty_embedding_response,
    create_empty_reranking_response,
)

__all__ = [
    "MixedbreadClient",
    "get_logger",
    "SerializationMixin",
    "validate_documents",
    "create_response_meta",
    "create_empty_response",
    "create_empty_documents_response",
    "create_empty_embedding_response", 
    "create_empty_reranking_response",
]
