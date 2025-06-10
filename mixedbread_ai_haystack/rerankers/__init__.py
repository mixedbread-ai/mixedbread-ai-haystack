"""
Document reranking components for Mixedbread AI.

This module provides Haystack components for reranking documents using
Mixedbread AI's reranking models to improve search relevance.
"""

from .reranker import MixedbreadReranker

__all__ = ["MixedbreadReranker"]
