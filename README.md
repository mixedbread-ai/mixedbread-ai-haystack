# Mixedbread AI Haystack Integration

[![PyPI version](https://badge.fury.io/py/mixedbread-ai-haystack.svg)](https://badge.fury.io/py/mixedbread-ai-haystack)
[![Python versions](https://img.shields.io/pypi/pyversions/mixedbread-ai-haystack.svg)](https://pypi.org/project/mixedbread-ai-haystack/)

**Mixedbread AI** integration for **Haystack 2.2.1**, providing state-of-the-art embedding, reranking and parsing.

## Overview

[Mixedbread](https://www.mixedbread.com) provides best-in-class embedding and reranking models, both open-source and proprietary. This integration brings four powerful components to your Haystack pipelines:

- **MixedbreadTextEmbedder** - For embedding single texts and queries
- **MixedbreadDocumentEmbedder** - For embedding documents with metadata support  
- **MixedbreadReranker** - For reranking documents by relevance
- **MixedbreadDocumentParser** - For parsing and extracting structured content from various file formats

More information can be found in the [official documentation](https://www.mixedbread.com/docs).


## Installation

```bash
pip install mixedbread-ai-haystack
```

## Quick Start

### 1. Get your API key
Sign up at [mixedbread.com](https://www.mixedbread.com) and get your API key from the [dashboard](https://www.platform.mixedbread.com/).

### 2. Store your API key in an environment variable

### 3. Basic usage
```python
from mixedbread_ai_haystack import MixedbreadTextEmbedder

embedder = MixedbreadTextEmbedder(model="mixedbread-ai/mxbai-embed-large-v1")

result = embedder.run(text="What is the capital of France?")
embedding = result["embedding"]
```

## Components

### MixedbreadTextEmbedder
- **Purpose**: Embed single texts, queries, and short passages
- **Input**: String
- **Output**: List of float embeddings
- **Use case**: Query encoding in retrieval pipelines

Learn more: [Embeddings API Documentation](https://www.mixedbread.com/docs/embeddings/overview) | [API Reference](https://www.mixedbread.com/api-reference/endpoints/embeddings)

### MixedbreadDocumentEmbedder  
- **Purpose**: Embed multiple documents with batch processing
- **Input**: List of Haystack Documents
- **Output**: Documents with embeddings attached
- **Features**: Metadata field embedding, progress bars, customizable separators
- **Use case**: Document indexing in retrieval pipelines

Learn more: [Embeddings API Documentation](https://www.mixedbread.com/docs/embeddings/overview) | [API Reference](https://www.mixedbread.com/api-reference/endpoints/embeddings)

### MixedbreadReranker
- **Purpose**: Rerank documents by relevance to a query
- **Input**: Query string + List of Documents  
- **Output**: Reranked documents with relevance scores
- **Features**: Configurable top-k ranking
- **Use case**: Improving retrieval precision in RAG pipelines

Learn more: [Reranking API Documentation](https://www.mixedbread.com/docs/reranking/overview) | [API Reference](https://www.mixedbread.com/api-reference/endpoints/reranking)

### MixedbreadDocumentParser
- **Purpose**: Parse and extract structured content from various file formats
- **Input**: File paths or ByteStream objects (PDF, DOCX, PPTX, images, etc.)
- **Output**: Haystack Documents with parsed content and rich metadata
- **Features**: Multiple chunking strategies, element type filtering, async support
- **Use case**: Document preprocessing and content extraction for RAG pipelines

Learn more: [Parsing API Documentation](https://www.mixedbread.com/docs/parsing/overview) | [API Reference](https://www.mixedbread.com/api-reference/endpoints/parsing)

## Examples

Complete examples are available in the [`examples/`](./examples/) directory:

- **[Text Embedding](./examples/mixedbread_text_embedding.py)** - Basic text embedding with retrieval
- **[Document Embedding](./examples/mixedbread_document_embedding.py)** - Document indexing and search  
- **[Reranking](./examples/mixedbread_reranking.py)** - Document reranking for improved relevance
- **[Document Parsing](./examples/mixedbread_document_parsing.py)** - File parsing and content extraction

## Advanced Usage

### Custom Models
```python
embedder = MixedbreadTextEmbedder(model="mixedbread-ai/mxbai-embed-2d-large-v1")
reranker = MixedbreadReranker(model="mixedbread-ai/mxbai-rerank-xsmall-v1")
```

### Metadata Integration
```python
from haystack import Document

embedder = MixedbreadDocumentEmbedder(
    meta_fields_to_embed=["title", "category"],
    embedding_separator=" | "
)

# Note: rank_fields feature is not available in the current version
reranker = MixedbreadReranker(top_k=5)
```

## Testing

The package includes comprehensive tests. See [`tests/README.md`](./tests/README.md) for testing instructions.

```bash
# Run all tests
pytest tests/

# Run only integration tests (requires API key)
pytest -m integration tests/

# Run only unit tests (no API key needed)  
pytest -m "not integration" tests/
```

## License

This project is licensed under the Apache 2.0 License - see the LICENSE file for details.
