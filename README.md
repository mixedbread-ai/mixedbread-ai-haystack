# Mixedbread AI Haystack Integration

[![PyPI version](https://badge.fury.io/py/mixedbread-ai-haystack.svg)](https://badge.fury.io/py/mixedbread-ai-haystack)
[![Python versions](https://img.shields.io/pypi/pyversions/mixedbread-ai-haystack.svg)](https://pypi.org/project/mixedbread-ai-haystack/)
[![Python versions](https://img.shields.io/pypi/pyversions/mixedbread-ai-haystack.svg)](https://pypi.org/project/mixedbread-ai-haystack/)

**Mixedbread AI** integration for **Haystack 2.x**, providing state-of-the-art embedding and reranking components.

## Overview

[Mixedbread AI](https://www.mixedbread.com) provides best-in-class embedding and reranking models, both open-source and proprietary. This integration brings three powerful components to your Haystack pipelines:

- **MixedbreadTextEmbedder** - For embedding single texts and queries
- **MixedbreadDocumentEmbedder** - For embedding documents with metadata support  
- **MixedbreadReranker** - For reranking documents by relevance

More information can be found in the [official documentation](https://www.mixedbread.com/api-reference/integrations#haystack).

## Installation

```bash
pip install mixedbread-ai-haystack
```

## Quick Start

### 1. Get your API key
Sign up at [mixedbread.com](https://www.mixedbread.com) and get your API key from the [dashboard](https://www.mixedbread.com/dashboard?next=api-keys).

### 2. Set your API key
```bash
export MXBAI_API_KEY="your-api-key-here"
```

### 3. Basic usage
```python
from mixedbread_ai_haystack import MixedbreadTextEmbedder

# Initialize the embedder
embedder = MixedbreadTextEmbedder(model="mixedbread-ai/mxbai-embed-large-v1")

# Generate embeddings
result = embedder.run(text="What is the capital of France?")
embedding = result["embedding"]  # List of floats
```

## Components

### MixedbreadTextEmbedder
- **Purpose**: Embed single texts, queries, and short passages
- **Input**: String
- **Output**: List of float embeddings
- **Use case**: Query encoding in retrieval pipelines

### MixedbreadDocumentEmbedder  
- **Purpose**: Embed multiple documents with batch processing
- **Input**: List of Haystack Documents
- **Output**: Documents with embeddings attached
- **Features**: Metadata field embedding, progress bars, customizable separators
- **Use case**: Document indexing in retrieval pipelines

### MixedbreadReranker
- **Purpose**: Rerank documents by relevance to a query
- **Input**: Query string + List of Documents  
- **Output**: Reranked documents with relevance scores
- **Features**: Metadata field inclusion, configurable top-k
- **Use case**: Improving retrieval precision in RAG pipelines

## Examples

Complete examples are available in the [`examples/`](./examples/) directory:

- **[Text Embedding](./examples/mixedbread_text_embedding.py)** - Basic text embedding with retrieval
- **[Document Embedding](./examples/mixedbread_document_embedding.py)** - Document indexing and search  
- **[Reranking](./examples/mixedbread_reranking.py)** - Document reranking for improved relevance

## Advanced Usage

### Custom Models
```python
embedder = MixedbreadTextEmbedder(model="mixedbread-ai/mxbai-embed-2d-large-v1")
reranker = MixedbreadReranker(model="mixedbread-ai/mxbai-rerank-xsmall-v1")
```

### Custom Base URL
```python
embedder = MixedbreadTextEmbedder(
    base_url="https://your-custom-endpoint.com",
    api_key="your-api-key"
)
```

### Metadata Integration
```python
from haystack import Document

embedder = MixedbreadDocumentEmbedder(
    meta_fields_to_embed=["title", "category"],
    embedding_separator=" | "
)

reranker = MixedbreadReranker(
    rank_fields=["title", "summary"]
)
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
