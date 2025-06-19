# Mixedbread AI Haystack Integration

[![PyPI version](https://badge.fury.io/py/mixedbread-ai-haystack.svg)](https://badge.fury.io/py/mixedbread-ai-haystack)
[![Python versions](https://img.shields.io/pypi/pyversions/mixedbread-ai-haystack.svg)](https://pypi.org/project/mixedbread-ai-haystack/)

**Mixedbread AI** integration for **Haystack 2.2.1, providing state-of-the-art embedding, reranking, document parsing, and retrieval capabilities.

## Components

- **MixedbreadTextEmbedder** - Embed single texts and queries
- **MixedbreadDocumentEmbedder** - Embed documents with metadata support  
- **MixedbreadReranker** - Rerank documents by relevance
- **MixedbreadDocumentParser** - Parse and extract content from various file formats
- **MixedbreadVectorStoreRetriever** - Search indexed documents with vector similarity

## Installation

```bash
pip install mixedbread-ai-haystack
```

## Quick Start

Get your API key from the [Mixedbread Platform](https://www.platform.mixedbread.com/) and set it as an environment variable:

```bash
export MIXEDBREAD_API_KEY="your-api-key"
```

### Basic Usage

```python
from mixedbread_ai_haystack import MixedbreadTextEmbedder

# Embed text
embedder = MixedbreadTextEmbedder(model="mixedbread-ai/mxbai-embed-large-v1")
result = embedder.run(text="What is the capital of France?")
embedding = result["embedding"]
```

## Async Support

All components support async operations:

```python
import asyncio

async def embed_text():
    embedder = MixedbreadTextEmbedder()
    result = await embedder.run_async(text="Async embedding example")
    return result["embedding"]

# Run async
embedding = asyncio.run(embed_text())
```

## Examples

See the [`examples/`](./examples/) directory for complete usage examples:

- **[Embedders](./examples/embedders_example.py)** - Text and document embedding
- **[Reranker](./examples/reranker_example.py)** - Document reranking
- **[Document Parser](./examples/document_parser_example.py)** - File parsing
- **[Vector Retriever](./examples/retriever_example.py)** - Vector-based search

## Testing

```bash
# Install dev dependencies
pip install -e .[dev]

# Run tests
pytest tests/
```

## Documentation

Learn more at [mixedbread.com/docs](https://www.mixedbread.com/docs):
- [Embeddings API](https://www.mixedbread.com/docs/embeddings/overview)
- [Reranking API](https://www.mixedbread.com/docs/reranking/overview)  
- [Parsing API](https://www.mixedbread.com/docs/parsing/overview)
- [Vector Stores API](https://www.mixedbread.com/docs/vector-stores/overview)

## License

Apache 2.0 License