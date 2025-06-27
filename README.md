# Mixedbread AI Haystack Integration

[![PyPI version](https://badge.fury.io/py/mixedbread-ai-haystack.svg)](https://badge.fury.io/py/mixedbread-ai-haystack)
[![Python versions](https://img.shields.io/pypi/pyversions/mixedbread-ai-haystack.svg)](https://pypi.org/project/mixedbread-ai-haystack/)

**Mixedbread AI** integration for **Haystack**. This package provides seamless access to Mixedbread's multimodal AI capabilities, enabling intelligent search that understands meaning across text, images, code, PDFs, and diverse document types. Use our state of the art embedding and reranking models as part of your haystack workflows.

## Components

- **MixedbreadTextEmbedder** - State-of-the-art embedding models that generate vectors capturing deep contextual meaning for single texts and queries
- **MixedbreadDocumentEmbedder** - Embed full documents using advanced embedding models
- **MixedbreadReranker** - Powerful semantic reranking that significantly boosts search relevance
- **MixedbreadDocumentParser** - Layout-aware document parsing supporting PDF, PPTX, HTML and more formats
- **MixedbreadVectorStoreRetriever** - AI-native search engine that enables conversational queries across multimodal data

## Installation

```bash
pip install mixedbread-ai-haystack
```

## Quick Start

Get your API key from the [Mixedbread Platform](https://platform.mixedbread.com/) and set it as an environment variable:

```bash
export MXBAI_API_KEY="your-api-key"
```

### Basic Usage

```python
from mixedbread_ai_haystack import MixedbreadTextEmbedder

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

embedding = asyncio.run(embed_text())
```

## Examples

See the [`examples/`](./examples/) directory for complete usage examples:

- **[Embedders](https://github.com/mixedbread-ai/mixedbread-ai-haystack/blob/main/examples/embedders_example.py)** - Text and document embedding
- **[Reranker](https://github.com/mixedbread-ai/mixedbread-ai-haystack/blob/main/examples/reranker_example.py)** - Document reranking
- **[Document Parser](https://github.com/mixedbread-ai/mixedbread-ai-haystack/blob/main/examples/document_parser_example.py)** - File parsing
- **[Vector Retriever](https://github.com/mixedbread-ai/mixedbread-ai-haystack/blob/main/examples/retriever_example.py)** - Vector-based search

## Testing

```bash
# Install dev dependencies
pip install -e .[dev]

# Run all tests
python run_tests.py all

# Run only unit tests
python run_tests.py unit

# Run only integration tests (requires API key)
python run_tests.py integration

# Run specific test files
python run_tests.py tests/test_text_embedder.py
```

## Documentation

Learn more at [mixedbread.com/docs](https://www.mixedbread.com/docs):

- [Embeddings API](https://www.mixedbread.com/docs/embeddings/overview)
- [Reranking API](https://www.mixedbread.com/docsreranking/overview)
- [Parsing API](https://www.mixedbread.com/docs/parsing/overview)
- [Vector Stores API](https://www.mixedbread.com/docs/vector-stores/overview)

## License

Apache 2.0 License
