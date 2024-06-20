# mixedbread ai Haystack 2.0 Integration
[![PyPI version](https://badge.fury.io/py/mixedbread-ai-haystack.svg)](https://badge.fury.io/py/mixedbread-ai-haystack)
[![Python versions](https://img.shields.io/pypi/pyversions/mixedbread-ai-haystack.svg)](https://pypi.org/project/mixedbread-ai-haystack/) 
### **Table of Contents**

- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)

## Overview

[mixedbread ai](https://www.mixedbread.ai) is an AI start-up that provides open-source, as well as, in-house embedding and reranking models. You can choose from various foundation models to find the one best suited for your use case. More information can be found on the [documentation page](https://www.mixedbread.ai/api-reference/integrations#haystack).

## Installation

Install the mixedbread ai integration with a simple pip command:

```bash
pip install mixedbread-ai-haystack
```

## Usage

This integration comes with 3 components:
- [`MixedbreadAITextEmbedder`](https://github.com/mixedbread-ai/mixedbread-ai-haystack/blob/main/mixedbread_ai_haystack/embedders/text_embedder.py)
- [`MixedbreadAIDocumentEmbedder`](https://github.com/mixedbread-ai/mixedbread-ai-haystack/blob/main/mixedbread_ai_haystack/embedders/document_embedder.py).
- [`MixedbreadAIReranker`](https://github.com/mixedbread-ai/mixedbread-ai-haystack/blob/main/mixedbread_ai_haystack/rerankers/reranker.py)

For documents you can use `MixedbreadAIDocumentEmbedder` and for queries you can use `MixedbreadAITextEmbedder`. Once you've selected the component for your specific use case, initialize the component with the `model` and the [`api_key`](https://www.mixedbread.ai/dashboard?next=api-keys). You can also set the environment variable `MXBAI_API_KEY` instead of passing the api key as an argument.



### Embedders In a Pipeline

```python
from haystack import Document, Pipeline
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.writers import DocumentWriter
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from mixedbread_ai_haystack.embedders import MixedbreadAIDocumentEmbedder, MixedbreadAITextEmbedder


# -------------------------------------
# Indexing Pipeline
# -------------------------------------
document_store = InMemoryDocumentStore(embedding_similarity_function="cosine")
documents = [
    Document(content="china is the most populous country in the world."), 
    Document(content="india is the second most populous country in the world."), 
    Document(content="united states is the third most populous country in the world.")
]

indexing_pipeline = Pipeline()
indexing_pipeline.add_component("doc_embedder", MixedbreadAIDocumentEmbedder(model="mixedbread-ai/mxbai-embed-large-v1"))
indexing_pipeline.add_component("writer", DocumentWriter(document_store=document_store))
indexing_pipeline.connect("doc_embedder", "writer")

indexing_pipeline.run({"doc_embedder": {"documents": documents}})

# -------------------------------------
# Query Pipeline
# -------------------------------------
text_embedder = MixedbreadAITextEmbedder(model="mixedbread-ai/mxbai-embed-large-v1")

# Query Pipeline
query_pipeline = Pipeline()
query_pipeline.add_component("text_embedder", text_embedder)
query_pipeline.add_component("retriever", InMemoryEmbeddingRetriever(document_store=document_store))
query_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")

results = query_pipeline.run({"text_embedder": {"text": "Which country has the biggest population?"}})
top_document = results["retriever"]["documents"][0].content
print(top_document)
```

### Reranker in a Pipeline
```python
from haystack import Document, Pipeline
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from mixedbread_ai_haystack.rerankers import MixedbreadAIReranker

documents = [
    Document(content="china is the most populous country in the world."), 
    Document(content="india is the second most populous country in the world."), 
    Document(content="united states is the third most populous country in the world.")
]
document_store = InMemoryDocumentStore()
document_store.write_documents(documents)

retriever = InMemoryBM25Retriever(document_store=document_store)
reranker = MixedbreadAIReranker(model="mixedbread-ai/mxbai-rerank-large-v1", top_k=3)

rerank_pipeline = Pipeline()
rerank_pipeline.add_component("retriever", retriever)
rerank_pipeline.add_component("reranker", reranker)

rerank_pipeline.connect("retriever.documents", "reranker.documents")

query = "Which country has the second largest population"
results = rerank_pipeline.run({"retriever": {"query": query}, "reranker": {"query": query, "top_k": 3}})
print(results)
```
