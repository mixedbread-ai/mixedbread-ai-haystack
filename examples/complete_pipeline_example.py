"""
Complete Mixedbread AI Pipeline Example

This example demonstrates a full RAG (Retrieval-Augmented Generation) pipeline using
all Mixedbread AI components together:
1. Document parsing with MixedbreadDocumentParser
2. Document embedding with MixedbreadDocumentEmbedder
3. Query embedding with MixedbreadTextEmbedder
4. Document retrieval and reranking with MixedbreadReranker

This creates a complete search and retrieval system.
"""

import time
from haystack import Document, Pipeline
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.document_stores.in_memory import InMemoryDocumentStore

from mixedbread_ai_haystack import (
    MixedbreadDocumentParser,
    MixedbreadDocumentEmbedder,
    MixedbreadTextEmbedder,
    MixedbreadReranker,
)


def main():
    print("=== Complete Mixedbread AI Pipeline Example ===\n")

    # Step 1: Document Parsing
    print("1. Document Parsing")
    print("-" * 50)

    try:
        parser = MixedbreadDocumentParser(
            chunking_strategy="page",
            return_format="markdown",
            element_types=["text", "title", "list-item"],
        )

        # Parse the example document
        parse_result = parser.run(sources=["data/acme_invoice.pdf"])
        parsed_documents = parse_result["documents"]

        print(f"✓ Parsed {len(parsed_documents)} chunks from document")
        print(f"✓ Parse model: {parse_result['meta'].get('model', 'N/A')}")

        # Show sample parsed content
        for i, doc in enumerate(parsed_documents[:2]):
            print(f"\n  Chunk {i+1}: {doc.content[:100]}...")

    except Exception as e:
        print(f"✗ Document parsing failed: {e}")
        print("  Creating sample documents instead...")

        # Fallback: Create sample documents if parsing fails
        parsed_documents = [
            Document(
                content="ACME Corporation Invoice #12345. Date: 2024-01-15. Bill To: John Doe, 123 Main St, Anytown, ST 12345.",
                meta={"source": "invoice", "page": 1, "type": "header"},
            ),
            Document(
                content="Product A - Consulting Services. Quantity: 10 hours. Rate: $150/hour.",
                meta={"source": "invoice", "page": 1, "type": "line_item"},
            ),
            Document(
                content="Total: $1,500.00. Payment terms: Net 30 days.",
                meta={"source": "invoice", "page": 1, "type": "footer"},
            ),
        ]
        print(f"✓ Created {len(parsed_documents)} sample documents")

    # Step 2: Document Embedding
    print(f"\n\n2. Document Embedding")
    print("-" * 50)

    try:
        document_embedder = MixedbreadDocumentEmbedder(
            model="mixedbread-ai/mxbai-embed-large-v1",
            normalized=True,
        )

        start_time = time.time()
        embed_result = document_embedder.run(documents=parsed_documents)
        end_time = time.time()

        embedded_documents = embed_result["documents"]

        print(f"✓ Embedded {len(embedded_documents)} documents")
        print(f"✓ Embedding model: {embed_result['meta'].get('model', 'N/A')}")
        print(
            f"✓ Embedding dimensions: {embed_result['meta'].get('dimensions', 'N/A')}"
        )
        print(f"✓ Processing time: {(end_time - start_time)*1000:.1f}ms")
        print(f"✓ Token usage: {embed_result['meta'].get('usage', {})}")

    except Exception as e:
        print(f"✗ Document embedding failed: {e}")
        return

    # Step 3: Store Documents
    print(f"\n\n3. Document Storage")
    print("-" * 50)

    try:
        document_store = InMemoryDocumentStore(embedding_similarity_function="cosine")
        document_store.write_documents(embedded_documents)

        print(f"✓ Stored {document_store.count_documents()} documents in vector store")
        print(f"✓ Similarity function: cosine")

    except Exception as e:
        print(f"✗ Document storage failed: {e}")
        return

    # Step 4: Query Processing & Retrieval
    print(f"\n\n4. Query Processing & Retrieval")
    print("-" * 50)

    queries = [
        "What is the invoice number and total amount?",
        "Who is the bill to customer?",
        "What services were provided?",
    ]

    try:
        text_embedder = MixedbreadTextEmbedder(
            model="mixedbread-ai/mxbai-embed-large-v1",
            normalized=True,
        )

        retriever = InMemoryEmbeddingRetriever(
            document_store=document_store,
            top_k=5,
        )

        for i, query in enumerate(queries, 1):
            print(f"\nQuery {i}: {query}")

            # Embed the query
            query_result = text_embedder.run(text=query)
            query_embedding = query_result["embedding"]

            # Retrieve documents
            retrieval_result = retriever.run(
                query_embedding=query_embedding,
                top_k=3,
            )
            retrieved_docs = retrieval_result["documents"]

            print(f"  ✓ Retrieved {len(retrieved_docs)} documents")

            # Show top result
            if retrieved_docs:
                top_doc = retrieved_docs[0]
                score = top_doc.score if hasattr(top_doc, "score") else "N/A"
                print(f"  ✓ Top result (score: {score}): {top_doc.content[:80]}...")

    except Exception as e:
        print(f"✗ Query processing failed: {e}")
        return

    # Step 5: Reranking
    print(f"\n\n5. Document Reranking")
    print("-" * 50)

    try:
        reranker = MixedbreadReranker(
            model="mixedbread-ai/mxbai-rerank-large-v1",
            top_k=2,
        )

        test_query = queries[0]  # Use first query for reranking demo

        # Get documents for reranking (using all embedded documents)
        rerank_result = reranker.run(
            documents=embedded_documents,
            query=test_query,
        )

        reranked_docs = rerank_result["documents"]

        print(f"✓ Reranked {len(reranked_docs)} documents for query: '{test_query}'")
        print(f"✓ Rerank model: {rerank_result['meta'].get('model', 'N/A')}")
        print(f"✓ Token usage: {rerank_result['meta'].get('usage', {})}")

        # Show reranked results
        for i, doc in enumerate(reranked_docs):
            score = doc.meta.get("rerank_score", "N/A")
            print(f"\n  Rank {i+1} (score: {score:.4f}): {doc.content[:80]}...")

    except Exception as e:
        print(f"✗ Document reranking failed: {e}")

    # Step 6: Complete Pipeline Setup
    print(f"\n\n6. Complete Pipeline Setup")
    print("-" * 50)

    try:
        # Create a Haystack pipeline that combines everything
        pipeline = Pipeline()

        # Add components
        pipeline.add_component("text_embedder", text_embedder)
        pipeline.add_component("retriever", retriever)
        pipeline.add_component("reranker", reranker)

        # Connect components
        pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
        pipeline.connect("retriever.documents", "reranker.documents")

        print("✓ Created complete pipeline with components:")
        print("  text_embedder → retriever → reranker")

        # Test the pipeline
        test_query = "invoice total amount"

        pipeline_result = pipeline.run(
            {"text_embedder": {"text": test_query}, "reranker": {"query": test_query}}
        )

        final_documents = pipeline_result["reranker"]["documents"]

        print(f"\n✓ Pipeline execution successful!")
        print(f"✓ Query: '{test_query}'")
        print(f"✓ Final results: {len(final_documents)} documents")

        # Show final ranked results
        for i, doc in enumerate(final_documents):
            score = doc.meta.get("rerank_score", "N/A")
            print(f"  Result {i+1} (score: {score:.4f}): {doc.content[:60]}...")

    except Exception as e:
        print(f"✗ Pipeline setup failed: {e}")

    # Performance Summary
    print(f"\n\n=== Performance Summary ===")
    print("✓ Document Parsing: Parse various file formats into chunks")
    print("✓ Document Embedding: Convert text to high-dimensional vectors")
    print("✓ Vector Storage: Store embeddings for fast similarity search")
    print("✓ Query Processing: Convert queries to matching vector space")
    print("✓ Semantic Retrieval: Find similar documents using cosine similarity")
    print("✓ Reranking: Improve relevance with cross-encoder models")
    print("✓ Pipeline Integration: Seamless Haystack pipeline integration")

    print(f"\n=== Complete Pipeline Example Finished ===")
    print("This demonstrates a full RAG system using all Mixedbread AI components!")


if __name__ == "__main__":
    main()
