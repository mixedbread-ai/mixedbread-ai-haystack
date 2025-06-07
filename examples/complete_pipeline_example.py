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

    # Step 7: Async Processing
    print(f"\n\n7. Async Processing")
    print("-" * 50)
    
    try:
        import asyncio
        
        async def async_pipeline_demo():
            """Demonstrate async processing capabilities."""
            print("Setting up async components...")
            
            # Initialize components for async use
            async_text_embedder = MixedbreadTextEmbedder(
                model="mixedbread-ai/mxbai-embed-large-v1"
            )
            async_doc_embedder = MixedbreadDocumentEmbedder(
                model="mixedbread-ai/mxbai-embed-large-v1"
            )
            async_reranker = MixedbreadReranker(
                model="mixedbread-ai/mxbai-rerank-large-v1",
                top_k=3
            )
            
            # Test queries for concurrent processing
            test_queries = [
                "invoice total amount calculation",
                "payment terms and conditions",
                "billing address and contact info"
            ]
            
            print(f"Processing {len(test_queries)} queries concurrently...")
            
            # Concurrent query embedding
            start_time = time.time()
            query_tasks = [
                async_text_embedder.run_async(text=query) 
                for query in test_queries
            ]
            await asyncio.gather(*query_tasks)
            query_time = time.time() - start_time
            
            print(f"✓ Query embedding: {query_time*1000:.1f}ms for {len(test_queries)} queries")
            
            # Concurrent document reranking
            start_time = time.time()
            rerank_tasks = [
                async_reranker.run_async(documents=embedded_documents[:5], query=query)
                for query in test_queries
            ]
            rerank_results = await asyncio.gather(*rerank_tasks)
            rerank_time = time.time() - start_time
            
            print(f"✓ Document reranking: {rerank_time*1000:.1f}ms for {len(test_queries)} queries")
            
            # Document embedding batches (if we had more documents)
            if len(embedded_documents) >= 6:
                doc_batches = [embedded_documents[i:i+3] for i in range(0, min(9, len(embedded_documents)), 3)]
                
                start_time = time.time()
                batch_tasks = [
                    async_doc_embedder.run_async(documents=batch)
                    for batch in doc_batches
                ]
                batch_results = await asyncio.gather(*batch_tasks)
                batch_time = time.time() - start_time
                
                total_docs = sum(len(result["documents"]) for result in batch_results)
                print(f"✓ Batch document embedding: {batch_time*1000:.1f}ms for {total_docs} documents")
            
            # Performance summary
            print(f"\nAsync Performance Summary:")
            print(f"  Query processing: {len(test_queries)} queries in {query_time*1000:.1f}ms")
            print(f"  Average per query: {query_time*1000/len(test_queries):.1f}ms")
            print(f"  Reranking: {len(test_queries)} operations in {rerank_time*1000:.1f}ms")
            
            # Show sample results
            print(f"\nSample async results:")
            for i, (query, rerank_result) in enumerate(zip(test_queries, rerank_results)):
                top_doc = rerank_result["documents"][0]
                score = top_doc.meta.get("rerank_score", 0)
                print(f"  Query {i+1}: '{query[:30]}...'")
                print(f"    Top result (score: {score:.4f}): {top_doc.content[:50]}...")
            
            return {
                'query_time': query_time,
                'rerank_time': rerank_time,
                'total_queries': len(test_queries)
            }
        
        # Run async demo
        async_results = asyncio.run(async_pipeline_demo())
        
        print(f"\n✓ Async processing demonstration completed!")
        print(f"✓ Demonstrated concurrent query embedding and reranking")
        print(f"✓ Total processing time: {(async_results['query_time'] + async_results['rerank_time'])*1000:.1f}ms")
        
    except Exception as e:
        print(f"✗ Async processing demo failed: {e}")

    print(f"\n=== Complete Pipeline Example Finished ===")
    print("This demonstrates a full RAG system using all Mixedbread AI components!")
    print("The async capabilities enable efficient concurrent processing for better throughput.")


if __name__ == "__main__":
    main()
