from haystack import Document, Pipeline
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.document_stores.in_memory import InMemoryDocumentStore

from mixedbread_ai_haystack import MixedbreadDocumentEmbedder, MixedbreadTextEmbedder

print("=== Document Embedding & Retrieval Example ===\n")

document_store = InMemoryDocumentStore(embedding_similarity_function="cosine")

documents = [
    Document(
        content="German rye bread is known for its dense texture and rich flavor, made with a combination of rye and wheat flour",
        meta={
            "category": "bread",
            "region": "Germany",
            "difficulty": "intermediate",
            "prep_time": "4 hours",
        },
    ),
    Document(
        content="Pretzels are a traditional German baked good often served with beer, featuring a distinctive twisted shape",
        meta={
            "category": "snack",
            "region": "Germany",
            "difficulty": "advanced",
            "prep_time": "3 hours",
        },
    ),
    Document(
        content="Black Forest cake is a famous German dessert made with chocolate sponge, cherries, and whipped cream",
        meta={
            "category": "dessert",
            "region": "Germany",
            "difficulty": "advanced",
            "prep_time": "2 hours",
        },
    ),
    Document(
        content="Stollen is a traditional German Christmas bread filled with dried fruits, nuts, and marzipan",
        meta={
            "category": "bread",
            "region": "Germany",
            "difficulty": "advanced",
            "prep_time": "6 hours",
        },
    ),
    Document(
        content="German sourdough bread uses wild yeast starter and has a tangy flavor with chewy texture",
        meta={
            "category": "bread",
            "region": "Germany",
            "difficulty": "expert",
            "prep_time": "24 hours",
        },
    ),
]

print(f"Indexing {len(documents)} documents...")

document_embedder = MixedbreadDocumentEmbedder(
    model="mixedbread-ai/mxbai-embed-large-v1",
)

try:
    documents_with_embeddings = document_embedder.run(documents)["documents"]
    document_store.write_documents(documents_with_embeddings)

    print(
        f"Successfully embedded and stored {len(documents_with_embeddings)} documents"
    )
    print(f"Embedding dimension: {len(documents_with_embeddings[0].embedding)}")

except Exception as e:
    print(f"Error during embedding: {e}")
    exit(1)

print("\n" + "-" * 60)

print("Building retrieval pipeline...")

query_pipeline = Pipeline()
query_pipeline.add_component(
    "text_embedder",
    MixedbreadTextEmbedder(model="mixedbread-ai/mxbai-embed-large-v1"),
)
query_pipeline.add_component(
    "retriever",
    InMemoryEmbeddingRetriever(document_store=document_store, top_k=3),
)
query_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")

print("Pipeline ready!")

print("\n" + "-" * 60)

queries = [
    "What is German bread like?",
    "Tell me about German desserts",
    "How long does it take to make traditional German baked goods?",
    "What are some difficult recipes from Germany?",
]

print("Testing retrieval with different queries:\n")

for query in queries:
    print(f"Query: '{query}'")

    try:
        result = query_pipeline.run({"text_embedder": {"text": query}})
        retrieved_docs = result["retriever"]["documents"]

        print(f"Found {len(retrieved_docs)} relevant documents:")

        for i, doc in enumerate(retrieved_docs, 1):
            score = doc.score if hasattr(doc, "score") else "N/A"
            print(f"  {i}. [Score: {score:.4f}] {doc.content[:80]}...")
            print(
                f"     Region: {doc.meta.get('region', 'N/A')} | "
                f"Category: {doc.meta.get('category', 'N/A')} | "
                f"Prep: {doc.meta.get('prep_time', 'N/A')}"
            )

    except Exception as e:
        print(f"Error during retrieval: {e}")

    print("-" * 40)

print("\n=== Async Document Embedding ===\n")

# Async document embedding example
try:
    import asyncio
    
    async def async_document_embedding():
        """Demonstrate async document embedding with concurrent processing."""
        document_embedder = MixedbreadDocumentEmbedder(
            model="mixedbread-ai/mxbai-embed-large-v1",
        )
        
        # Split documents into batches for concurrent processing
        batch_size = 3
        batches = [documents[i:i + batch_size] for i in range(0, len(documents), batch_size)]
        
        print(f"Processing {len(documents)} documents in {len(batches)} async batches...")
        
        # Process batches concurrently
        tasks = [
            document_embedder.run_async(documents=batch) 
            for batch in batches
        ]
        
        import time
        start_time = time.time()
        results = await asyncio.gather(*tasks)
        end_time = time.time()
        
        # Combine results
        all_embedded_docs = []
        for result in results:
            all_embedded_docs.extend(result["documents"])
        
        print(f"✓ Async embedding completed:")
        print(f"  Total documents: {len(all_embedded_docs)}")
        print(f"  Processing time: {(end_time - start_time)*1000:.1f}ms")
        print(f"  Average per document: {(end_time - start_time)*1000/len(all_embedded_docs):.1f}ms")
        print(f"  Embedding dimensions: {len(all_embedded_docs[0].embedding)}")
        
        return all_embedded_docs
    
    # Run async example
    embedded_docs = asyncio.run(async_document_embedding())
    
    print("\n✓ Async document embedding completed successfully!")
    
except Exception as e:
    print(f"✗ Error with async document embedding: {e}")

print("\n=== All Examples Complete ===")
print("Documents are now embedded and can be used for retrieval!")
