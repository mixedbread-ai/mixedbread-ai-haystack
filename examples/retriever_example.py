import asyncio
from mixedbread_ai_haystack import MixedbreadVectorStoreRetriever


def chunk_search():
    """Basic chunk-level search example."""
    print("=== Chunk-Level Search ===")

    # Initialize retriever for chunk search
    retriever = MixedbreadVectorStoreRetriever(
        vector_store_identifiers=["your-vector-store-id"],
        search_type="chunk",
        top_k=5,
    )

    # Search for relevant chunks
    query = "machine learning algorithms"
    result = retriever.run(query=query)
    documents = result["documents"]

    print(f"Query: {query}")
    print(f"Found {len(documents)} relevant chunks")
    print(f"Meta: {result['meta']}")
    print()

    for i, doc in enumerate(documents):
        print(f"Chunk {i+1}:")
        print(f"  Content: {doc.content[:100]}...")
        print(f"  Score: {doc.meta['retrieval_score']:.3f}")
        if "filename" in doc.meta:
            print(f"  Source: {doc.meta['filename']}")
        print()


def file_search():
    """File-level search example."""
    print("=== File-Level Search ===")

    # Initialize retriever for file search
    retriever = MixedbreadVectorStoreRetriever(
        vector_store_identifiers=["your-vector-store-id"],
        search_type="file",
        top_k=3,
    )

    # Search for relevant files
    query = "neural networks and deep learning"
    result = retriever.run(query=query)
    documents = result["documents"]

    print(f"Query: {query}")
    print(f"Found {len(documents)} relevant files")
    print(f"Meta: {result['meta']}")
    print()

    for i, doc in enumerate(documents):
        print(f"File {i+1}:")
        print(f"  Content: {doc.content[:150]}...")
        print(f"  Score: {doc.meta['retrieval_score']:.3f}")
        if "filename" in doc.meta:
            print(f"  Filename: {doc.meta['filename']}")
        print()


async def async_search():
    """Async search example."""
    print("=== Async Search ===")

    # Initialize retriever
    retriever = MixedbreadVectorStoreRetriever(
        vector_store_identifiers=[
            "your-vector-store-id",
            "your-vector-store-id",
        ],
        search_type="chunk",
        top_k=4,
    )

    # Async search across multiple stores
    query = "artificial intelligence research"
    result = await retriever.run_async(query=query)
    documents = result["documents"]

    print(f"Async query: {query}")
    print(f"Found {len(documents)} results across multiple stores")
    print(f"Searched stores: {result['meta']['vector_stores']}")
    print()

    for i, doc in enumerate(documents):
        print(
            f"  {i+1}. Score {doc.meta['retrieval_score']:.3f}: {doc.content[:80]}..."
        )


def main():
    """Run all examples."""
    try:
        chunk_search()
        file_search()

        # Run async example
        asyncio.run(async_search())

    except Exception as e:
        print(f"Error: {e}")
        print("Make sure to:")
        print("1. Set MXBAI_API_KEY environment variable")
        print("2. Replace vector store identifiers with actual ones")
        print("3. Have vector stores with indexed documents")


if __name__ == "__main__":
    main()
