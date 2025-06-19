import asyncio
from haystack import Document
from mixedbread_ai_haystack import MixedbreadReranker


def basic_usage():
    """Basic reranking example."""
    print("=== Basic Reranking Usage ===")

    # Initialize reranker
    reranker = MixedbreadReranker(model="mixedbread-ai/mxbai-rerank-large-v1", top_k=3)

    # Create documents to rerank
    documents = [
        Document(content="Machine learning is a subset of artificial intelligence."),
        Document(content="Deep learning uses neural networks with multiple layers."),
        Document(
            content="Natural language processing helps computers understand text."
        ),
        Document(content="Computer vision enables machines to analyze visual data."),
        Document(
            content="Reinforcement learning trains agents through trial and error."
        ),
    ]

    # Rerank documents based on a query
    query = "What is deep learning and neural networks?"
    result = reranker.run(documents=documents, query=query)
    reranked_docs = result["documents"]

    print(f"Query: {query}")
    print(f"Original documents: {len(documents)}")
    print(f"Reranked documents: {len(reranked_docs)}")
    print()

    for i, doc in enumerate(reranked_docs):
        print(f"Rank {i+1}:")
        print(f"  Content: {doc.content[:60]}...")
        print(f"  Score: {doc.meta['rerank_score']:.3f}")
        print()

    print(f"Model used: {result['meta']['model']}")
    print(f"Usage: {result['meta']['usage']}")
    print()


async def async_usage():
    """Async reranking example."""
    print("=== Async Reranking Usage ===")

    # Initialize reranker
    reranker = MixedbreadReranker(model="mixedbread-ai/mxbai-rerank-large-v1", top_k=2)

    # Create documents about programming
    documents = [
        Document(content="Python is a high-level programming language."),
        Document(content="JavaScript is commonly used for web development."),
        Document(content="Rust provides memory safety without garbage collection."),
        Document(content="Go is designed for scalable network services."),
    ]

    # Async reranking
    query = "Which language is best for web development?"
    result = await reranker.run_async(documents=documents, query=query)
    reranked_docs = result["documents"]

    print(f"Async Query: {query}")
    print(f"Top {len(reranked_docs)} results:")
    print()

    for i, doc in enumerate(reranked_docs):
        print(f"  {i+1}. {doc.content}")
        print(f"     Score: {doc.meta['rerank_score']:.3f}")
        print()


def custom_configuration():
    """Example with custom configuration."""
    print("=== Custom Configuration ===")

    # Initialize with custom settings
    reranker = MixedbreadReranker(
        model="mixedbread-ai/mxbai-rerank-xsmall-v1",  # Different model
        top_k=5,  # Return top 5 results
        return_input=True,  # Include input in API response
    )

    # Create documents
    documents = [
        Document(content="Artificial intelligence is transforming industries."),
        Document(content="Machine learning algorithms learn from data patterns."),
        Document(content="Deep learning networks have multiple hidden layers."),
        Document(content="Natural language models understand human text."),
        Document(content="Computer vision systems analyze images and videos."),
        Document(content="Robotics combines AI with mechanical engineering."),
    ]

    # Rerank with custom configuration
    query = "How do AI systems learn from data?"
    result = reranker.run(documents=documents, query=query)
    reranked_docs = result["documents"]

    print(f"Custom config returned {len(reranked_docs)} documents")
    print(f"Model: {result['meta']['model']}")
    print(f"Top K: {result['meta']['top_k']}")
    print()

    for i, doc in enumerate(reranked_docs):
        print(f"Rank {i+1}: Score {doc.meta['rerank_score']:.3f}")
        print(f"  {doc.content}")
        print()


def edge_cases():
    """Example handling edge cases."""
    print("=== Edge Cases ===")

    reranker = MixedbreadReranker(top_k=3)

    # Empty documents list
    result = reranker.run(documents=[], query="test query")
    print(f"Empty documents result: {len(result['documents'])} documents")

    # Empty query
    documents = [
        Document(content="Test document 1"),
        Document(content="Test document 2"),
    ]
    result = reranker.run(documents=documents, query="")
    print(
        f"Empty query result: {len(result['documents'])} documents (returns original)"
    )

    # Single document
    single_doc = [Document(content="Single test document")]
    result = reranker.run(documents=single_doc, query="test")
    print(f"Single document result: {len(result['documents'])} documents")
    print()


def main():
    """Run all examples."""
    try:
        basic_usage()
        custom_configuration()
        edge_cases()

        # Run async example
        asyncio.run(async_usage())

    except Exception as e:
        print(f"Error: {e}")
        print("Make sure to:")
        print("1. Set MXBAI_API_KEY environment variable")
        print("2. Have a valid Mixedbread API key")


if __name__ == "__main__":
    main()
