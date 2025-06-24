import asyncio
from haystack import Document
from mixedbread_ai_haystack import MixedbreadTextEmbedder, MixedbreadDocumentEmbedder


def text_embedder_basic():
    """Basic text embedding example."""
    print("=== Text Embedder - Basic Usage ===")

    # Initialize text embedder
    embedder = MixedbreadTextEmbedder(model="mixedbread-ai/mxbai-embed-large-v1")

    # Embed a single text
    text = "Machine learning is transforming artificial intelligence."
    result = embedder.run(text=text)

    print(f"Text: {text}")
    print(f"Embedding dimension: {len(result['embedding'])}")
    print(f"Model used: {result['meta']['model']}")
    print(f"Usage: {result['meta']['usage']}")
    print()


def document_embedder_basic():
    """Basic document embedding example."""
    print("=== Document Embedder - Basic Usage ===")

    # Initialize document embedder
    embedder = MixedbreadDocumentEmbedder(model="mixedbread-ai/mxbai-embed-large-v1")

    # Create documents to embed
    documents = [
        Document(content="Artificial intelligence is revolutionizing technology."),
        Document(content="Machine learning algorithms learn from data patterns."),
        Document(
            content="Neural networks are inspired by biological brain structures."
        ),
    ]

    # Embed documents
    result = embedder.run(documents=documents)
    embedded_docs = result["documents"]

    print(f"Embedded {len(embedded_docs)} documents")
    for i, doc in enumerate(embedded_docs):
        print(f"Doc {i+1}: {doc.content[:50]}...")
        print(f"  Embedding dimension: {len(doc.embedding)}")
        print(f"  First 3 values: {doc.embedding[:3]}")

    print(f"Model used: {result['meta']['model']}")
    print(f"Total usage: {result['meta']['usage']}")
    print()


async def async_examples():
    """Async embedding examples."""
    print("=== Async Embedding Examples ===")

    # Initialize embedders
    text_embedder = MixedbreadTextEmbedder()
    doc_embedder = MixedbreadDocumentEmbedder()

    # Async text embedding
    text_result = await text_embedder.run_async("Deep learning uses neural networks.")
    print(f"Async text embedding dimension: {len(text_result['embedding'])}")

    # Async document embedding
    documents = [
        Document(content="Computer vision enables image understanding."),
        Document(content="Natural language processing handles text data."),
    ]

    doc_result = await doc_embedder.run_async(documents=documents)
    print(f"Async embedded {len(doc_result['documents'])} documents")

    for i, doc in enumerate(doc_result["documents"]):
        print(f"  Doc {i+1} embedding dimension: {len(doc.embedding)}")
    print()


def main():
    """Run all examples."""
    try:
        text_embedder_basic()
        document_embedder_basic()

        # Run async examples
        asyncio.run(async_examples())

    except Exception as e:
        print(f"Error: {e}")
        print("Make sure to:")
        print("1. Set MXBAI_API_KEY environment variable")
        print("2. Have a valid Mixedbread API key")


if __name__ == "__main__":
    main()
