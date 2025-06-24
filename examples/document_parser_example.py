import asyncio
from mixedbread_ai_haystack import MixedbreadDocumentParser


def basic_usage():
    """Basic document parsing example."""
    print("=== Basic Document Parsing ===")

    # Initialize parser
    parser = MixedbreadDocumentParser(return_format="markdown")

    # Parse a document
    result = parser.run(sources=["data/acme_invoice.pdf"])
    documents = result["documents"]

    print(f"Parsed {len(documents)} chunks from document")

    # Show first chunk
    if documents:
        doc = documents[0]
        print(f"\nFirst chunk:")
        print(f"Content: {doc.content[:200]}...")
        print(f"Filename: {doc.meta['filename']}")
        print(f"Pages: {doc.meta['pages']}")
        print(f"Elements: {len(doc.meta['elements'])}")


async def async_usage():
    """Async document parsing with multiple files."""
    print("\n=== Async Document Parsing ===")

    # Initialize parser
    parser = MixedbreadDocumentParser(return_format="plain")

    # Parse multiple documents concurrently
    sources = ["data/acme_invoice.pdf"]  # Add more files as needed
    result = await parser.run_async(sources=sources)
    documents = result["documents"]

    print(f"Parsed {len(documents)} total chunks from {len(sources)} files")

    # Show document filenames
    filenames_found = set(doc.meta["filename"] for doc in documents)
    print(f"Files processed: {', '.join(filenames_found)}")


def main():
    """Run all examples."""
    try:
        basic_usage()

        # Run async example
        asyncio.run(async_usage())

    except Exception as e:
        print(f"Error: {e}")
        print("Make sure to:")
        print("1. Set MXBAI_API_KEY environment variable")
        print("2. Have a test PDF file at data/acme_invoice.pdf")


if __name__ == "__main__":
    main()
