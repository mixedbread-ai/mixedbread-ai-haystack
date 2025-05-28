from mixedbread_ai_haystack import MixedbreadDocumentParser

parser = MixedbreadDocumentParser(
    element_types=["text", "title", "list-item", "table"],
)

print("=== Document Parsing Examples ===\n")

print("Example 1: File Path Parsing")
print("Note: Replace 'path/to/your/document.pdf' with an actual file path")
try:
    result = parser.run(sources=["path/to/your/document.pdf"])
    documents = result["documents"]
    print(f"Parsed {len(documents)} chunks from the document")
    print(f"First chunk: {documents[0].content[:200]}...")
except Exception as e:
    print(f"Error: {e}")
