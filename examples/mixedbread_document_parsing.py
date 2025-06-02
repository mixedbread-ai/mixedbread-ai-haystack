from mixedbread_ai_haystack import MixedbreadDocumentParser

parser = MixedbreadDocumentParser(
    element_types=["text", "title", "list-item", "table"],
)

print("=== Document Parsing Example ===\n")
print("Replace path with your own file path")

try:
    result = parser.run(sources=["data/acme_invoice.pdf"])
    documents = result["documents"]
    print(f"Parsed {len(documents)} chunks from the document")
    print(f"First chunk: {documents[0].content[:200]}...")
except Exception as e:
    print(f"Error: {e}")
