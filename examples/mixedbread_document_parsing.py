"""
Mixedbread AI Document Parsing Example

This example demonstrates how to parse documents using the MixedbreadDocumentParser.
The parser can handle various file formats including PDF, DOCX, PPTX, and images.
"""

from mixedbread_ai_haystack import MixedbreadDocumentParser

def main():
    print("=== Mixedbread AI Document Parsing Example ===\n")
    
    # Example 1: Basic document parsing
    print("1. Basic Document Parsing")
    print("-" * 40)
    
    parser = MixedbreadDocumentParser(
        chunking_strategy="page",
        return_format="markdown",
        element_types=["text", "title", "list-item", "table"],
    )
    
    try:
        # Parse the example PDF file
        result = parser.run(sources=["data/acme_invoice.pdf"])
        documents = result["documents"]
        
        print(f"✓ Successfully parsed {len(documents)} chunks from the document")
        print(f"✓ Model used: {result['meta'].get('model', 'N/A')}")
        print(f"✓ Return format: {result['meta'].get('return_format', 'N/A')}")
        
        # Show detailed information about parsed chunks
        for i, doc in enumerate(documents[:3]):  # Show first 3 chunks
            print(f"\nChunk {i + 1}:")
            print(f"  Content: {doc.content[:150]}{'...' if len(doc.content) > 150 else ''}")
            print(f"  Elements: {doc.meta.get('element_count', 0)}")
            print(f"  Page range: {doc.meta.get('page_range', 'N/A')}")
            
        if len(documents) > 3:
            print(f"\n... and {len(documents) - 3} more chunks")
            
    except Exception as e:
        print(f"✗ Error parsing document: {e}")
        print("  Make sure the file exists or replace with your own PDF file")
    
    # Example 2: Different chunking strategies
    print(f"\n\n2. Different Chunking Strategies")
    print("-" * 40)
    
    strategies = ["page", "section"]
    
    for strategy in strategies:
        try:
            parser_strategy = MixedbreadDocumentParser(
                chunking_strategy=strategy,
                return_format="plain",
                element_types=["text", "title"],
            )
            
            result = parser_strategy.run(sources=["data/acme_invoice.pdf"])
            documents = result["documents"]
            
            print(f"✓ Strategy '{strategy}': {len(documents)} chunks")
            
        except Exception as e:
            print(f"✗ Error with strategy '{strategy}': {e}")
    
    # Example 3: Different element types
    print(f"\n\n3. Element Type Filtering")
    print("-" * 40)
    
    element_configs = [
        (["text"], "Text only"),
        (["title"], "Titles only"),
        (["text", "title", "table"], "Text, titles, and tables"),
    ]
    
    for elements, description in element_configs:
        try:
            parser_elements = MixedbreadDocumentParser(
                element_types=elements,
                return_format="markdown",
            )
            
            result = parser_elements.run(sources=["data/acme_invoice.pdf"])
            documents = result["documents"]
            
            print(f"✓ {description}: {len(documents)} chunks")
            
        except Exception as e:
            print(f"✗ Error with element filter '{description}': {e}")
    
    print(f"\n\n4. Parsing Metadata Information")
    print("-" * 40)
    
    try:
        result = parser.run(sources=["data/acme_invoice.pdf"])
        if result["documents"]:
            doc = result["documents"][0]
            metadata = doc.meta
            
            print("Available metadata fields:")
            for key, value in metadata.items():
                if key not in ['elements']:  # Skip complex nested data
                    print(f"  {key}: {value}")
                    
    except Exception as e:
        print(f"✗ Error showing metadata: {e}")
    
    print(f"\n\n=== Parsing Complete ===")
    print("Try replacing 'data/acme_invoice.pdf' with your own file path!")
    print("Supported formats: PDF, DOCX, PPTX, images")

if __name__ == "__main__":
    main()
