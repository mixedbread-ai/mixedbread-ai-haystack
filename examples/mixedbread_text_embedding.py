"""
Mixedbread AI Text Embedding Example

This example demonstrates various features of the MixedbreadTextEmbedder including:
- Different models and configurations
- Various encoding formats and dimensions
- Custom prompts for instruction-based embedding
- Performance comparison between configurations
"""

import time
from mixedbread_ai_haystack import MixedbreadTextEmbedder
from mixedbread_ai_haystack.embedders.embedding_types import MixedbreadEmbeddingType

def main():
    print("=== Mixedbread AI Text Embedding Example ===\n")
    
    # Sample queries for testing
    queries = [
        "What are the benefits of sourdough bread?",
        "How to make traditional German pretzels?",
        "Compare nutritional value of different bread types",
    ]
    
    # Example 1: Basic text embedding
    print("1. Basic Text Embedding")
    print("-" * 50)
    
    # Initialize with default settings
    text_embedder = MixedbreadTextEmbedder(
        model="mixedbread-ai/mxbai-embed-large-v1",
    )
    
    for i, query in enumerate(queries, 1):
        try:
            start_time = time.time()
            result = text_embedder.run(text=query)
            end_time = time.time()
            
            embedding = result["embedding"]
            meta = result["meta"]
            
            print(f"Query {i}: {query}")
            print(f"✓ Embedding dimensions: {len(embedding)}")
            print(f"✓ Model: {meta.get('model', 'N/A')}")
            print(f"✓ Normalized: {meta.get('normalized', 'N/A')}")
            print(f"✓ Processing time: {(end_time - start_time)*1000:.1f}ms")
            print(f"✓ First 5 values: {[round(x, 4) for x in embedding[:5]]}")
            print(f"✓ Token usage: {meta.get('usage', {})}")
            print()

        except Exception as e:
            print(f"✗ Error processing query {i}: {e}")
            print()
    
    # Example 2: Different encoding formats
    print("2. Different Encoding Formats")
    print("-" * 50)
    
    formats = [
        (MixedbreadEmbeddingType.FLOAT, "Float (default)"),
        (MixedbreadEmbeddingType.BASE64, "Base64 encoded"),
    ]
    
    test_text = queries[0]  # Use first query for comparison
    
    for encoding_format, description in formats:
        try:
            embedder = MixedbreadTextEmbedder(
                model="mixedbread-ai/mxbai-embed-large-v1",
                encoding_format=encoding_format,
            )
            
            result = embedder.run(text=test_text)
            embedding = result["embedding"]
            meta = result["meta"]
            
            print(f"✓ {description}:")
            print(f"  Format: {meta.get('encoding_format', 'N/A')}")
            print(f"  Type: {type(embedding).__name__}")
            if encoding_format == MixedbreadEmbeddingType.FLOAT:
                print(f"  Sample values: {[round(x, 4) for x in embedding[:3]]}")
            else:
                print(f"  Length: {len(embedding) if hasattr(embedding, '__len__') else 'N/A'}")
            print()
            
        except Exception as e:
            print(f"✗ Error with {description}: {e}")
            print()
    
    # Example 3: Different embedding dimensions
    print("3. Different Embedding Dimensions")
    print("-" * 50)
    
    dimensions = [256, 512, 1024]
    
    for dim in dimensions:
        try:
            embedder = MixedbreadTextEmbedder(
                model="mixedbread-ai/mxbai-embed-large-v1",
                dimensions=dim,
            )
            
            result = embedder.run(text=test_text)
            embedding = result["embedding"]
            meta = result["meta"]
            
            print(f"✓ Requested dimensions: {dim}")
            print(f"  Actual dimensions: {len(embedding)}")
            print(f"  Meta dimensions: {meta.get('dimensions', 'N/A')}")
            print()
            
        except Exception as e:
            print(f"✗ Error with dimensions {dim}: {e}")
            print()
    
    # Example 4: Custom prompts for instruction-based embedding  
    print("4. Custom Prompts")
    print("-" * 50)
    
    prompts = [
        (None, "No custom prompt"),
        ("Represent this query for retrieving relevant baking documents:", "Baking-focused prompt"),
        ("Generate embeddings for semantic search:", "General search prompt"),
    ]
    
    for prompt, description in prompts:
        try:
            embedder = MixedbreadTextEmbedder(
                model="mixedbread-ai/mxbai-embed-large-v1",
                prompt=prompt,
            )
            
            result = embedder.run(text=test_text)
            embedding = result["embedding"]
            
            print(f"✓ {description}:")
            print(f"  Prompt: {prompt or 'None'}")
            print(f"  Embedding range: [{min(embedding):.4f}, {max(embedding):.4f}]")
            print(f"  Mean value: {sum(embedding)/len(embedding):.4f}")
            print()
            
        except Exception as e:
            print(f"✗ Error with prompt '{description}': {e}")
            print()
    
    # Example 5: Model comparison
    print("5. Model Comparison")
    print("-" * 50)
    
    models = [
        "mixedbread-ai/mxbai-embed-large-v1",
        "mixedbread-ai/mxbai-embed-2d-large-v1",
    ]
    
    for model in models:
        try:
            embedder = MixedbreadTextEmbedder(model=model)
            
            start_time = time.time()
            result = embedder.run(text=test_text)
            end_time = time.time()
            
            embedding = result["embedding"]
            meta = result["meta"]
            
            print(f"✓ Model: {model}")
            print(f"  Dimensions: {len(embedding)}")
            print(f"  Processing time: {(end_time - start_time)*1000:.1f}ms")
            print(f"  Normalized: {meta.get('normalized', 'N/A')}")
            print()
            
        except Exception as e:
            print(f"✗ Error with model {model}: {e}")
            print()
    
    # Example 6: Async processing
    print("6. Async Processing")
    print("-" * 50)
    
    try:
        import asyncio
        
        async def embed_async():
            embedder = MixedbreadTextEmbedder(
                model="mixedbread-ai/mxbai-embed-large-v1"
            )
            
            start_time = time.time()
            result = await embedder.run_async(text=test_text)
            end_time = time.time()
            
            embedding = result["embedding"]
            print(f"✓ Async embedding completed:")
            print(f"  Dimensions: {len(embedding)}")
            print(f"  Processing time: {(end_time - start_time)*1000:.1f}ms")
            
        # Run async example
        asyncio.run(embed_async())
        
    except Exception as e:
        print(f"✗ Error with async processing: {e}")
    
    print("\n=== Text Embedding Examples Complete ===")
    print("Try experimenting with different models, prompts, and configurations!")

if __name__ == "__main__":
    main()
