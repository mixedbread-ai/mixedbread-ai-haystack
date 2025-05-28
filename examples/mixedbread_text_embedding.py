from mixedbread_ai_haystack import MixedbreadTextEmbedder

# Initialize with a specific model (default is mixedbread-ai/mxbai-embed-large-v1)
text_embedder = MixedbreadTextEmbedder(
    model="mixedbread-ai/mxbai-embed-large-v1",
)

queries = [
    "What are the benefits of sourdough bread?",
    "How to make traditional German pretzels?",
    "Compare nutritional value of different bread types",
]

print("=== Text Embedding Examples ===\n")

for i, query in enumerate(queries, 1):
    try:
        result = text_embedder.run(text=query)
        embedding = result["embedding"]

        print(f"Query {i}: {query}")
        print(f"Embedding dimension: {len(embedding)}")
        print(f"First 5 values: {embedding[:5]}")
        print("-" * 50)

    except Exception as e:
        print(f"Error processing query {i}: {e}")
        print("-" * 50)
