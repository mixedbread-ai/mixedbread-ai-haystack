from mixedbread_ai_haystack import MixedbreadTextEmbedder

text_embedder = MixedbreadTextEmbedder()

sentence = "German rye bread is known for its dense texture and rich flavor."
result = text_embedder.run(text=sentence)

print(f"Text: {sentence}")
print(f"Embedding vector: {result['embedding']}")
print(f"Embedding dimension: {len(result['embedding'])}")
