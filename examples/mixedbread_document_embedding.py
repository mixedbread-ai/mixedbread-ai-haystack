from haystack import Document, Pipeline
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.document_stores.in_memory import InMemoryDocumentStore

from mixedbread_ai_haystack import MixedbreadDocumentEmbedder, MixedbreadTextEmbedder

document_store = InMemoryDocumentStore(embedding_similarity_function="cosine")

documents = [
    Document(content="German rye bread is known for its dense texture and rich flavor"),
    Document(content="Pretzels are a traditional German baked good often served with beer"),
    Document(content="Black Forest cake is a famous German dessert made with chocolate and cherries"),
]

document_embedder = MixedbreadDocumentEmbedder()
documents_with_embeddings = document_embedder.run(documents)["documents"]
document_store.write_documents(documents_with_embeddings)

query_pipeline = Pipeline()
query_pipeline.add_component("text_embedder", MixedbreadTextEmbedder())
query_pipeline.add_component("retriever", InMemoryEmbeddingRetriever(document_store=document_store))
query_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")

query = "What is German bread like?"

result = query_pipeline.run({"text_embedder": {"text": query}})

print(result["retriever"]["documents"][0])
