from haystack import Document

from mixedbread_ai_haystack import MixedbreadReranker

documents = [
    Document(content="German rye bread is known for its dense texture and rich flavor"),
    Document(content="Pretzels are a traditional German baked good often served with beer"),
    Document(content="Black Forest cake is a famous German dessert made with chocolate and cherries"),
    Document(content="Stollen is a traditional German Christmas bread with dried fruits"),
    Document(content="German bakeries are famous for their sourdough bread recipes"),
]

reranker = MixedbreadReranker(top_k=3, model="mixedbread-ai/mxbai-rerank-large-v2")
result = reranker.run(documents=documents, query="Tell me about German bread traditions")

for i, doc in enumerate(result["documents"], 1):
    print(f"{i}. {doc.content}")
