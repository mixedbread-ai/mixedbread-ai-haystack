from haystack import Document
from mixedbread_ai_haystack import MixedbreadReranker

documents = [
    Document(
        content="German rye bread is known for its dense texture and rich flavor",
        meta={"category": "bread", "region": "Germany", "type": "traditional"},
    ),
    Document(
        content="Pretzels are a traditional German baked good often served with beer",
        meta={"category": "snack", "region": "Germany", "type": "traditional"},
    ),
    Document(
        content="Black Forest cake is a famous German dessert made with chocolate and cherries",
        meta={"category": "dessert", "region": "Germany", "type": "cake"},
    ),
    Document(
        content="Stollen is a traditional German Christmas bread with dried fruits",
        meta={"category": "bread", "region": "Germany", "type": "seasonal"},
    ),
    Document(
        content="German bakeries are famous for their sourdough bread recipes",
        meta={"category": "bread", "region": "Germany", "type": "modern"},
    ),
    Document(
        content="French baguettes have a crispy crust and soft interior",
        meta={"category": "bread", "region": "France", "type": "traditional"},
    ),
    Document(
        content="Italian focaccia is a flat bread topped with herbs and olive oil",
        meta={"category": "bread", "region": "Italy", "type": "traditional"},
    ),
]

print("=== Document Reranking Examples ===\n")

# Example 1: Basic reranking with different models
queries = [
    "Tell me about German bread traditions",
    "What desserts are popular in Germany?",
    "How do different bread types compare?",
]

models = ["mixedbread-ai/mxbai-rerank-large-v2", "mixedbread-ai/mxbai-rerank-base-v1"]

for query in queries:
    print(f"Query: '{query}'\n")

    for model in models:
        print(f"Using model: {model}")

        reranker = MixedbreadReranker(top_k=3, model=model)

        try:
            result = reranker.run(documents=documents, query=query)

            print("Top 3 results:")
            for i, doc in enumerate(result["documents"], 1):
                score = doc.meta.get("rerank_score", "N/A")
                print(f"  {i}. [Score: {score:.4f}] {doc.content}")
                print(
                    f"     Category: {doc.meta.get('category', 'N/A')}, Region: {doc.meta.get('region', 'N/A')}"
                )
            print()

        except Exception as e:
            print(f"  Error: {e}\n")

    print("-" * 70)

# Example 2: Reranking with different top_k values
print("\nExample 2: Reranking with Different Top-K Values")

reranker_top5 = MixedbreadReranker(
    top_k=5,
    model="mixedbread-ai/mxbai-rerank-large-v2",
)

query = "German baking traditions"
print(f"Query: '{query}'")
print("Ranking with top_k=5\n")

try:
    result = reranker_top5.run(documents=documents, query=query)

    print("Reranked results:")
    for i, doc in enumerate(result["documents"], 1):
        score = doc.meta.get("rerank_score", 0)
        print(f"{i}. [Score: {score:.4f}] {doc.content}")
        print(
            f"   {doc.meta.get('region', 'Unknown')} | {doc.meta.get('category', 'Unknown')} | {doc.meta.get('type', 'Unknown')}"
        )
        print()

except Exception as e:
    print(f"Error: {e}")
