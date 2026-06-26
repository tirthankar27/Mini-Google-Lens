from backend.retrieval import search_text

results = search_text("golden retriever")

for r in results:
    print(r)