from mixedbread_ai_haystack import MixedbreadDocumentParser

parser = MixedbreadDocumentParser()
result = parser.run(sources=["./data/report.pdf"])
documents = result["documents"]

print(documents[0])
