from rag import delete_source, list_sources

print("Before:", list_sources())
delete_source("paper.pdf.pdf")
print("After:", list_sources())