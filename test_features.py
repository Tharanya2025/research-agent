from rag import get_paper_text, list_sources
from agent import run_agent
from verifier import verify_answer
from reproducibility import check_reproducibility
from litreview import review_papers

print("Papers indexed:", list_sources())

print("\n--- 1. Answer Verifier ---")
answer, _, log, evidence = run_agent("What is this paper about?")
result = verify_answer(answer, "\n\n".join(evidence))
print(f"Faithfulness score: {result['score']:.0%}")
for c in result["claims"]:
    print(f"  [{c['verdict']}] {c['claim']}")

print("\n--- 2. Reproducibility Checker ---")
res = check_reproducibility(get_paper_text("paper.pdf"))
print(f"Score: {res['score']:.0%}")
for r in res["rows"]:
    print(f"  {r['field']}: {r['value']}")
print("Not reported:", res["missing"])

print("\n--- 3. Literature Review (one paper, just a test) ---")
summaries, draft = review_papers(["paper.pdf"], get_paper_text)
print(summaries[0])
print()
print(draft)