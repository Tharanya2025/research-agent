import csv
import time

from agent import run_agent
from verifier import verify_answer

# Each "expect" is a list of groups. The answer must contain at least one
# word or phrase from EVERY group to count as correct.
NOT_FOUND = ["not provide", "not mention", "not contain", "not state", "not specif",
             "does not", "doesn't", "no link", "no github", "cannot", "can't",
             "unable", "not available", "not found", "no information"]

QUESTIONS = [
    {"q": "How many student-term records are in the dataset of paper.pdf?", "expect": [["1,000", "1000"]]},
    {"q": "Which explanation method does paper.pdf use?", "expect": [["SHAP"]]},
    {"q": "Which models are compared in paper.pdf?", "expect": [["logistic"], ["decision tree"], ["random forest"]]},
    {"q": "What F1-score did the random forest get in paper.pdf?", "expect": [["0.77"]]},
    {"q": "Why does paper.pdf prefer a chronological split over a random split?", "expect": [["leakage"]]},
    {"q": "What learning rate was used for gradient boosting in paper2.pdf?", "expect": [["0.05"]]},
    {"q": "How many records are in the dataset of paper2.pdf?", "expect": [["20,000", "20000"]]},
    {"q": "Which model performed best in paper2.pdf?", "expect": [["gradient boosting"]]},
    {"q": "What random seed was used in paper2.pdf?", "expect": [["42"]]},
    {"q": "Which explanation method does paper2.pdf use?", "expect": [["LIME"]]},
    # Trap question: the paper does not give this, so the agent should say so
    {"q": "What is the GitHub link to the code for paper.pdf?", "expect": [NOT_FOUND]},
]

FIELDS = ["question", "correct", "faithfulness", "used_search", "seconds", "error", "answer"]


def matches(answer, expect):
    a = answer.lower()
    return all(any(alt.lower() in a for alt in group) for group in expect)


def save(rows):
    with open("eval_results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


rows = []
for i, item in enumerate(QUESTIONS, 1):
    print(f"[{i}/{len(QUESTIONS)}] {item['q']}", flush=True)
    start = time.time()
    row = {"question": item["q"], "correct": False, "faithfulness": "", "used_search": False,
           "seconds": "", "error": "", "answer": ""}
    try:
        answer, _, log, evidence = run_agent(item["q"])
        row["answer"] = answer.replace("\n", " ")
        row["correct"] = matches(answer, item["expect"])
        row["used_search"] = any(name == "search_documents" for name, _ in log)
        if evidence and not answer.startswith("I couldn't"):
            try:
                row["faithfulness"] = round(verify_answer(answer, "\n\n".join(evidence))["score"], 3)
            except Exception as e:
                print("   verifier failed:", e)
    except Exception as e:
        row["error"] = str(e)[:200]
        print("   FAILED:", e)
    row["seconds"] = round(time.time() - start, 1)
    print(f"   correct={row['correct']}  faithfulness={row['faithfulness']}  time={row['seconds']}s", flush=True)
    rows.append(row)
    save(rows)      # saved after every question, so a crash never loses results
    time.sleep(3)   # be gentle with the free tier

n = len(rows)
ok = [r for r in rows if not r["error"]]
faith = [r["faithfulness"] for r in ok if r["faithfulness"] != ""]
print("\n=========== RESULTS ===========")
print(f"Questions run:        {n} ({n - len(ok)} failed with errors)")
print(f"Answer accuracy:      {sum(r['correct'] for r in rows) / n:.0%}")
print(f"Used document search: {sum(r['used_search'] for r in rows) / n:.0%}")
if faith:
    print(f"Avg faithfulness:     {sum(faith) / len(faith):.0%}")
print("Details saved to eval_results.csv")