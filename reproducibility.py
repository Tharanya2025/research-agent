from llm import ask_json

FIELDS = [
    "dataset", "train_test_split", "preprocessing", "model_architecture",
    "hyperparameters", "optimizer_and_learning_rate", "evaluation_metrics",
    "reported_results", "hardware", "random_seed", "code_link",
]

PROMPT = f"""You are auditing a research paper for reproducibility.
Using ONLY the paper text, fill in each field. If the paper does not state it, use null. Never guess.
Return ONLY a JSON list of objects with keys "field", "value" (text or null), and "page" (number or null).
Fields, in order: {FIELDS}"""


def check_reproducibility(paper_text):
    rows = ask_json(PROMPT, paper_text)
    if isinstance(rows, dict):
        rows = rows.get("rows", [])
    reported = [r for r in rows if r.get("value")]
    missing = [r["field"] for r in rows if not r.get("value")]
    score = len(reported) / len(rows) if rows else 0.0
    return {"rows": rows, "score": score, "missing": missing}