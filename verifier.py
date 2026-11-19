from llm import ask_json

PROMPT = """You are a strict fact-checker. You get an ANSWER and the SOURCE PASSAGES it was based on.
Split the answer into individual factual claims (ignore filler and citations).
For each claim return an object with these keys:
"claim": the claim text,
"verdict": one of "supported", "partially_supported", "unsupported",
"evidence": a short supporting quote or page reference, or an empty string.
A claim is supported ONLY if the passages clearly state it.
Return ONLY a JSON list of these objects."""

WEIGHT = {"supported": 1.0, "partially_supported": 0.5, "unsupported": 0.0}


def verify_answer(answer, evidence_text):
    claims = ask_json(PROMPT, f"ANSWER:\n{answer}\n\nSOURCE PASSAGES:\n{evidence_text}")
    if isinstance(claims, dict):  # sometimes the model wraps the list in an object
        claims = claims.get("claims", [])
    score = sum(WEIGHT.get(c.get("verdict"), 0) for c in claims) / len(claims) if claims else 0.0
    return {"claims": claims, "score": score}