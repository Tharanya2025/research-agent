from llm import ask, ask_json

SUMMARY_PROMPT = """Summarize this research paper. Using ONLY the text, return ONE JSON object with keys:
"title", "problem", "method", "dataset", "key_results", "limitations".
Use "not stated" if something is absent. Return ONLY JSON."""

REVIEW_PROMPT = """You are writing the 'Related Work' section of a paper.
Using ONLY the paper summaries given, write 3 to 5 coherent paragraphs that group the papers by theme,
compare their methods and results, and point out gaps. Cite each paper as [filename].
Do not invent facts that are not in the summaries."""


def review_papers(sources, get_text):
    summaries = []
    for s in sources:
        info = ask_json(SUMMARY_PROMPT, get_text(s))
        if isinstance(info, list):
            info = info[0] if info else {}
        info["file"] = s
        summaries.append(info)
    summary_text = "\n\n".join(f"[{i['file']}]\n{i}" for i in summaries)
    draft = ask(REVIEW_PROMPT, summary_text)
    return summaries, draft