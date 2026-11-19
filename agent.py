from google.genai import types

from llm import generate
from rag import list_sources
from tools import TOOLS, run_tool

BASE_SYSTEM = """You are a research assistant. Use search_documents first for questions
about the user's files, and web_search for anything else or to verify facts.
Cite sources as (filename, p. N) for documents and with the URL for web results.
If the sources don't contain the answer, say so instead of guessing.
Make at most 2 or 3 searches, then write your answer from what you found.
When the user names a specific file, pass that file name in the 'source' argument."""


def make_config(mode="AUTO"):
    sources = list_sources()
    system = BASE_SYSTEM + (
        "\n\nThe user has these papers indexed: " + (", ".join(sources) or "none") + ". "
        "Phrases like 'this paper' or 'the paper' refer to these papers. "
        "Never ask the user to upload or name a paper; search the indexed ones instead."
    )
    if mode == "ANY" and not sources:
        mode = "AUTO"
    return types.GenerateContentConfig(
        system_instruction=system,
        tools=TOOLS,
        temperature=0.2,
        tool_config=types.ToolConfig(
            function_calling_config=types.FunctionCallingConfig(mode=mode)
        ),
        # We run the loop ourselves so we can log tools and collect evidence
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )


def run_agent(question, history=None, max_steps=6):
    contents = list(history or [])
    contents.append(types.Content(role="user", parts=[types.Part.from_text(text=question)]))
    tool_log, evidence = [], []

    for step in range(max_steps):
        if step == 0:
            mode = "ANY"              # must search at least once
        elif step >= max_steps - 1:
            mode = "NONE"             # last step: no more tools, write the answer
        else:
            mode = "AUTO"

        resp = generate(contents, make_config(mode))

        if not resp.candidates or resp.candidates[0].content is None:
            return "The model returned no answer. Try rephrasing your question.", contents, tool_log, evidence

        contents.append(resp.candidates[0].content)
        calls = resp.function_calls

        if not calls:  # no more tools needed, so this is the final answer
            answer = (resp.text or "").strip() or "(empty answer)"
            return answer, contents, tool_log, evidence

        parts = []
        for call in calls:
            args = dict(call.args or {})
            out = run_tool(call.name, args)
            tool_log.append((call.name, args))
            evidence.append(out)
            parts.append(types.Part.from_function_response(name=call.name, response={"result": out}))
        contents.append(types.Content(role="user", parts=parts))

    return "I couldn't finish within the step limit.", contents, tool_log, evidence