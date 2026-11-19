import json

from ddgs import DDGS
from google.genai import types

from rag import search_docs

search_documents_decl = types.FunctionDeclaration(
    name="search_documents",
    description=(
        "Search the user's uploaded PDFs. Use this first for any question that may be "
        "answered by their documents. If the user names a specific file, pass its name "
        "in 'source' so only that file is searched."
    ),
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "query": types.Schema(type=types.Type.STRING, description="What to look for"),
            "source": types.Schema(
                type=types.Type.STRING,
                description="Optional file name to restrict the search to, e.g. paper2.pdf",
            ),
        },
        required=["query"],
    ),
)

web_search_decl = types.FunctionDeclaration(
    name="web_search",
    description="Search the web for current or general information not found in the uploaded documents.",
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "query": types.Schema(type=types.Type.STRING, description="The search query"),
        },
        required=["query"],
    ),
)

TOOLS = [types.Tool(function_declarations=[search_documents_decl, web_search_decl])]


def run_tool(name, args):
    try:
        if name == "search_documents":
            hits = search_docs(args["query"], source=args.get("source"))
            return json.dumps(hits) if hits else "No relevant passages found in the documents."
        if name == "web_search":
            results = DDGS().text(args["query"], max_results=5)
            return json.dumps(
                [{"title": r["title"], "url": r["href"], "snippet": r["body"]} for r in results]
            )
        return f"Unknown tool: {name}"
    except Exception as e:
        return f"Tool error: {e}"