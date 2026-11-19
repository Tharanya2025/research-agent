import os

os.environ["HF_HUB_OFFLINE"] = "1"  # the model is already downloaded, so skip network checks

import chromadb
from pypdf import PdfReader

_client = chromadb.PersistentClient(path="chroma_db")
_col = _client.get_or_create_collection("docs")
_model = None


def get_model():
    """Load the embedding model only when it is first needed."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def chunk(text, size=800, overlap=150):
    step = size - overlap
    pieces = [text[i:i + size] for i in range(0, len(text), step)]
    return [p for p in pieces if p.strip()]


def ingest_pdf(path):
    name = os.path.basename(path)
    _col.delete(where={"source": name})  # re-indexing the same file replaces it
    ids, docs, metas = [], [], []
    for page_no, page in enumerate(PdfReader(path).pages, start=1):
        for j, c in enumerate(chunk(page.extract_text() or "")):
            ids.append(f"{name}-{page_no}-{j}")
            docs.append(c)
            metas.append({"source": name, "page": page_no})
    model = get_model()
    for i in range(0, len(docs), 500):
        batch = docs[i:i + 500]
        _col.upsert(
            ids=ids[i:i + 500],
            documents=batch,
            metadatas=metas[i:i + 500],
            embeddings=model.encode(batch).tolist(),
        )
    return len(docs)


def list_sources():
    if _col.count() == 0:
        return []
    metas = _col.get(include=["metadatas"])["metadatas"]
    return sorted({m["source"] for m in metas})


def delete_source(name):
    _col.delete(where={"source": name})


def search_docs(query, k=5, source=None):
    if _col.count() == 0:
        return []
    where = {"source": source} if source and source in list_sources() else None
    r = _col.query(
        query_embeddings=get_model().encode([query]).tolist(),
        n_results=k,
        where=where,
    )
    return [
        {"text": d, "source": m["source"], "page": m["page"]}
        for d, m in zip(r["documents"][0], r["metadatas"][0])
    ]


def get_paper_text(source, max_chars=60000):
    data = _col.get(where={"source": source}, include=["documents", "metadatas"])
    items = sorted(
        zip(data["ids"], data["metadatas"], data["documents"]),
        key=lambda x: (x[1]["page"], int(x[0].rsplit("-", 1)[1])),
    )
    text = "\n".join(f"[p.{m['page']}] {d}" for _, m, d in items)
    return text[:max_chars]