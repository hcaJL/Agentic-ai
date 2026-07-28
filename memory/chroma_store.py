"""Layer-2 theory memory backed by ChromaDB.

`query_theory(query, k)` returns the Chinese theory snippets whose English
keyword text best matches the query built from the pipeline's event vocabulary
(event types / motifs / phase / piece names — see retrieve_memory_node).

Follows the project's offline-first contract: if chromadb (or its embedding
model download) is unavailable, falls back to plain keyword-overlap scoring so
the pipeline keeps running — just with cruder retrieval.
"""
import os

from memory.theory_seed import THEORY

_DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db")
_collection = None
_fallback = False


def _get_collection():
    global _collection, _fallback
    if _collection is not None or _fallback:
        return _collection
    try:
        import chromadb
        client = chromadb.PersistentClient(path=_DB_PATH)
        col = client.get_or_create_collection("chess_theory")
        if col.count() < len(THEORY):  # seed once (or re-seed after corpus grows)
            col.upsert(
                ids=[t["id"] for t in THEORY],
                documents=[t["keywords"] for t in THEORY],  # embedded side
                metadatas=[{"text": t["text"]} for t in THEORY],
            )
        _collection = col
    except Exception:
        _fallback = True  # no chromadb / no embedding model -> keyword overlap
    return _collection


def _keyword_fallback(query: str, k: int) -> list:
    q = set(query.lower().split())
    scored = sorted(
        THEORY,
        key=lambda t: len(q & set(t["keywords"].split())),
        reverse=True,
    )
    return [t["text"] for t in scored[:k] if q & set(t["keywords"].split())]


def query_theory(query: str, k: int = 2) -> list:
    """Return up to k theory snippets (Chinese) relevant to the query string."""
    col = _get_collection()
    if col is None:
        return _keyword_fallback(query, k)
    try:
        res = col.query(query_texts=[query], n_results=k)
        return [m["text"] for m in res["metadatas"][0]]
    except Exception:
        return _keyword_fallback(query, k)
