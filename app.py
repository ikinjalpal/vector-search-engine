"""
app.py
------
Flask API that exposes the vector search engine over HTTP:

  POST /index   - add a new document, embed it, and insert it into
                  both the brute-force store and the live LSH index
  POST /search  - embed a query and return its nearest neighbors,
                  using either the LSH index (default, fast/approximate)
                  or brute force (exact) via ?method=brute_force
  GET  /stats   - basic info about the current in-memory index
                  (document count, embedding dimension, LSH config)

State (the loaded embeddings + LSH index) lives in memory for the
lifetime of the process; documents added via /index are also persisted
to the SQLite `documents` table and appended to the .npy embedding
files so they survive a restart.
"""

import os
import time

import numpy as np
from flask import Flask, jsonify, request, send_from_directory

from brute_force import brute_force_search
from data.loader import DB_PATH, get_all_documents
from embedder import IDS_PATH, VECTORS_PATH, get_embedding, load_embeddings
from lsh import LSHIndex

app = Flask(__name__, static_folder="static", static_url_path="")

N_BITS = 6
N_TABLES = 15

# --- In-memory index state, populated at startup by load_state() ---
state = {
    "vectors": None,       # (n_docs, dim) float32 matrix, row-aligned with "ids"
    "ids": None,           # (n_docs,) int64 array of document ids
    "documents": {},       # id -> text
    "lsh_index": None,     # LSHIndex built over the current vectors/ids
    "next_id": 0,          # next id to assign to a newly indexed document
}


def load_state():
    """Loads persisted embeddings + documents from disk and builds the LSH index."""
    vectors, ids = load_embeddings()
    documents = dict(get_all_documents())

    lsh_index = LSHIndex(dim=vectors.shape[1], n_bits=N_BITS, n_tables=N_TABLES)
    lsh_index.build(ids.tolist(), vectors)

    state["vectors"] = vectors
    state["ids"] = ids
    state["documents"] = documents
    state["lsh_index"] = lsh_index
    state["next_id"] = (int(ids.max()) + 1) if len(ids) else 0

    app.logger.info(
        "Loaded %d documents (%d-dim) and built LSH index (n_bits=%d, n_tables=%d)",
        len(ids), vectors.shape[1] if len(vectors) else 0, N_BITS, N_TABLES,
    )


def add_document_to_sqlite(doc_id: int, text: str):
    import sqlite3

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("INSERT INTO documents (id, text) VALUES (?, ?)", (doc_id, text))
    conn.commit()
    conn.close()


@app.route("/index", methods=["POST"])
def index_document():
    """
    Body: {"text": "some document text"}
    Embeds the text, appends it to the in-memory vectors/ids and LSH
    index, persists it to SQLite (documents table) and the .npy
    embedding files, and returns the assigned document id.
    """
    payload = request.get_json(silent=True) or {}
    text = payload.get("text")
    if not text or not isinstance(text, str):
        return jsonify({"error": "Request body must include a non-empty 'text' string."}), 400

    doc_id = state["next_id"]
    vector = get_embedding(text)

    state["vectors"] = np.vstack([state["vectors"], vector[None, :]])
    state["ids"] = np.append(state["ids"], np.int64(doc_id))
    state["documents"][doc_id] = text
    state["lsh_index"].build([doc_id], vector[None, :])
    state["next_id"] += 1

    add_document_to_sqlite(doc_id, text)
    np.save(VECTORS_PATH, state["vectors"])
    np.save(IDS_PATH, state["ids"])

    return jsonify({"id": doc_id, "text": text, "message": "Document indexed."}), 201


@app.route("/search", methods=["POST"])
def search():
    """
    Body: {"query": "search text", "top_k": 10, "method": "lsh" | "brute_force"}
    Embeds the query and returns the top_k most similar indexed
    documents. Defaults to the LSH index; pass "method": "brute_force"
    (or ?method=brute_force as a query string) to use the exact
    O(n) baseline instead, e.g. for comparing result quality.
    """
    payload = request.get_json(silent=True) or {}
    query_text = payload.get("query")
    if not query_text or not isinstance(query_text, str):
        return jsonify({"error": "Request body must include a non-empty 'query' string."}), 400

    top_k = int(payload.get("top_k", 10))
    method = payload.get("method") or request.args.get("method", "lsh")

    if state["vectors"] is None or len(state["ids"]) == 0:
        return jsonify({"error": "Index is empty. POST to /index first."}), 400

    query_vector = get_embedding(query_text)

    start = time.perf_counter()
    if method == "brute_force":
        raw_results = brute_force_search(query_vector, state["vectors"], top_k=top_k)
        results = [(int(state["ids"][row_idx]), score) for row_idx, score in raw_results]
    elif method == "lsh":
        results = state["lsh_index"].search(query_vector, top_k=top_k)
    else:
        return jsonify({"error": f"Unknown method '{method}'. Use 'lsh' or 'brute_force'."}), 400
    latency_ms = (time.perf_counter() - start) * 1000

    formatted = [
        {
            "id": doc_id,
            "score": round(score, 4),
            "text": state["documents"].get(doc_id, "<missing>"),
        }
        for doc_id, score in results
    ]

    return jsonify({
        "query": query_text,
        "method": method,
        "latency_ms": round(latency_ms, 3),
        "results": formatted,
    })


@app.route("/stats", methods=["GET"])
def stats():
    """Returns basic information about the currently loaded index."""
    n_docs = len(state["ids"]) if state["ids"] is not None else 0
    dim = int(state["vectors"].shape[1]) if state["vectors"] is not None and len(state["vectors"]) else 0

    return jsonify({
        "num_documents": n_docs,
        "embedding_dim": dim,
        "lsh_n_bits": N_BITS,
        "lsh_n_tables": N_TABLES,
        "db_path": DB_PATH,
    })


@app.route("/", methods=["GET"])
def root():
    """Serves the minimal search demo page (static/index.html)."""
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api", methods=["GET"])
def api_info():
    return jsonify({
        "service": "vector-search-engine",
        "endpoints": {
            "POST /index": "{'text': str} -> embeds and indexes a new document",
            "POST /search": "{'query': str, 'top_k': int, 'method': 'lsh'|'brute_force'} -> nearest neighbors",
            "GET /stats": "current index statistics",
        },
    })


load_state()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
