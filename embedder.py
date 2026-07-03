"""
embedder.py
-----------
Turns raw text into dense 384-dimensional vectors using the
sentence-transformers "all-MiniLM-L6-v2" model, and persists the
resulting document vectors to a single .npy matrix on disk so
brute_force.py, lsh.py, benchmark.py, and app.py can all load the
same embeddings without re-running the (relatively expensive) model.
"""

import os

import numpy as np

MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
VECTORS_PATH = os.path.join(_PROJECT_ROOT, "embeddings.npy")
IDS_PATH = os.path.join(_PROJECT_ROOT, "embeddings_ids.npy")

_model = None  # lazily loaded singleton so importing this module is cheap


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(MODEL_NAME)
    return _model


def get_embedding(text: str) -> np.ndarray:
    """
    Embeds a single piece of text into a 384-dim numpy vector.
    Used for one-off queries (search requests, indexing a new document).
    """
    model = _get_model()
    vector = model.encode(text, convert_to_numpy=True, normalize_embeddings=False)
    return vector.astype(np.float32)


def embed_documents(ids, texts, batch_size: int = 64) -> np.ndarray:
    """
    Embeds a batch of documents at once (much faster than calling
    get_embedding in a loop) and returns an (n_docs, 384) matrix in
    the same row order as `ids`/`texts`.
    """
    model = _get_model()
    vectors = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )
    return vectors.astype(np.float32)


def build_and_save_embeddings(db_path: str = None, chunk_size: int = 2000, max_seconds: float = None):
    """
    Reads all documents from SQLite (via data/loader.py) and embeds them
    in resumable chunks, saving progress to disk after every chunk:
      - embeddings.npy      -> shape (n_docs, 384) float32 matrix
      - embeddings_ids.npy  -> shape (n_docs,) int64 array of document ids

    Row i of embeddings.npy corresponds to document id embeddings_ids[i].

    Resumable: if embeddings.npy / embeddings_ids.npy already contain a
    prefix of ids, this function skips documents already embedded and
    continues from where it left off. `max_seconds`, if given, stops
    the run early (after finishing the current chunk) so a caller can
    invoke this repeatedly across multiple short-lived processes.
    """
    import time

    from data.loader import DB_PATH, get_all_documents

    db_path = db_path or DB_PATH
    rows = get_all_documents(db_path)
    if not rows:
        raise RuntimeError("No documents found. Run `python data/loader.py` first.")

    all_ids = [r[0] for r in rows]
    all_texts = {r[0]: r[1] for r in rows}

    if os.path.exists(VECTORS_PATH) and os.path.exists(IDS_PATH):
        done_vectors = np.load(VECTORS_PATH)
        done_ids = np.load(IDS_PATH)
    else:
        done_vectors = np.zeros((0, EMBEDDING_DIM), dtype=np.float32)
        done_ids = np.zeros((0,), dtype=np.int64)

    done_id_set = set(done_ids.tolist())
    remaining_ids = [i for i in all_ids if i not in done_id_set]

    if not remaining_ids:
        print(f"All {len(all_ids)} documents already embedded. Nothing to do.")
        return done_vectors, done_ids

    print(
        f"Embedding {len(remaining_ids)} remaining documents "
        f"({len(done_id_set)} already done out of {len(all_ids)}) with '{MODEL_NAME}' ..."
    )

    start_time = time.time()
    vectors_acc = [done_vectors]
    ids_acc = [done_ids]

    for start in range(0, len(remaining_ids), chunk_size):
        chunk_ids = remaining_ids[start : start + chunk_size]
        chunk_texts = [all_texts[i] for i in chunk_ids]

        chunk_vectors = embed_documents(chunk_ids, chunk_texts)

        vectors_acc.append(chunk_vectors)
        ids_acc.append(np.array(chunk_ids, dtype=np.int64))

        combined_vectors = np.concatenate(vectors_acc, axis=0)
        combined_ids = np.concatenate(ids_acc, axis=0)
        np.save(VECTORS_PATH, combined_vectors)
        np.save(IDS_PATH, combined_ids)

        vectors_acc = [combined_vectors]
        ids_acc = [combined_ids]

        print(f"  progress: {len(combined_ids)}/{len(all_ids)} embedded and saved")

        if max_seconds is not None and (time.time() - start_time) > max_seconds:
            print(f"Stopping early after {time.time() - start_time:.1f}s (max_seconds={max_seconds}).")
            break

    final_vectors = vectors_acc[0]
    final_ids = ids_acc[0]
    print(f"Saved {final_vectors.shape} embedding matrix to {VECTORS_PATH}")
    print(f"Saved {len(final_ids)} ids to {IDS_PATH}")
    return final_vectors, final_ids


def load_embeddings():
    """Loads the persisted (vectors, ids) pair from disk."""
    if not (os.path.exists(VECTORS_PATH) and os.path.exists(IDS_PATH)):
        raise FileNotFoundError(
            "Embeddings not found. Run `python embedder.py` to build them first."
        )
    vectors = np.load(VECTORS_PATH)
    ids = np.load(IDS_PATH)
    return vectors, ids


if __name__ == "__main__":
    build_and_save_embeddings()
