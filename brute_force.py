"""
brute_force.py
---------------
The baseline exact-search algorithm. This is what every vector search
system is implicitly compared against: check the query against every
single stored vector, compute exact similarity, and return the best
matches. It is O(n) per query (n = number of stored vectors) and is
used here as (a) a correctness oracle for LSH's approximate results,
and (b) the "before" side of the benchmark in Stage 5.

No sklearn/scipy is used for the actual math — just numpy.
"""

import numpy as np


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """
    Cosine similarity between two vectors, implemented from scratch:

        cos(theta) = (A . B) / (||A|| * ||B||)

    It measures the angle between two vectors, not their magnitude —
    two vectors pointing in the same direction score 1.0 even if one
    is much "longer" than the other. That's exactly what we want for
    text embeddings, where the *direction* of the vector encodes
    meaning far more than its length does.
    """
    dot_product = np.dot(vec_a, vec_b)
    magnitude_a = np.sqrt(np.sum(vec_a ** 2))
    magnitude_b = np.sqrt(np.sum(vec_b ** 2))

    if magnitude_a == 0 or magnitude_b == 0:
        # A zero vector has no direction, so similarity is undefined;
        # we define it as 0 rather than dividing by zero.
        return 0.0

    return float(dot_product / (magnitude_a * magnitude_b))


def brute_force_search(query_vector: np.ndarray, all_vectors: np.ndarray, top_k: int = 10):
    """
    Compares `query_vector` against every row of `all_vectors`
    (shape (n_docs, dim)) and returns the top_k most similar rows as
    a list of (index, score) pairs, sorted best-first.

    This is intentionally vectorized (not a Python for-loop calling
    cosine_similarity per row) because computing all n similarities
    at once with matrix operations is dramatically faster in numpy —
    but the underlying math is identical to calling cosine_similarity
    n times.
    """
    # Dot product of the query against every stored vector at once:
    # shape (n_docs, dim) @ (dim,) -> (n_docs,)
    dot_products = all_vectors @ query_vector

    # Magnitude (L2 norm) of every stored vector, and of the query.
    doc_magnitudes = np.linalg.norm(all_vectors, axis=1)
    query_magnitude = np.linalg.norm(query_vector)

    # Avoid divide-by-zero for any all-zero rows.
    denom = doc_magnitudes * query_magnitude
    denom[denom == 0] = 1e-10

    similarities = dot_products / denom

    # argpartition gets the top_k largest values in O(n) instead of a
    # full O(n log n) sort, then we sort just those top_k for ordering.
    if top_k >= len(similarities):
        top_indices = np.argsort(-similarities)
    else:
        candidate_indices = np.argpartition(-similarities, top_k)[:top_k]
        top_indices = candidate_indices[np.argsort(-similarities[candidate_indices])]

    return [(int(idx), float(similarities[idx])) for idx in top_indices[:top_k]]


if __name__ == "__main__":
    from data.loader import get_all_documents
    from embedder import get_embedding, load_embeddings

    vectors, ids = load_embeddings()
    documents = dict(get_all_documents())

    query = "football match results"
    print(f"Query: {query!r}\n")

    query_vector = get_embedding(query)
    results = brute_force_search(query_vector, vectors, top_k=5)

    print("Top 5 brute-force results:")
    for rank, (row_idx, score) in enumerate(results, start=1):
        doc_id = int(ids[row_idx])
        text = documents.get(doc_id, "<missing>")
        print(f"  {rank}. score={score:.4f}  id={doc_id}  text={text[:90]!r}")
