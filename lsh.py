"""
lsh.py
------
Locality-Sensitive Hashing (LSH) via random hyperplane projections.

The core intuition: imagine slicing the vector space with a random
hyperplane through the origin. Two vectors that point in similar
directions are very likely to land on the *same side* of that plane.
Two vectors pointing in very different directions are more likely to
land on *opposite* sides. If we slice the space with several random
hyperplanes at once and record which side of each plane a vector
falls on (as a bit: 1 or 0), we get a short binary "fingerprint" for
that vector. Similar vectors tend to produce the same fingerprint
(the same bucket), while dissimilar vectors tend to scatter into
different buckets.

That turns "find similar vectors" into "look up the bucket for this
fingerprint" — O(1) instead of O(n) — at the cost of occasionally
missing a true neighbor whose fingerprint happened to differ (this is
the "approximate" in Approximate Nearest Neighbor / recall < 100%).

We mitigate that recall loss with two techniques, both implemented
below:
  1. Multiple independent hash tables (n_tables) — a vector only
     needs to match in *any one* table to be considered a candidate,
     which recovers many of the "near misses" a single table would drop.
  2. Re-ranking candidates by true cosine similarity — LSH is only
     used to cheaply shrink millions of vectors down to a small
     candidate pool; the final ranking within that pool is exact.

No third-party LSH library is used — only numpy for the linear algebra.
"""

from collections import defaultdict

import numpy as np

from brute_force import cosine_similarity


class LSHIndex:
    def __init__(self, dim: int, n_bits: int = 8, n_tables: int = 10, seed: int = 42):
        """
        dim: dimensionality of the vectors being indexed (384 for our model)
        n_bits: number of random hyperplanes per table. Each hyperplane
            contributes one bit to a vector's hash code, so a table's
            bucket space has 2**n_bits possible buckets. More bits ->
            smaller, more precise buckets -> higher precision, lower
            recall (fewer false candidates, but easier to miss neighbors).
        n_tables: number of independent hash tables. A vector must only
            share a bucket with the query in ONE table to become a
            candidate, so more tables -> higher recall at the cost of
            more memory and more candidates to re-rank.
        """
        self.dim = dim
        self.n_bits = n_bits
        self.n_tables = n_tables

        rng = np.random.default_rng(seed)
        # self.planes[t] is a (n_bits, dim) matrix: each row is a random
        # hyperplane's normal vector for table t. Drawing from a standard
        # normal distribution gives hyperplanes with directions uniformly
        # spread over the sphere, which is what makes the "similar
        # vectors -> similar sign pattern" property hold on average.
        self.planes = rng.standard_normal((n_tables, n_bits, dim)).astype(np.float32)

        # One dict per table: hash_code (tuple of 0/1 bits) -> list of vector ids.
        self.tables = [defaultdict(list) for _ in range(n_tables)]

        # Kept so search() can re-rank candidates against the vectors
        # that were actually indexed, keyed by id.
        self._id_to_vector = {}

    def _hash(self, vector: np.ndarray, table_idx: int) -> tuple:
        """
        Projects `vector` onto every hyperplane in table `table_idx` and
        returns a binary hash code (a tuple of 0s and 1s) — the bucket key.

        The sign of the dot product between the vector and a hyperplane's
        normal tells us which side of that plane the vector falls on:
        positive -> bit 1, negative/zero -> bit 0. Stacking n_bits of
        these signs together gives a compact fingerprint. Two vectors
        with a small angle between them are unlikely to be split by any
        given random hyperplane, so they tend to agree on most/all bits.
        """
        projections = self.planes[table_idx] @ vector  # shape (n_bits,)
        bits = (projections > 0).astype(np.int8)
        return tuple(bits.tolist())

    def build(self, ids, vectors: np.ndarray):
        """
        Hashes every vector into every table, populating
        self.tables[table_idx][hash_code] with the list of vector ids
        that landed in that bucket.

        ids: sequence of document ids, aligned row-for-row with `vectors`
        vectors: (n_docs, dim) matrix
        """
        for row_idx, doc_id in enumerate(ids):
            vector = vectors[row_idx]
            self._id_to_vector[doc_id] = vector
            for table_idx in range(self.n_tables):
                code = self._hash(vector, table_idx)
                self.tables[table_idx][code].append(doc_id)

    def search(self, query_vector: np.ndarray, all_vectors: np.ndarray = None, top_k: int = 10, id_to_row: dict = None):
        """
        Approximate nearest-neighbor search:
          1. Hash the query into every table using the same random
             hyperplanes used at build time.
          2. Collect every vector id sharing that table's bucket — these
             are our *candidates*. A vector only needs to match in one
             of the n_tables to be considered, which is what recovers
             recall that a single table would lose.
          3. Deduplicate candidates (a vector can land in the query's
             bucket in more than one table).
          4. Re-rank the (small) candidate set by *exact* cosine
             similarity — LSH's job was only to shrink the search space,
             not to produce the final ranking.
          5. Return the top_k (id, score) pairs, best-first.

        If no candidates are found in any bucket (possible with very
        few hash bits/tables or a very small dataset), this degrades to
        an empty result rather than silently falling back to brute
        force — callers can decide how to handle that case.
        """
        candidate_ids = set()
        for table_idx in range(self.n_tables):
            code = self._hash(query_vector, table_idx)
            candidate_ids.update(self.tables[table_idx].get(code, []))

        if not candidate_ids:
            return []

        # Re-rank candidates with a single vectorized cosine-similarity
        # pass (same math as brute_force_search) instead of scoring one
        # candidate at a time in a Python loop — this is what makes the
        # re-ranking step cheap even when the candidate pool is sizeable.
        candidate_ids = list(candidate_ids)
        candidate_vectors = np.stack([self._id_to_vector[cid] for cid in candidate_ids])

        dot_products = candidate_vectors @ query_vector
        candidate_magnitudes = np.linalg.norm(candidate_vectors, axis=1)
        query_magnitude = np.linalg.norm(query_vector)
        denom = candidate_magnitudes * query_magnitude
        denom[denom == 0] = 1e-10
        similarities = dot_products / denom

        if top_k >= len(similarities):
            top_local_indices = np.argsort(-similarities)
        else:
            partial = np.argpartition(-similarities, top_k)[:top_k]
            top_local_indices = partial[np.argsort(-similarities[partial])]

        return [(candidate_ids[i], float(similarities[i])) for i in top_local_indices[:top_k]]


if __name__ == "__main__":
    from data.loader import get_all_documents
    from embedder import get_embedding, load_embeddings

    vectors, ids = load_embeddings()
    documents = dict(get_all_documents())

    index = LSHIndex(dim=vectors.shape[1], n_bits=8, n_tables=10)
    index.build(ids.tolist(), vectors)

    query = "football match results"
    print(f"Query: {query!r}\n")

    query_vector = get_embedding(query)
    results = index.search(query_vector, top_k=5)

    print(f"Top 5 LSH results (from {len(results)} candidates considered):")
    for rank, (doc_id, score) in enumerate(results, start=1):
        text = documents.get(doc_id, "<missing>")
        print(f"  {rank}. score={score:.4f}  id={doc_id}  text={text[:90]!r}")
