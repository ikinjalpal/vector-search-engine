"""
tests/test_core.py
------------------
Smoke tests for the core search algorithms.
Uses only numpy — no sentence-transformers model or pre-built
embeddings file required, so this runs cleanly in CI.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from brute_force import cosine_similarity, brute_force_search
from lsh import LSHIndex


RNG = np.random.default_rng(42)
DIM = 64
N = 200


def make_vectors(n=N, dim=DIM):
    v = RNG.standard_normal((n, dim)).astype(np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def test_cosine_similarity_identical():
    v = RNG.standard_normal(DIM).astype(np.float32)
    assert abs(cosine_similarity(v, v) - 1.0) < 1e-5, "identical vectors should score 1.0"


def test_cosine_similarity_orthogonal():
    a = np.zeros(DIM, dtype=np.float32); a[0] = 1.0
    b = np.zeros(DIM, dtype=np.float32); b[1] = 1.0
    assert abs(cosine_similarity(a, b)) < 1e-5, "orthogonal vectors should score ~0"


def test_cosine_similarity_zero_vector():
    v = RNG.standard_normal(DIM).astype(np.float32)
    z = np.zeros(DIM, dtype=np.float32)
    assert cosine_similarity(v, z) == 0.0, "zero vector should score 0"


def test_brute_force_returns_top_k():
    vectors = make_vectors()
    query = vectors[0]
    results = brute_force_search(query, vectors, top_k=10)
    assert len(results) == 10, f"expected 10 results, got {len(results)}"


def test_brute_force_self_is_top():
    vectors = make_vectors()
    query = vectors[7]
    results = brute_force_search(query, vectors, top_k=1)
    idx, score = results[0]
    assert idx == 7, f"expected self (idx=7) as top result, got idx={idx}"
    assert abs(score - 1.0) < 1e-4, f"self-similarity should be ~1.0, got {score}"


def test_brute_force_scores_descending():
    vectors = make_vectors()
    query = vectors[3]
    results = brute_force_search(query, vectors, top_k=20)
    scores = [s for _, s in results]
    assert scores == sorted(scores, reverse=True), "results must be in descending score order"


def test_lsh_build_and_search():
    vectors = make_vectors()
    ids = list(range(N))
    index = LSHIndex(dim=DIM, n_bits=4, n_tables=10, seed=0)
    index.build(ids, vectors)
    query = vectors[0]
    results = index.search(query, top_k=5)
    assert len(results) > 0, "LSH search returned no results"
    scores = [s for _, s in results]
    assert scores == sorted(scores, reverse=True), "LSH results must be in descending score order"


def test_lsh_self_in_results():
    vectors = make_vectors()
    ids = list(range(N))
    index = LSHIndex(dim=DIM, n_bits=4, n_tables=20, seed=1)
    index.build(ids, vectors)
    query = vectors[5]
    results = index.search(query, top_k=10)
    found_ids = [doc_id for doc_id, _ in results]
    assert 5 in found_ids, "LSH should find the query vector itself"


def test_lsh_result_scores_are_valid():
    vectors = make_vectors()
    ids = list(range(N))
    index = LSHIndex(dim=DIM, n_bits=5, n_tables=10, seed=2)
    index.build(ids, vectors)
    query = vectors[10]
    results = index.search(query, top_k=5)
    for doc_id, score in results:
        assert -1.0 <= score <= 1.0 + 1e-5, f"cosine score out of range: {score}"


if __name__ == "__main__":
    tests = [
        test_cosine_similarity_identical,
        test_cosine_similarity_orthogonal,
        test_cosine_similarity_zero_vector,
        test_brute_force_returns_top_k,
        test_brute_force_self_is_top,
        test_brute_force_scores_descending,
        test_lsh_build_and_search,
        test_lsh_self_in_results,
        test_lsh_result_scores_are_valid,
    ]
    passed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL  {t.__name__}: {e}")
    print(f"\n{passed}/{len(tests)} tests passed")
    sys.exit(0 if passed == len(tests) else 1)
