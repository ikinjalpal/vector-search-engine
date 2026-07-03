"""
benchmark.py
------------
Generates the resume metrics for this project: at 1,000 / 5,000 /
10,000 vector dataset sizes, measures brute-force vs. LSH query
latency and LSH's Recall@10 (how much of brute force's top-10 does
LSH also find), then prints a results table and saves a chart.
"""

import time

import numpy as np

from brute_force import brute_force_search
from lsh import LSHIndex

DATASET_SIZES = [1000, 5000, 10000]
N_QUERIES = 50
TOP_K = 10
N_BITS = 6
N_TABLES = 15


def recall_at_k(lsh_result_ids, brute_force_result_ids, k=TOP_K):
    """
    Fraction of brute force's true top-k ids that also appear in LSH's
    top-k ids. 1.0 means LSH found every true neighbor; 0.0 means it
    found none.
    """
    bf_set = set(brute_force_result_ids[:k])
    lsh_set = set(lsh_result_ids[:k])
    if not bf_set:
        return 0.0
    return len(bf_set & lsh_set) / len(bf_set)


def benchmark_size(all_vectors: np.ndarray, all_ids: np.ndarray, size: int, rng: np.random.Generator):
    """Runs the latency + recall benchmark for a single dataset size."""
    subset_vectors = all_vectors[:size]
    subset_ids = all_ids[:size]

    index = LSHIndex(dim=subset_vectors.shape[1], n_bits=N_BITS, n_tables=N_TABLES)
    index.build(subset_ids.tolist(), subset_vectors)

    query_row_indices = rng.choice(size, size=min(N_QUERIES, size), replace=False)

    bf_latencies = []
    lsh_latencies = []
    recalls = []

    for row_idx in query_row_indices:
        query_vector = subset_vectors[row_idx]

        t0 = time.perf_counter()
        bf_results = brute_force_search(query_vector, subset_vectors, top_k=TOP_K)
        bf_latencies.append((time.perf_counter() - t0) * 1000)
        bf_result_ids = [int(subset_ids[i]) for i, _ in bf_results]

        t0 = time.perf_counter()
        lsh_results = index.search(query_vector, top_k=TOP_K)
        lsh_latencies.append((time.perf_counter() - t0) * 1000)
        lsh_result_ids = [doc_id for doc_id, _ in lsh_results]

        recalls.append(recall_at_k(lsh_result_ids, bf_result_ids, k=TOP_K))

    return {
        "size": size,
        "bf_latency_ms": float(np.mean(bf_latencies)),
        "lsh_latency_ms": float(np.mean(lsh_latencies)),
        "recall_at_10": float(np.mean(recalls)),
    }


def run_benchmark():
    from embedder import load_embeddings

    all_vectors, all_ids = load_embeddings()
    rng = np.random.default_rng(7)

    results = []
    for size in DATASET_SIZES:
        size = min(size, len(all_ids))
        result = benchmark_size(all_vectors, all_ids, size, rng)
        results.append(result)

    print_results_table(results)
    save_chart(results)
    return results


def print_results_table(results):
    print(f"{'Dataset Size':<13}| {'BF Latency':<11}| {'LSH Latency':<12}| {'Speedup':<8}| {'Recall@10'}")
    print("-" * 13 + "|" + "-" * 12 + "|" + "-" * 13 + "|" + "-" * 9 + "|" + "-" * 10)
    for r in results:
        speedup = r["bf_latency_ms"] / r["lsh_latency_ms"] if r["lsh_latency_ms"] > 0 else float("inf")
        print(
            f"{r['size']:<13,}| {r['bf_latency_ms']:.2f}ms{'':<4}"
            f"| {r['lsh_latency_ms']:.2f}ms{'':<5}"
            f"| {speedup:.1f}x{'':<4}"
            f"| {r['recall_at_10'] * 100:.0f}%"
        )


def save_chart(results, path: str = "latency_vs_recall.png"):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sizes = [r["size"] for r in results]
    bf_latencies = [r["bf_latency_ms"] for r in results]
    lsh_latencies = [r["lsh_latency_ms"] for r in results]
    recalls = [r["recall_at_10"] * 100 for r in results]

    fig, ax1 = plt.subplots(figsize=(8, 5))

    ax1.set_xlabel("Dataset size (# vectors)")
    ax1.set_ylabel("Query latency (ms)")
    ax1.plot(sizes, bf_latencies, marker="o", color="tab:red", label="Brute-force latency")
    ax1.plot(sizes, lsh_latencies, marker="o", color="tab:blue", label="LSH latency")
    ax1.tick_params(axis="y")

    ax2 = ax1.twinx()
    ax2.set_ylabel("Recall@10 (%)")
    ax2.plot(sizes, recalls, marker="s", color="tab:green", linestyle="--", label="LSH Recall@10")
    ax2.set_ylim(0, 105)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center left")

    plt.title("LSH vs. Brute-Force: Speed / Accuracy Tradeoff")
    fig.tight_layout()
    plt.savefig(path, dpi=150)
    print(f"\nSaved chart to {path}")


if __name__ == "__main__":
    run_benchmark()
