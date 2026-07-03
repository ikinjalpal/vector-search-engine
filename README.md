# Vector Search Engine

A semantic search engine built from first principles: text embeddings,
an exact brute-force nearest-neighbor baseline, and a custom
Locality-Sensitive Hashing (LSH) index for approximate nearest-neighbor
(ANN) search — served over a small Flask API.

The goal of this project isn't to wrap an existing vector database; it's
to implement the actual retrieval algorithms (cosine similarity, random
hyperplane LSH, recall/latency benchmarking) from scratch with just
`numpy`, so the tradeoffs of exact vs. approximate search are visible
and measured, not assumed.

## Problem

Semantic search over text means: given a query, find the documents
whose *meaning* is closest to it, not just documents that share
keywords. The standard approach is to embed both the query and every
document into a shared vector space (e.g. 384 dimensions) where
semantic similarity corresponds to a small angle between vectors
(cosine similarity), then find the nearest vectors to the query.

The naive way to find those nearest vectors — compare the query
against every single stored vector — is exact, but its cost grows
linearly with the size of the collection (O(n) per query). At the
scale real search engines operate on (millions to billions of
vectors), that becomes too slow for interactive latency. The problem
this project explores is: **how do you trade a small amount of
correctness for a large amount of speed, and how do you measure that
tradeoff honestly?**

## Solution

The project is split into independent, testable layers:

1. **Dataset & embeddings** (`data/loader.py`, `embedder.py`) — 10,000
   real news articles from the AG News dataset are embedded into
   384-dimensional vectors using `sentence-transformers`
   (`all-MiniLM-L6-v2`), then persisted to disk (`embeddings.npy` /
   `embeddings_ids.npy`) so every other component reads the same
   vectors without re-running the model.

2. **Brute-force baseline** (`brute_force.py`) — implements cosine
   similarity from scratch (`(A · B) / (‖A‖ ‖B‖)`) and a vectorized
   `brute_force_search` that scores every stored vector against a
   query in one numpy matrix operation. This is the correctness oracle
   every other search method is measured against, and the "before"
   side of the benchmark. No `scikit-learn` / `scipy` is used for the
   actual similarity math.

3. **Custom LSH index** (`lsh.py`) — implements Locality-Sensitive
   Hashing via **random hyperplane projections**, without using any
   off-the-shelf ANN/LSH library:
   - Each of `n_tables` independent hash tables holds `n_bits` random
     hyperplanes (drawn once at index-build time).
   - A vector is hashed per table by checking which side of each
     hyperplane it falls on (sign of the dot product), producing an
     `n_bits`-bit binary "fingerprint" that becomes its bucket key.
     Vectors pointing in similar directions tend to land in the same
     bucket; dissimilar vectors tend to scatter.
   - At query time, the query is hashed into every table, all vectors
     sharing a bucket with it in *any* table become **candidates**
     (multiple tables recover recall a single table would lose), and
     those candidates are re-ranked by *exact* cosine similarity in a
     single vectorized pass — LSH's job is only to shrink the search
     space cheaply, not to produce the final ranking.

4. **Benchmarking** (`benchmark.py`) — measures brute-force vs. LSH
   query latency and LSH's Recall@10 (how much of brute force's true
   top-10 neighbors LSH also finds) at 1,000 / 5,000 / 10,000 vectors,
   prints a results table, and saves a latency/recall chart
   (`latency_vs_recall.png`).

5. **Flask API** (`app.py`) — exposes the engine over HTTP:
   - `POST /index` — embed and add a new document to the live index
   - `POST /search` — embed a query and return nearest neighbors,
     via LSH (default) or exact brute force (`"method": "brute_force"`)
   - `GET /stats` — current document count, embedding dimension, and
     LSH configuration

## Benchmark

Measured on this machine, 50 random queries per dataset size, `n_bits=6`,
`n_tables=15`:

| Dataset Size | Brute-Force Latency | LSH Latency | Speedup | Recall@10 |
|-------------:|---------------------:|------------:|--------:|----------:|
| 1,000        | 0.49 ms              | 0.35 ms     | 1.4x    | 70%       |
| 5,000        | 2.02 ms              | 1.45 ms     | 1.4x    | 80%       |
| 10,000       | 3.45 ms              | 2.93 ms     | 1.2x    | 86%       |

![Latency vs. Recall](latency_vs_recall.png)

**How to read this**: LSH is consistently a bit faster than brute
force, and its recall — how much of the true top-10 it manages to find
— improves as the dataset grows (buckets become more populated and
representative). But the raw speedup here is modest (1.2–1.4x), which
is itself an important, honest finding — see Tradeoffs below.

## Tradeoffs

- **Recall vs. speed is tunable, not free.** `n_bits` controls how
  fine-grained the hash buckets are (more bits = smaller, more
  precise buckets = fewer candidates but easier to miss a true
  neighbor); `n_tables` controls how many independent chances a
  vector gets to be found (more tables = higher recall, more memory,
  more candidates to re-rank). I tuned these by sweeping a grid of
  `(n_bits, n_tables)` combinations and picking the pair that gave
  the best recall without the candidate pool ballooning back toward
  the full dataset size.

- **Brute force via numpy/BLAS is a genuinely fast baseline at this
  scale.** At 1k–10k documents and 384 dimensions, a single matrix
  multiply (`all_vectors @ query_vector`) is so well-optimized in
  numpy that it's hard for a Python-level LSH implementation to beat
  it by a wide margin — the real crossover point where LSH's O(1)
  bucket lookup dominates a linear scan shows up at much larger n
  (millions of vectors) or in settings where brute force *isn't*
  vectorized (e.g. row-by-row comparisons in application code, or
  comparisons against a non-numeric backing store). The moderate
  1.2–1.4x speedup measured here is the honest result of that,
  not a limitation of the LSH algorithm itself.

- **Multi-probe re-ranking is what makes small-recall buckets usable.**
  Re-ranking candidates by exact cosine similarity (rather than
  trusting the LSH bucket order) means small precision losses "wash
  out" as long as the true neighbor is *anywhere* in the candidate
  set — which is why increasing `n_tables` (more chances to catch it)
  helps recall more directly than increasing `n_bits` does.

- **In-memory only.** The current index rebuilds `n_tables` full hash
  tables in memory; there's no persistence for the LSH structure
  itself (only the underlying vectors are persisted), and no sharding
  — it wouldn't scale past what fits in one process's RAM.

## What I'd improve

- **Multi-probe LSH**: instead of only checking the query's exact
  bucket, also check buckets 1 Hamming-bit away, recovering more
  recall without adding full extra hash tables.
- **Product quantization / IVF** for memory efficiency at much larger
  scale (millions+ vectors), where holding every full-precision vector
  in RAM stops being feasible.
- **Persist the LSH structure** itself (not just the raw vectors) so
  the API doesn't have to rebuild all hash tables from scratch on
  every restart.
- **Batch queries**: the API currently embeds and searches one query
  at a time; batching multiple queries into a single matrix operation
  would improve throughput under load.
- **A proper ANN recall benchmark against ground truth for *held-out*
  queries** (rather than queries drawn from the indexed set itself),
  to better reflect real-world query patterns.

## How to run

Requires Python 3.11 and the packages in `requirements.txt`.

```bash
pip install -r requirements.txt

# 1. Load the dataset into SQLite
python data/loader.py

# 2. Build and save embeddings (resumable; safe to re-run if interrupted)
python embedder.py

# 3. Sanity-check the brute-force baseline
python brute_force.py

# 4. Sanity-check the LSH index
python lsh.py

# 5. Run the benchmark (prints a table, saves latency_vs_recall.png)
python benchmark.py

# 6. Start the API
python app.py
```

Example API usage once `app.py` is running:

```bash
curl http://localhost:5000/stats

curl -X POST http://localhost:5000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "football match results", "top_k": 5}'

curl -X POST http://localhost:5000/index \
  -H "Content-Type: application/json" \
  -d '{"text": "A groundbreaking new AI chip was unveiled by researchers today."}'
```

## Project structure

```
vector-search-engine/
├── app.py              # Flask API: /index, /search, /stats
├── benchmark.py         # Latency + Recall@10 benchmark, saves a chart
├── brute_force.py       # From-scratch cosine similarity + exact search
├── lsh.py               # Custom LSH index (random hyperplane projections)
├── embedder.py           # Builds and persists sentence-transformer embeddings
├── data/
│   └── loader.py         # Loads AG News into SQLite
├── requirements.txt
└── README.md
```
