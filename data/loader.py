"""
data/loader.py
---------------
Loads the AG News dataset (news article descriptions) and persists
the first N rows into a local SQLite database so the rest of the
pipeline (embedder, brute-force search, LSH index, API) all read
from one durable source of truth instead of re-downloading data.

Run directly to (re)build the database:
    python data/loader.py
"""

import os
import sqlite3

N_ROWS = 10_000

# The SQLite file lives next to this script's parent (project root)
# so every other module can find it with the same relative path.
DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "vectors.db")


def _load_rows_from_huggingface(n_rows: int = N_ROWS):
    """
    Pull the first n_rows of AG News from the HuggingFace `datasets`
    library. AG News is a 4-class news topic classification dataset;
    we only care about the free-text "description" field, which gives
    us realistic short documents to index and search over.
    """
    from datasets import load_dataset

    # "ag_news" was renamed on the HuggingFace Hub to the namespaced
    # "fancyzhx/ag_news" (the canonical mirror of the original dataset).
    ds = load_dataset("fancyzhx/ag_news", split=f"train[:{n_rows}]")
    # AG News' text column is literally called "text" (title + description
    # concatenated by the dataset authors). We treat it as our "description".
    rows = [(i, row["text"]) for i, row in enumerate(ds)]
    return rows


def _load_rows_fallback(n_rows: int = N_ROWS):
    """
    Fallback used only if the `datasets` library / network download is
    unavailable in this environment. Generates deterministic, varied
    pseudo-news sentences so the rest of the pipeline (embeddings, LSH,
    benchmarking) can still be built, run, and demoed end-to-end.
    """
    import random

    random.seed(42)
    topics = {
        "World": [
            "peace talks", "border dispute", "trade summit", "election results",
            "refugee crisis", "diplomatic tensions", "coalition government", "ceasefire agreement",
        ],
        "Sports": [
            "football match results", "championship final", "olympic qualifiers", "tennis grand slam",
            "basketball playoffs", "marathon record", "transfer news", "league standings",
        ],
        "Business": [
            "stock market rally", "quarterly earnings", "interest rate hike", "merger deal",
            "startup funding round", "oil price surge", "inflation report", "central bank policy",
        ],
        "Sci/Tech": [
            "artificial intelligence breakthrough", "space mission launch", "smartphone release",
            "cybersecurity breach", "quantum computing advance", "renewable energy project",
            "software update", "robotics research",
        ],
    }
    templates = [
        "Reports indicate that {topic} continues to develop as analysts watch closely.",
        "Officials confirmed new details about the {topic} earlier today.",
        "The {topic} dominated headlines this week across major outlets.",
        "Experts weighed in on the {topic}, citing long-term implications.",
        "A new update on the {topic} was released by sources close to the matter.",
    ]

    rows = []
    idx = 0
    categories = list(topics.keys())
    while idx < n_rows:
        category = categories[idx % len(categories)]
        topic = random.choice(topics[category])
        template = random.choice(templates)
        text = f"{category}: {template.format(topic=topic)}"
        rows.append((idx, text))
        idx += 1
    return rows


def load_and_store(n_rows: int = N_ROWS, db_path: str = DB_PATH) -> int:
    """
    Loads text rows (HuggingFace AG News, falling back to a synthetic
    generator if unavailable) and writes them into a `documents` table
    in SQLite. Returns the number of rows stored.
    """
    try:
        rows = _load_rows_from_huggingface(n_rows)
        print(f"Loaded {len(rows)} rows from HuggingFace 'ag_news' dataset.")
    except Exception as exc:  # network / package unavailable
        print(f"Could not load 'ag_news' from HuggingFace ({exc}).")
        print("Falling back to a synthetic news-style dataset generator instead.")
        rows = _load_rows_fallback(n_rows)

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("DROP TABLE IF EXISTS documents")
    cur.execute(
        """
        CREATE TABLE documents (
            id INTEGER PRIMARY KEY,
            text TEXT NOT NULL
        )
        """
    )
    cur.executemany("INSERT INTO documents (id, text) VALUES (?, ?)", rows)
    conn.commit()
    conn.close()

    print(f"Stored {len(rows)} documents in {db_path}")
    return len(rows)


def get_all_documents(db_path: str = DB_PATH):
    """Returns a list of (id, text) tuples for every stored document."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT id, text FROM documents ORDER BY id")
    rows = cur.fetchall()
    conn.close()
    return rows


if __name__ == "__main__":
    load_and_store()
