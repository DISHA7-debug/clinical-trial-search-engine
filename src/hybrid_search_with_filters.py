import faiss
import pickle
import sqlite3
import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer

K = 60

print("Loading data...")

df = pd.read_csv(
    "data/trials.csv"
)

index = faiss.read_index(
    "indexes/trials.index"
)

with open(
    "indexes/bm25.pkl",
    "rb"
) as f:

    bm25 = pickle.load(f)

conn = sqlite3.connect(
    "data/metadata.db"
)

metadata_df = pd.read_sql_query(
    "SELECT * FROM trials_metadata",
    conn
)

metadata_map = {
    row["nct_id"]: {
        "phase": str(row["phase"]),
        "status": str(row["status"])
    }
    for _, row in metadata_df.iterrows()
}

model = SentenceTransformer(
    "/Users/disha/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
)

query = input(
    "\nSearch Query: "
)

phase_filter = input(
    "Phase Filter (leave empty if none): "
).strip()

status_filter = input(
    "Status Filter (leave empty if none): "
).strip()

query_embedding = model.encode(
    [query],
    normalize_embeddings=True
).astype("float32")

faiss_scores, faiss_indices = index.search(
    query_embedding,
    20
)

tokens = query.lower().split()

bm25_scores = bm25.get_scores(
    tokens
)

bm25_indices = np.argsort(
    bm25_scores
)[::-1][:20]

rrf_scores = {}

for rank, idx in enumerate(
    faiss_indices[0],
    start=1
):

    rrf_scores[idx] = (
        rrf_scores.get(idx, 0)
        + 1 / (K + rank)
    )

for rank, idx in enumerate(
    bm25_indices,
    start=1
):

    rrf_scores[idx] = (
        rrf_scores.get(idx, 0)
        + 1 / (K + rank)
    )

results = sorted(
    rrf_scores.items(),
    key=lambda x: x[1],
    reverse=True
)

filtered_results = []

for idx, score in results:

    row = df.iloc[idx]

    nct_id = row["nct_id"]

    if nct_id not in metadata_map:
        continue

    phase = metadata_map[nct_id]["phase"]
    status = metadata_map[nct_id]["status"]

    if phase_filter:

        phase_values = [
            p.strip().lower()
            for p in phase.split(",")
        ]

        if (
            phase_filter.strip().lower()
            not in phase_values
        ):
            continue

    if status_filter:

        if (
            status.strip().lower()
            != status_filter.strip().lower()
        ):
            continue

    filtered_results.append(
        (
            row,
            score,
            phase,
            status
        )
    )

print("\nTop Results\n")

if len(filtered_results) == 0:

    print(
        "No results found with the selected filters."
    )

for rank, (
    row,
    score,
    phase,
    status
) in enumerate(
    filtered_results[:10],
    start=1
):

    print(
        f"{rank}. {row['title']}"
    )

    print(
        f"NCT ID: {row['nct_id']}"
    )

    print(
        f"Condition: {row['condition']}"
    )

    print(
        f"Phase: {phase}"
    )

    print(
        f"Status: {status}"
    )

    print(
        f"Score: {score:.5f}"
    )

    print(
        "-" * 60
    )

conn.close()