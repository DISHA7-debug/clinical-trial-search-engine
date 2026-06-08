import pandas as pd
import faiss
import pickle
import numpy as np

from sentence_transformers import SentenceTransformer

QUERIES = [
    "EGFR mutation",
    "KRAS mutation",
    "checkpoint inhibitor",
    "pembrolizumab",
    "lung cancer immunotherapy"
]

print("Loading data...")

df = pd.read_csv("data/trials.csv")

index = faiss.read_index(
    "indexes/trials.index"
)

model = SentenceTransformer(
    "/Users/disha/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
)

rows = []

for query in QUERIES:

    print(f"\nProcessing: {query}")

    query_embedding = model.encode(
        [query],
        normalize_embeddings=True
    ).astype("float32")

    scores, indices = index.search(
        query_embedding,
        10
    )

    for rank, idx in enumerate(
        indices[0],
        start=1
    ):

        row = df.iloc[idx]

        rows.append({
            "query": query,
            "rank": rank,
            "nct_id": row["nct_id"],
            "title": row["title"],
            "condition": row["condition"],
            "relevance": ""
        })

judgments = pd.DataFrame(rows)

judgments.to_csv(
    "data/relevance_judgments.csv",
    index=False
)

print(
    "\nSaved: data/relevance_judgments.csv"
)