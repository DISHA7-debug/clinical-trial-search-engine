import faiss
import pickle
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

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

query = input(
    "\nEnter query: "
)

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

final_results = sorted(
    rrf_scores.items(),
    key=lambda x: x[1],
    reverse=True
)[:10]

print("\nTop Results\n")

for rank, (idx, score) in enumerate(
    final_results,
    start=1
):

    row = df.iloc[idx]

    print(
        f"{rank}. {row['title']}"
    )

    print(
        f"Condition: {row['condition']}"
    )

    print(
        f"RRF Score: {score:.5f}"
    )

    print("-" * 60)