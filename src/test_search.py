import faiss
import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer

print("Loading FAISS index...")

index = faiss.read_index(
    "indexes/trials.index"
)

print("Loading trials...")

df = pd.read_csv(
    "data/trials.csv"
)

print("Loading model...")

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

query = "lung cancer immunotherapy"

print(f"\nQuery: {query}")

query_embedding = model.encode(
    [query],
    normalize_embeddings=True
).astype("float32")

scores, indices = index.search(
    query_embedding,
    k=5
)

print("\nTop Results:\n")

for rank, idx in enumerate(indices[0]):

    row = df.iloc[idx]

    print(
        f"{rank+1}. {row['title']}"
    )

    print(
        f"Condition: {row['condition']}"
    )

    print(
        f"Score: {scores[0][rank]:.4f}"
    )

    print("-" * 60)