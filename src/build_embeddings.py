import pandas as pd
import numpy as np

from sentence_transformers import SentenceTransformer

print("Loading trials.csv...")

df = pd.read_csv(
    "data/trials.csv"
)

print(
    f"Loaded {len(df)} trials"
)

print("\nLoading embedding model...")

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

print("\nGenerating embeddings...")

embeddings = model.encode(
    df["search_text"].tolist(),
    batch_size=32,
    show_progress_bar=True,
    normalize_embeddings=True
)

print("\nEmbedding Matrix Shape:")
print(embeddings.shape)

np.save(
    "data/embeddings.npy",
    embeddings
)

print("\nSaved:")
print("data/embeddings.npy")

print("\nFirst Vector Preview:")
print(embeddings[0][:10])