import faiss
import numpy as np
import pandas as pd
import pickle

print("Loading embeddings...")

embeddings = np.load(
    "data/embeddings.npy"
).astype("float32")

print(
    f"Embedding Shape: {embeddings.shape}"
)

dimension = embeddings.shape[1]

print(
    f"Vector Dimension: {dimension}"
)

index = faiss.IndexFlatIP(
    dimension
)

print("Adding vectors to FAISS...")

index.add(
    embeddings
)

print(
    f"Vectors Stored: {index.ntotal}"
)

faiss.write_index(
    index,
    "indexes/trials.index"
)

df = pd.read_csv(
    "data/trials.csv"
)

id_map = df["nct_id"].tolist()

with open(
    "indexes/id_map.pkl",
    "wb"
) as f:
    pickle.dump(
        id_map,
        f
    )

print("\nSaved:")
print("indexes/trials.index")
print("indexes/id_map.pkl")