import pandas as pd
import pickle

from rank_bm25 import BM25Okapi

print("Loading trials...")

df = pd.read_csv(
    "data/trials.csv"
)

tokenized_docs = [
    text.lower().split()
    for text in df["search_text"]
]

print("Building BM25 index...")

bm25 = BM25Okapi(
    tokenized_docs
)

with open(
    "indexes/bm25.pkl",
    "wb"
) as f:
    pickle.dump(
        bm25,
        f
    )

print("Saved:")
print("indexes/bm25.pkl")