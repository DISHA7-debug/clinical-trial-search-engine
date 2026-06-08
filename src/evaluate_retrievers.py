import faiss
import pickle
import pandas as pd
import numpy as np

from sentence_transformers import SentenceTransformer
from math import log2

print("Loading data...")

df = pd.read_csv(
    "data/trials.csv"
)

judgments = pd.read_csv(
    "data/relevance_judgments.csv"
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
    "/Users/disha/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
)

K = 60


def dcg(rels):

    score = 0

    for rank, rel in enumerate(
        rels,
        start=1
    ):

        score += (
            (2 ** rel - 1)
            / log2(rank + 1)
        )

    return score


def ndcg(
    predicted_ids,
    relevant_dict,
    k=10
):

    rels = []

    for doc_id in predicted_ids[:k]:

        rels.append(
            relevant_dict.get(
                doc_id,
                0
            )
        )

    dcg_score = dcg(
        rels
    )

    ideal = sorted(
        relevant_dict.values(),
        reverse=True
    )[:k]

    idcg = dcg(
        ideal
    )

    if idcg == 0:
        return 0

    return dcg_score / idcg


def faiss_search(query):

    embedding = model.encode(
        [query],
        normalize_embeddings=True
    ).astype("float32")

    scores, indices = index.search(
        embedding,
        10
    )

    ids = []

    for idx in indices[0]:

        ids.append(
            df.iloc[idx]["nct_id"]
        )

    return ids


def bm25_search(query):

    tokens = query.lower().split()

    scores = bm25.get_scores(
        tokens
    )

    indices = np.argsort(
        scores
    )[::-1][:10]

    ids = []

    for idx in indices:

        ids.append(
            df.iloc[idx]["nct_id"]
        )

    return ids


def hybrid_search(query):

    embedding = model.encode(
        [query],
        normalize_embeddings=True
    ).astype("float32")

    faiss_scores, faiss_indices = index.search(
        embedding,
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

    final = sorted(
        rrf_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )[:10]

    ids = []

    for idx, _ in final:

        ids.append(
            df.iloc[idx]["nct_id"]
        )

    return ids


queries = judgments["query"].unique()

faiss_scores = []
bm25_scores = []
hybrid_scores = []

print("\nEvaluating...\n")

for query in queries:

    group = judgments[
        judgments["query"] == query
    ]

    relevant_dict = dict(
        zip(
            group["nct_id"],
            group["relevance"]
        )
    )

    faiss_results = faiss_search(
        query
    )

    bm25_results = bm25_search(
        query
    )

    hybrid_results = hybrid_search(
        query
    )

    faiss_ndcg = ndcg(
        faiss_results,
        relevant_dict
    )

    bm25_ndcg = ndcg(
        bm25_results,
        relevant_dict
    )

    hybrid_ndcg = ndcg(
        hybrid_results,
        relevant_dict
    )

    faiss_scores.append(
        faiss_ndcg
    )

    bm25_scores.append(
        bm25_ndcg
    )

    hybrid_scores.append(
        hybrid_ndcg
    )

    print(f"{query}")

    print(
        f"  FAISS  : {faiss_ndcg:.4f}"
    )

    print(
        f"  BM25   : {bm25_ndcg:.4f}"
    )

    print(
        f"  Hybrid : {hybrid_ndcg:.4f}"
    )

    print()

print("=" * 50)

print(
    f"\nMean FAISS NDCG@10  : {np.mean(faiss_scores):.4f}"
)

print(
    f"Mean BM25 NDCG@10   : {np.mean(bm25_scores):.4f}"
)

print(
    f"Mean Hybrid NDCG@10 : {np.mean(hybrid_scores):.4f}"
)