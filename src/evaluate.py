import pandas as pd
import numpy as np
from math import log2

print("Loading relevance judgments...")

df = pd.read_csv(
    "data/relevance_judgments.csv"
)

df["relevance"] = pd.to_numeric(
    df["relevance"],
    errors="coerce"
).fillna(0)

def dcg(relevances):

    score = 0.0

    for rank, rel in enumerate(
        relevances,
        start=1
    ):

        score += (
            (2 ** rel - 1)
            / log2(rank + 1)
        )

    return score


def ndcg_at_k(group, k=10):

    predicted = (
        group
        .sort_values("rank")
        ["relevance"]
        .tolist()
    )[:k]

    ideal = sorted(
        predicted,
        reverse=True
    )[:k]

    dcg_score = dcg(
        predicted
    )

    idcg_score = dcg(
        ideal
    )

    if idcg_score == 0:
        return 0.0

    return dcg_score / idcg_score


def precision_at_k(group, k=10):

    predicted = (
        group
        .sort_values("rank")
        ["relevance"]
        .tolist()
    )[:k]

    relevant = sum(
        1 for r in predicted
        if r > 0
    )

    return relevant / k


def recall_at_k(group, k=10):

    predicted = (
        group
        .sort_values("rank")
        ["relevance"]
        .tolist()
    )[:k]

    retrieved_relevant = sum(
        1 for r in predicted
        if r > 0
    )

    total_relevant = sum(
        1 for r in group["relevance"]
        if r > 0
    )

    if total_relevant == 0:
        return 0.0

    return (
        retrieved_relevant
        / total_relevant
    )


def mrr(group):

    predicted = (
        group
        .sort_values("rank")
        ["relevance"]
        .tolist()
    )

    for rank, rel in enumerate(
        predicted,
        start=1
    ):

        if rel > 0:
            return 1 / rank

    return 0.0


ndcg_scores = []
precision_scores = []
recall_scores = []
mrr_scores = []

print("\nPer Query Metrics\n")

for query, group in df.groupby(
    "query"
):

    ndcg_score = ndcg_at_k(
        group
    )

    precision_score = precision_at_k(
        group
    )

    recall_score = recall_at_k(
        group
    )

    mrr_score = mrr(
        group
    )

    ndcg_scores.append(
        ndcg_score
    )

    precision_scores.append(
        precision_score
    )

    recall_scores.append(
        recall_score
    )

    mrr_scores.append(
        mrr_score
    )

    print(f"{query}")

    print(
        f"  NDCG@10      : {ndcg_score:.4f}"
    )

    print(
        f"  Precision@10 : {precision_score:.4f}"
    )

    print(
        f"  Recall@10    : {recall_score:.4f}"
    )

    print(
        f"  MRR          : {mrr_score:.4f}"
    )

    print()

print("=" * 40)

print(
    f"\nMean NDCG@10      : {np.mean(ndcg_scores):.4f}"
)

print(
    f"Mean Precision@10 : {np.mean(precision_scores):.4f}"
)

print(
    f"Mean Recall@10    : {np.mean(recall_scores):.4f}"
)

print(
    f"Mean MRR          : {np.mean(mrr_scores):.4f}"
)