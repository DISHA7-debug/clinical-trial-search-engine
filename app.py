import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import faiss
import pickle
import sqlite3

import numpy as np
import pandas as pd
import streamlit as st

from sentence_transformers import SentenceTransformer


st.set_page_config(
    page_title="Clinical Trial Search Engine",
    layout="wide"
)

st.title(
    "🔬 Clinical Trial Search Engine"
)

K = 60


@st.cache_resource
def load_resources():

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

    conn.close()

    metadata_map = {
        row["nct_id"]: {
            "phase": str(row["phase"]),
            "status": str(row["status"])
        }
        for _, row in metadata_df.iterrows()
    }

    model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    return (
        df,
        index,
        bm25,
        metadata_map,
        model
    )


df, index, bm25, metadata_map, model = load_resources()

def get_faiss_results(query, top_k=20):

    query_embedding = model.encode(
        [query],
        normalize_embeddings=True
    ).astype("float32")

    _, indices = index.search(
        query_embedding,
        top_k
    )

    return indices[0]


def get_bm25_results(query, top_k=20):

    tokens = query.lower().split()

    scores = bm25.get_scores(tokens)

    indices = np.argsort(
        scores
    )[::-1][:top_k]

    return indices


def get_hybrid_results(query, top_k=20):

    faiss_indices = get_faiss_results(
        query,
        top_k
    )

    bm25_indices = get_bm25_results(
        query,
        top_k
    )

    rrf_scores = {}

    for rank, idx in enumerate(
        faiss_indices,
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

    ranked = sorted(
        rrf_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    return [
        x[0]
        for x in ranked[:top_k]
    ]


def render_results(
    indices,
    retrieval_name
):

    shown = 0

    for idx in indices:

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
                phase_filter.lower()
                not in phase_values
            ):
                continue

        if status_filter:

            if (
                status.lower()
                != status_filter.lower()
            ):
                continue

        trial_url = (
            f"https://clinicaltrials.gov/study/{nct_id}"
        )

        st.markdown(
            f"### 🧬 {row['title']}"
        )

        c1, c2, c3 = st.columns(3)

        with c1:
            st.write(
                f"**Condition:** {row['condition']}"
            )

        with c2:
            st.write(
                f"**Phase:** {phase}"
            )

        with c3:
            st.write(
                f"**Status:** {status}"
            )

        st.caption(
            f"{retrieval_name} Retrieval"
        )

        with st.expander(
            "View Trial Summary"
        ):
            st.write(
                row["summary"]
            )

        st.link_button(
            "🔗 Open Clinical Trial",
            trial_url
        )

        st.markdown("---")

        shown += 1

    if shown == 0:

        st.warning(
            "No matching trials found."
        )


st.markdown("---")

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric(
        "Trials Indexed",
        len(df)
    )

with c2:
    st.metric(
        "Embedding Model",
        "MiniLM-L6-v2"
    )

with c3:
    st.metric(
        "Retriever",
        "Hybrid"
    )

with c4:
    st.metric(
        "Vector Dimension",
        "384"
    )

st.markdown("---")


st.sidebar.title(
    "Search Filters"
)

query = st.sidebar.text_input(
    "Search Query"
)

phase_filter = st.sidebar.selectbox(
    "Phase",
    [
        "",
        "PHASE1",
        "PHASE2",
        "PHASE3",
        "PHASE4"
    ]
)

status_filter = st.sidebar.selectbox(
    "Status",
    [
        "",
        "RECRUITING",
        "NOT_YET_RECRUITING",
        "ACTIVE_NOT_RECRUITING",
        "COMPLETED"
    ]
)

search_clicked = st.sidebar.button(
    "Search"
)

st.sidebar.markdown("---")

st.sidebar.subheader(
    "Evaluation"
)

st.sidebar.metric(
    "NDCG@10",
    "0.93"
)

st.sidebar.metric(
    "MRR",
    "1.00"
)

st.sidebar.metric(
    "Trials Indexed",
    "500"
)


search_tab, analytics_tab = st.tabs(
    [
        "🔍 Search",
        "📊 Analytics"
    ]
)

with analytics_tab:

    st.header(
        "Retrieval Analytics"
    )

    metrics = pd.read_csv(
        "data/evaluation_metrics.csv"
    )

    st.subheader(
        "NDCG@10 Comparison"
    )

    st.bar_chart(
        metrics.set_index(
            "Retriever"
        )
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric(
            "Best Retriever",
            "FAISS"
        )

    with c2:
        st.metric(
            "Best NDCG@10",
            "0.9343"
        )

    with c3:
        st.metric(
            "Dataset Size",
            str(len(df))
        )

    st.markdown("---")

    st.write(
        """
        This system compares:

        - FAISS Dense Retrieval
        - BM25 Sparse Retrieval
        - Hybrid Retrieval (RRF)

        Evaluation Metric:
        NDCG@10
        """
    )

with search_tab:

    if not search_clicked:

        st.info(
            "Enter a query and click Search."
        )

    elif not query:

        st.warning(
            "Please enter a search query."
        )

    else:

        hybrid_results = get_hybrid_results(
            query
        )

        faiss_results = get_faiss_results(
            query
        )

        bm25_results = get_bm25_results(
            query
        )

        tab1, tab2, tab3 = st.tabs(
            [
                "🚀 Hybrid",
                "🧠 Semantic (FAISS)",
                "🔍 Keyword (BM25)"
            ]
        )

        with tab1:

            st.subheader(
                "Hybrid Retrieval"
            )

            st.caption(
                "FAISS + BM25 + Reciprocal Rank Fusion"
            )

            render_results(
                hybrid_results,
                "Hybrid"
            )

        with tab2:

            st.subheader(
                "Semantic Search"
            )

            st.caption(
                "Dense Retrieval using Embeddings"
            )

            render_results(
                faiss_results,
                "FAISS"
            )

        with tab3:

            st.subheader(
                "Keyword Search"
            )

            st.caption(
                "Sparse Retrieval using BM25"
            )

            render_results(
                bm25_results,
                "BM25"
            )