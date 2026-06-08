# Clinical Trial Search Engine

A retrieval-based search engine for discovering relevant clinical trials using semantic search, keyword search, and hybrid retrieval techniques.

## Features

* Semantic Search using FAISS and Sentence Transformers
* Keyword Search using BM25
* Hybrid Retrieval using Reciprocal Rank Fusion (RRF)
* Clinical Trial Metadata Filtering
* Interactive Streamlit Dashboard
* Retrieval Evaluation using NDCG, Precision, Recall and MRR
* ClinicalTrials.gov Integration

## Architecture

ClinicalTrials API

↓

Data Processing & Search Text Creation

↓

Sentence Transformer Embeddings (MiniLM)

↓

FAISS Vector Index

*

BM25 Index

↓

Reciprocal Rank Fusion (RRF)

↓

Streamlit Search Application

## Evaluation Results

| Retriever | NDCG@10 |
| --------- | ------- |
| FAISS     | 0.9343  |
| BM25      | 0.5958  |
| Hybrid    | 0.8302  |

## Tech Stack

* Python
* FAISS
* Sentence Transformers
* BM25
* SQLite
* Pandas
* Streamlit

## Example Queries

* lung cancer immunotherapy
* EGFR mutation
* KRAS mutation
* checkpoint inhibitor
* pembrolizumab

## Screenshots

(Add UI screenshots here)

## Future Improvements

* LLM-based Trial Summarization
* Retrieval-Augmented Generation (RAG)
* Personalized Trial Recommendations
* Live ClinicalTrials API Updates
