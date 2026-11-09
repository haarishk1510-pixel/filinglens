"""Vector, keyword, and reranked retrieval.

The three modes exist so evaluation can compare each upgrade against a baseline.
"""
import re
import threading
from dataclasses import dataclass
from typing import Literal

import numpy as np
from fastembed import TextEmbedding
from fastembed.rerank.cross_encoder import TextCrossEncoder
from rank_bm25 import BM25Okapi

from . import config, store

Mode = Literal["keyword", "vector", "hybrid", "hybrid_rerank"]

CANDIDATES_PER_RETRIEVER = 50
RERANK_POOL = 30
RRF_K = 60

_lock = threading.Lock()
_embedder: TextEmbedding | None = None
_reranker: TextCrossEncoder | None = None


def embedder() -> TextEmbedding:
    global _embedder
    with _lock:
        if _embedder is None:
            _embedder = TextEmbedding(config.EMBEDDING_MODEL)
        return _embedder


def reranker() -> TextCrossEncoder:
    global _reranker
    with _lock:
        if _reranker is None:
            _reranker = TextCrossEncoder(config.RERANK_MODEL)
        return _reranker


def contextual_text(company: str, form: str, fiscal_year: int, section: str, text: str) -> str:
    """Prefix a chunk with where it came from, so a bare table row stays findable."""
    return f"{company} {form} fiscal {fiscal_year}. {section}.\n{text}"


def _contextual(chunk: dict) -> str:
    return contextual_text(
        chunk["company"], chunk["form"], chunk["fiscal_year"], chunk["section"], chunk["text"]
    )


def embed_passages(texts: list[str]) -> np.ndarray:
    return np.array(list(embedder().passage_embed(texts, batch_size=32)), dtype=np.float32)


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:[.\-'][a-z0-9]+)*", text.lower())


def reciprocal_rank_fusion(rankings: list[list[int]], k: int = RRF_K) -> list[int]:
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=lambda item: scores[item], reverse=True)


@dataclass
class _Index:
    chunks: list[dict]
    vectors: np.ndarray
    bm25: BM25Okapi


_indexes: dict[tuple[str, ...], _Index] = {}


def _index_for(filing_ids: list[str]) -> _Index | None:
    key = tuple(sorted(filing_ids))
    with _lock:
        cached = _indexes.get(key)
    if cached:
        return cached

    chunks = store.load_chunks(list(key))
    if not chunks:
        return None
    index = _Index(
        chunks=chunks,
        vectors=np.vstack([np.frombuffer(c["embedding"], dtype=np.float32) for c in chunks]),
        bm25=BM25Okapi([tokenize(_contextual(c)) for c in chunks]),
    )
    with _lock:
        _indexes[key] = index
    return index


def clear_cache() -> None:
    with _lock:
        _indexes.clear()


def search(query: str, filing_ids: list[str], top_k: int = 8, mode: Mode = "hybrid_rerank") -> list[dict]:
    index = _index_for(filing_ids)
    if index is None:
        return []

    vector_ranking: list[int] = []
    bm25_ranking: list[int] = []
    if mode != "keyword":
        query_vector = next(iter(embedder().query_embed(query)))
        vector_scores = index.vectors @ query_vector
        vector_ranking = np.argsort(-vector_scores)[:CANDIDATES_PER_RETRIEVER].tolist()
    if mode != "vector":
        bm25_scores = index.bm25.get_scores(tokenize(query))
        bm25_ranking = [
            int(i) for i in np.argsort(-bm25_scores)[:CANDIDATES_PER_RETRIEVER] if bm25_scores[i] > 0
        ]

    if mode == "vector":
        ordered = vector_ranking
    elif mode == "keyword":
        ordered = bm25_ranking
    else:
        ordered = reciprocal_rank_fusion([vector_ranking, bm25_ranking])

    if mode == "hybrid_rerank":
        pool = ordered[:RERANK_POOL]
        scores = list(reranker().rerank(query, [_contextual(index.chunks[i]) for i in pool]))
        ordered = [i for _, i in sorted(zip(scores, pool), key=lambda pair: pair[0], reverse=True)]

    return [
        {key: value for key, value in index.chunks[i].items() if key != "embedding"}
        for i in ordered[:top_k]
    ]
