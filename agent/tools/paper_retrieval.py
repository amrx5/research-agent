"""Hybrid retrieval over the local paper corpus: Chroma (dense) + BM25 (keyword),
fused with reciprocal rank fusion. Each hit carries a cosine relevance in [0, 1]
so the Validator can reject weak matches instead of passing them to the LLM.

The index is built offline by scripts/build_corpus.py.
"""
from __future__ import annotations

import pickle
import re

from pydantic import BaseModel, Field

from ..config import Config
from ..types import ErrorCode, ToolError
from .base import RawItem, Tool

COLLECTION = "papers"
TOKEN_RE = re.compile(r"[a-z0-9]+(?:[.\-][a-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def rrf(rankings: list[list[str]], k: int = 60) -> dict[str, float]:
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return scores


def cosine(a, b) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


class PaperSearchArgs(BaseModel):
    query: str = Field(..., min_length=2, max_length=300, description="What to find, using the paper's own terms (model, dataset, metric names).")
    top_k: int = Field(5, ge=1, le=10)
    year_min: int | None = Field(None, description="Optional: only papers from this year on.")


class PaperSearchTool(Tool):
    name = "paper_search"
    evidence_prefix = "P"
    description = (
        "Search the team's local corpus of research papers (full text, with paper title, section and page). "
        "USE FOR: what a specific paper in the corpus reports (results tables, methods, definitions, numbers). "
        "DO NOT USE FOR: recent results published after the corpus was built (use web_search) or arithmetic (use python). "
        "Returns the best-matching passages; returns nothing if no passage is relevant enough."
    )
    Args = PaperSearchArgs

    def __init__(self, config: Config):
        import chromadb
        from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

        self.cfg = config
        self.timeout_s = config.retrieval_timeout_s
        self.embed = SentenceTransformerEmbeddingFunction(model_name=config.embed_model)
        client = chromadb.PersistentClient(path=str(config.index_dir / "chroma"))
        self.collection = client.get_collection(COLLECTION, embedding_function=self.embed)
        with open(config.index_dir / "bm25.pkl", "rb") as f:
            bm = pickle.load(f)
        self.bm25, self.bm25_ids = bm["bm25"], bm["ids"]

    def run(self, args: PaperSearchArgs) -> list[RawItem]:
        n = self.cfg.retrieval_candidates
        where = {"year": {"$gte": args.year_min}} if args.year_min else None
        try:
            dense = self.collection.query(query_texts=[args.query], n_results=n, where=where)
        except Exception as exc:  # noqa: BLE001
            raise ToolError(ErrorCode.INTERNAL, f"vector store query failed: {exc}") from exc
        dense_ids = dense["ids"][0]

        scores = self.bm25.get_scores(tokenize(args.query))
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n]
        sparse_ids = [self.bm25_ids[i] for i in order if scores[i] > 0]

        fused = rrf([dense_ids, sparse_ids])
        top_ids = sorted(fused, key=fused.get, reverse=True)[: args.top_k * 2]
        if not top_ids:
            return []

        got = self.collection.get(ids=top_ids, include=["documents", "metadatas", "embeddings"])
        q_emb = self.embed([args.query])[0]
        by_id = {
            i: (doc, meta, emb)
            for i, doc, meta, emb in zip(got["ids"], got["documents"], got["metadatas"], got["embeddings"])
        }
        items: list[RawItem] = []
        for doc_id in top_ids:
            if doc_id not in by_id:
                continue
            doc, meta, emb = by_id[doc_id]
            if args.year_min and meta.get("year", 0) < args.year_min:
                continue
            items.append(
                {
                    "source": f"{meta.get('paper_id', '?')} p.{meta.get('page', '?')}",
                    "title": f"{meta.get('title', 'Untitled')} ({meta.get('year', 'n.d.')}) - {meta.get('section', '')}".strip(" -"),
                    "content": doc,
                    "relevance": round(cosine(q_emb, emb), 4),
                    "meta": {**meta, "rrf": round(fused[doc_id], 5)},
                }
            )
            if len(items) >= args.top_k:
                break
        return items
