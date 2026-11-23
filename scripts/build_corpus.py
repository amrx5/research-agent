"""Build the local paper index (run once, offline, before asking questions).

    python scripts/build_corpus.py --pdf-dir data/papers --meta data/papers.csv

papers.csv columns: file,paper_id,title,year   (file = PDF filename in --pdf-dir).
Papers without a CSV row get paper_id = filename stem and year = 0.

Steps: PDF -> page text (PyMuPDF) -> ~500-token chunks with 15% overlap, tagged with
paper_id, title, year, page, section -> Chroma (dense) + BM25 (keyword) indexes.
"""
from __future__ import annotations

import argparse
import csv
import pickle
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agent.config import Config  # noqa: E402
from agent.tools.paper_retrieval import COLLECTION, tokenize  # noqa: E402

WORDS_PER_CHUNK = 375  # ~500 tokens
OVERLAP = 0.15
HEADING_RE = re.compile(
    r"^\s*(?:\d+(?:\.\d+)*\.?\s+[A-Z][\w\- ,:]{2,60}|abstract|introduction|related work|background|method(?:s|ology)?"
    r"|experiments?|results|discussion|conclusions?|references|appendix)\s*$",
    re.I,
)


def load_meta(path: Path | None) -> dict[str, dict]:
    if not path:
        return {}
    with path.open(newline="", encoding="utf-8") as f:
        return {row["file"]: row for row in csv.DictReader(f)}


def pdf_pages(pdf: Path) -> list[str]:
    import pymupdf

    with pymupdf.open(pdf) as doc:
        return [page.get_text("text") for page in doc]


def chunk_paper(pages: list[str]) -> list[dict]:
    """Chunk page by page so every chunk cites one page; track the current section heading."""
    chunks, section = [], "Front matter"
    step = int(WORDS_PER_CHUNK * (1 - OVERLAP))
    for page_no, text in enumerate(pages, start=1):
        words, word_sections = [], []
        for line in text.splitlines():
            if HEADING_RE.match(line) and len(line.split()) <= 8:
                section = line.strip()
            for w in line.split():
                words.append(w)
                word_sections.append(section)
        if section.lower() == "references":
            continue  # reference lists add noise for retrieval
        for start in range(0, max(len(words), 1), step):
            piece = words[start : start + WORDS_PER_CHUNK]
            if len(piece) < 40 and start > 0:
                break
            if piece:
                chunks.append({"page": page_no, "section": word_sections[start], "text": " ".join(piece)})
    return chunks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf-dir", type=Path, required=True)
    ap.add_argument("--meta", type=Path, default=None)
    args = ap.parse_args()

    import chromadb
    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
    from rank_bm25 import BM25Okapi

    cfg = Config()
    cfg.index_dir.mkdir(parents=True, exist_ok=True)
    meta = load_meta(args.meta)
    embed = SentenceTransformerEmbeddingFunction(model_name=cfg.embed_model)
    client = chromadb.PersistentClient(path=str(cfg.index_dir / "chroma"))
    try:
        client.delete_collection(COLLECTION)
    except Exception:  # noqa: BLE001 - first build
        pass
    col = client.create_collection(COLLECTION, embedding_function=embed, metadata={"hnsw:space": "cosine"})

    all_ids, all_tokens = [], []
    pdfs = sorted(args.pdf_dir.glob("*.pdf"))
    if not pdfs:
        sys.exit(f"no PDFs found in {args.pdf_dir}")
    for pdf in pdfs:
        m = meta.get(pdf.name, {})
        paper_id = m.get("paper_id") or pdf.stem
        title = m.get("title") or pdf.stem
        year = int(m.get("year") or 0)
        chunks = chunk_paper(pdf_pages(pdf))
        ids = [f"{paper_id}:{c['page']}:{i}" for i, c in enumerate(chunks)]
        metas = [{"paper_id": paper_id, "title": title, "year": year, "page": c["page"], "section": c["section"]} for c in chunks]
        docs = [c["text"] for c in chunks]
        for s in range(0, len(docs), 128):  # batch adds
            col.add(ids=ids[s : s + 128], documents=docs[s : s + 128], metadatas=metas[s : s + 128])
        all_ids += ids
        all_tokens += [tokenize(f"{title} {d}") for d in docs]
        print(f"{pdf.name}: {len(chunks)} chunks")

    with open(cfg.index_dir / "bm25.pkl", "wb") as f:
        pickle.dump({"bm25": BM25Okapi(all_tokens), "ids": all_ids}, f)
    print(f"done: {len(pdfs)} papers, {len(all_ids)} chunks -> {cfg.index_dir}")


if __name__ == "__main__":
    main()
