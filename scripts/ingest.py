"""Ingest a PDF into retrieval-ready chunks for the RAG chatbot.

Usage: python scripts/ingest.py <path-to-pdf>
Output: data/chunks.json  [{"id", "page", "text"}, ...]
"""
import json
import re
import sys
from pathlib import Path

import fitz  # PyMuPDF

CHUNK_SIZE = 1100   # chars
OVERLAP = 200       # chars


def clean(text: str) -> str:
    text = text.replace("’", "'").replace("‘", "'")
    text = text.replace("“", '"').replace("”", '"')
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_page(text: str, page_no: int, start_id: int):
    """Split page text into overlapping chunks, breaking on sentence/line ends."""
    chunks = []
    pos = 0
    cid = start_id
    while pos < len(text):
        end = min(pos + CHUNK_SIZE, len(text))
        if end < len(text):
            # try to break at a sentence or newline boundary
            window = text[pos:end]
            br = max(window.rfind(". "), window.rfind(".\n"), window.rfind("\n"))
            if br > CHUNK_SIZE // 2:
                end = pos + br + 1
        piece = text[pos:end].strip()
        if len(piece) > 80:  # skip near-empty fragments
            chunks.append({"id": cid, "page": page_no, "text": piece})
            cid += 1
        if end >= len(text):
            break
        pos = max(end - OVERLAP, pos + 1)
    return chunks, cid


def main(pdf_path: str):
    doc = fitz.open(pdf_path)
    all_chunks = []
    next_id = 0
    for i, page in enumerate(doc):
        text = clean(page.get_text())
        if not text:
            continue
        page_chunks, next_id = chunk_page(text, i + 1, next_id)
        all_chunks.extend(page_chunks)

    out = Path(__file__).resolve().parent.parent / "data" / "chunks.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(all_chunks, ensure_ascii=False))
    print(f"Wrote {len(all_chunks)} chunks from {doc.page_count} pages -> {out}")


if __name__ == "__main__":
    main(sys.argv[1])
