"""Ingest the Laws of the Game PDF into retrieval-ready chunks.

Pipeline
--------
1. Extract text per page (PyMuPDF).
2. Clean: drop PDF control characters (the source uses U+0007 as a list bullet),
   normalize Unicode punctuation and whitespace, while preserving paragraph and
   list structure.
3. Strip non-semantic running elements: the leading page-number line and the
   "Laws of the Game 2025/26 | <section>" running footer. The footer is reused
   as best-effort section metadata; the page number is kept as metadata, not as
   searchable text.
4. Chunk sentence-aware: pack whole sentences / list items up to a character
   target, never starting a chunk mid-word or mid-sentence; overlap carries
   whole trailing sentences.
5. Validate and print an ingestion summary.

Usage
-----
    python scripts/ingest.py path/to/laws.pdf [--output data/chunks.json]
                             [--exclude-pages 1-3,7] [--min-chunk-chars 80]

Chunk record: {"id": int, "page": int, "section": str|null, "text": str}

Determinism: identical input PDF + flags -> identical chunks.json (ids are
sequential in document order).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import fitz  # PyMuPDF
except ImportError:  # pragma: no cover - exercised only without the dep
    fitz = None

CHUNK_SIZE = 1100      # target characters per chunk
OVERLAP_CHARS = 200    # overlap budget, filled with whole trailing sentences
MIN_CHUNK_CHARS = 80   # drop fragments shorter than this

_FOOTER_RE = re.compile(r"Laws of the Game 20\d\d/\d\d\s*\|\s*(.+)")
_PAGENUM_RE = re.compile(r"^\d{1,4}$")
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?:])\s+(?=[A-Z0-9“\"(])")


# --------------------------------------------------------------------- cleaning

def clean_text(text: str) -> str:
    """Normalize a raw page string while preserving line/paragraph structure."""
    # U+0007 (BEL) is used as a list-item bullet in this PDF -> real bullet.
    text = text.replace("\x07", "• ")
    text = unicodedata.normalize("NFKC", text)
    text = (
        text.replace("’", "'").replace("‘", "'")
        .replace("“", '"').replace("”", '"')
        .replace(" ", " ")
    )
    text = _CONTROL_RE.sub(" ", text)
    text = re.sub(r"(?:•\s*){2,}", "• ", text)  # collapse stacked bullet glyphs
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def strip_running_elements(raw: str) -> Tuple[str, Optional[int], Optional[str]]:
    """Remove the leading page-number line and the running footer.

    Returns (body_text, printed_page_number_or_None, footer_section_or_None).
    """
    lines = raw.splitlines()
    printed_page: Optional[int] = None
    section: Optional[str] = None

    # Leading standalone page-number line.
    idx = 0
    while idx < len(lines) and not lines[idx].strip():
        idx += 1
    if idx < len(lines) and _PAGENUM_RE.fullmatch(lines[idx].strip()):
        printed_page = int(lines[idx].strip())
        lines = lines[:idx] + lines[idx + 1:]

    kept: List[str] = []
    for line in lines:
        m = _FOOTER_RE.match(line.strip())
        if m:
            section = re.sub(r"\s*\|\s*", " — ", m.group(1)).strip()
            continue  # drop the running footer from content
        kept.append(line)
    return "\n".join(kept), printed_page, section


# --------------------------------------------------------------------- chunking

def split_sentences(text: str) -> List[str]:
    """Split into sentence-ish units, respecting existing line breaks.

    Each non-empty line is a unit; long lines are further split on sentence
    terminators. Units are never broken mid-word.
    """
    units: List[str] = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        if len(line) <= CHUNK_SIZE:
            units.extend(s.strip() for s in _SENT_SPLIT_RE.split(line) if s.strip())
        else:
            units.extend(s.strip() for s in _SENT_SPLIT_RE.split(line) if s.strip())
    return units


def chunk_units(units: List[str], min_chars: int) -> List[str]:
    """Pack sentence units into ~CHUNK_SIZE chunks with whole-sentence overlap."""
    chunks: List[str] = []
    cur: List[str] = []
    cur_len = 0
    for unit in units:
        add = len(unit) + (1 if cur else 0)
        if cur and cur_len + add > CHUNK_SIZE:
            chunks.append(" ".join(cur))
            # Build overlap from trailing whole sentences of the finished chunk.
            overlap: List[str] = []
            olen = 0
            for prev in reversed(cur):
                if olen + len(prev) > OVERLAP_CHARS:
                    break
                overlap.insert(0, prev)
                olen += len(prev) + 1
            cur = list(overlap)
            cur_len = sum(len(u) + 1 for u in cur)
        cur.append(unit)
        cur_len += add
    if cur:
        chunks.append(" ".join(cur))
    # Drop tiny fragments but always keep at least one chunk if there was text.
    filtered = [c for c in chunks if len(c) >= min_chars]
    return filtered or ([chunks[0]] if chunks else [])


# ------------------------------------------------------------------- pipeline

def parse_page_ranges(spec: str) -> set:
    """Parse '1-3,7,10-12' into a set of ints."""
    out: set = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    return out


def ingest(
    pdf_path: Path,
    min_chunk_chars: int = MIN_CHUNK_CHARS,
    exclude_pages: Optional[set] = None,
) -> Tuple[List[dict], Dict]:
    """Run the full pipeline. Returns (chunks, summary)."""
    if fitz is None:
        raise RuntimeError("PyMuPDF (pymupdf) is required for ingestion")
    exclude_pages = exclude_pages or set()
    doc = fitz.open(pdf_path)

    chunks: List[dict] = []
    next_id = 0
    last_section: Optional[str] = None
    skipped_excluded: List[int] = []
    empty_pages: List[int] = []
    indexed_pages: List[int] = []

    for i in range(doc.page_count):
        page_no = i + 1  # 1-based PDF page index (== printed number in this PDF)
        if page_no in exclude_pages:
            skipped_excluded.append(page_no)
            continue
        raw = clean_text(doc[i].get_text())
        body, _printed, section = strip_running_elements(raw)
        if section:
            last_section = section
        body = clean_text(body)
        if not body or len(body) < min_chunk_chars:
            empty_pages.append(page_no)
            continue
        page_chunks = chunk_units(split_sentences(body), min_chunk_chars)
        if not page_chunks:
            empty_pages.append(page_no)
            continue
        indexed_pages.append(page_no)
        for piece in page_chunks:
            chunks.append({
                "id": next_id,
                "page": page_no,
                "section": last_section,
                "text": piece,
            })
            next_id += 1

    lengths = [len(c["text"]) for c in chunks]
    summary = {
        "pdf_page_count": doc.page_count,
        "indexed_page_range": [min(indexed_pages), max(indexed_pages)] if indexed_pages else None,
        "indexed_pages": len(indexed_pages),
        "chunk_count": len(chunks),
        "chunk_chars_min": min(lengths) if lengths else 0,
        "chunk_chars_mean": round(sum(lengths) / len(lengths), 1) if lengths else 0,
        "chunk_chars_max": max(lengths) if lengths else 0,
        "skipped_excluded_pages": skipped_excluded,
        "empty_or_short_pages": empty_pages,
    }
    return chunks, summary


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Ingest the Laws of the Game PDF into chunks.json")
    parser.add_argument("pdf", nargs="?", help="path to the source PDF")
    parser.add_argument("--output", "-o", default=None, help="output JSON path (default data/chunks.json)")
    parser.add_argument("--exclude-pages", default=None, help="page ranges to skip, e.g. '1-3,7'")
    parser.add_argument("--min-chunk-chars", type=int, default=MIN_CHUNK_CHARS)
    args = parser.parse_args(argv)

    if not args.pdf:
        parser.error("missing required PDF path argument")
    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        print(f"error: PDF not found: {pdf_path}", file=sys.stderr)
        return 2
    if fitz is None:
        print("error: PyMuPDF (pymupdf) is not installed", file=sys.stderr)
        return 3
    try:
        fitz.open(pdf_path).page_count
    except Exception as e:  # noqa: BLE001
        print(f"error: could not open PDF ({e})", file=sys.stderr)
        return 4

    exclude = parse_page_ranges(args.exclude_pages) if args.exclude_pages else set()
    chunks, summary = ingest(pdf_path, args.min_chunk_chars, exclude)

    out = Path(args.output) if args.output else Path(__file__).resolve().parent.parent / "data" / "chunks.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(chunks, ensure_ascii=False, indent=0), encoding="utf-8")

    print(f"Ingestion summary for {pdf_path.name}:")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"Wrote {len(chunks)} chunks -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
