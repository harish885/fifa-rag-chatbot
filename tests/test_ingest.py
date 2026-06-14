import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("ingest", ROOT / "scripts" / "ingest.py")
ingest = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ingest)


def test_control_chars_removed_and_bullets_normalized():
    raw = "\x07First item\x07Second item\x0c"  # BEL bullets + form feed
    out = ingest.clean_text(raw)
    assert "\x07" not in out and "\x0c" not in out
    assert "•" in out


def test_double_bullets_collapsed():
    out = ingest.clean_text("\x07 \x07 Only one bullet")
    assert "• •" not in out


def test_strip_leading_page_number_and_footer():
    raw = "43\nReal legal content here.\nLaws of the Game 2025/26  |  Law 1  |  The Field"
    body, page, section = ingest.strip_running_elements(raw)
    assert page == 43
    assert "Real legal content" in body
    assert "Laws of the Game" not in body
    assert section == "Law 1 — The Field"


def test_sentence_split_no_midword():
    units = ingest.split_sentences("First sentence. Second sentence about offside.")
    assert units == ["First sentence.", "Second sentence about offside."]
    assert all(not u.startswith(" ") for u in units)


def test_chunk_overlap_starts_on_sentence_boundary():
    units = [f"Sentence number {i} with some content about the laws." for i in range(60)]
    chunks = ingest.chunk_units(units, min_chars=ingest.MIN_CHUNK_CHARS)
    assert len(chunks) >= 2
    for c in chunks:
        # No chunk starts mid-word: first char is a letter and the first token
        # matches the start of some source sentence.
        assert c[0].isalnum()
        assert c.startswith("Sentence")


def test_tiny_page_filtered_but_one_kept():
    assert ingest.chunk_units(["Hi."], min_chars=80) == ["Hi."]
    assert ingest.chunk_units([], min_chars=80) == []


def test_parse_page_ranges():
    assert ingest.parse_page_ranges("1-3,7,10-11") == {1, 2, 3, 7, 10, 11}


def test_missing_pdf_path_errors(capsys):
    rc = ingest.main(["/no/such/file.pdf"])
    assert rc == 2


def test_no_argument_errors():
    try:
        ingest.main([])
    except SystemExit as e:
        assert e.code != 0
