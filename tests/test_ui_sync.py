from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_ui_copies_in_sync():
    """api/ui.html (bundled in the function) and public/index.html (static) must
    not drift. If this fails, copy one over the other."""
    a = (ROOT / "api" / "ui.html").read_text(encoding="utf-8")
    b = (ROOT / "public" / "index.html").read_text(encoding="utf-8")
    assert a == b, "api/ui.html and public/index.html differ — re-sync them."


def test_ui_has_non_official_disclaimer():
    html = (ROOT / "api" / "ui.html").read_text(encoding="utf-8").lower()
    assert "not affiliated" in html or "not endorsed" in html
