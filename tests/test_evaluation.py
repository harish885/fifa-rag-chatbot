import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("evaluate", ROOT / "eval" / "evaluate.py")
evaluate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluate)

ITEMS = [
    {"question": "offside", "gold_pages": [10]},
    {"question": "handball", "gold_pages": [12]},
]


def test_hit_semantics_all_hit():
    # retriever that always returns the gold page at rank 1
    def search(q):
        return [10 if q == "offside" else 12]
    res = evaluate.evaluate_system("perfect", search, ITEMS)
    assert res["metrics"]["hit@1"] == 1.0
    assert res["metrics"]["hit@5_count"] == "2/2"


def test_hit_semantics_miss_records():
    def search(q):
        return [999, 998, 997, 996, 995]
    res = evaluate.evaluate_system("miss", search, ITEMS)
    assert res["metrics"]["hit@5"] == 0.0
    assert len(res["misses"]) == 2


def test_mrr_rank_two():
    def search(q):
        return [999, 10 if q == "offside" else 12]
    res = evaluate.evaluate_system("rank2", search, ITEMS)
    assert abs(res["metrics"]["page_mrr@5"] - 0.5) < 1e-9


def test_wilson_interval_bounds():
    lo, hi = evaluate.wilson_interval(15, 20)
    assert 0.0 <= lo < hi <= 1.0


def test_per_query_present_and_deterministic():
    def search(q):
        return [10 if q == "offside" else 12]
    r1 = evaluate.evaluate_system("a", search, ITEMS)
    r2 = evaluate.evaluate_system("a", search, ITEMS)
    assert r1 == r2
    assert len(r1["per_query"]) == 2
    assert "expansion_phrases" in r1["per_query"][0]
