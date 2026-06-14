# Evaluation

This folder evaluates the retriever (automated, offline) and provides a harness
for human answer-level evaluation. **No Groq key is needed for retrieval
evaluation.**

## 1. Retrieval evaluation — `evaluate.py`

```bash
python eval/evaluate.py            # writes eval/results.json
```

**Metric: Page Hit Rate@k** (`hit@k`). A query is a *hit at k* when at least one
of the top-k retrieved chunks comes from a page labelled as containing the
answer. We also report page-level MRR@5 and the mean number of unique pages in
the top-5.

This is **not** information-retrieval recall, and it is reported honestly as a
page-level hit rate. Limitations:

- The correct page does not guarantee the chunk contains the answer span.
- Page labels are coarse — a labelled page may hold other material too.
- A query receives a single binary success value; the metric does not measure
  how many relevant chunks were retrieved.

The harness imports the production retrieval core (`rag/`), so evaluation and the
live API exercise identical retrieval behaviour. Output (`results.json`) is
deterministic and records run metadata, the corpus hash, per-query retrieved
pages, misses, whether query expansion fired, and 95% Wilson confidence
intervals for `hit@5`.

## 2. Datasets — development vs held-out

`eval_set.json` is **`laws_dev_v1`, role `development`**. It was used for error
analysis and to develop the query-expansion map, so its numbers are **optimistic
and not unbiased test performance**. The dataset schema is in `schema.json`.

### Protocol for a genuinely held-out test set (human-required)

Use `templates/heldout_set.template.json` as a starting point and follow this
protocol so the result is defensible:

1. A group member **who did not tune the synonym map** writes the questions.
2. Gold pages are labelled by reading the official PDF.
3. The retriever and synonym map are **frozen** (commit hash recorded) *before*
   evaluation.
4. The held-out set is **never** used to edit retrieval rules.
5. A second annotator checks a sample (or all) of the labels.
6. Disagreements are recorded (`label_agreement`) and resolved.
7. Final results report development and held-out performance **separately**.

> Do not generate questions with an LLM and call the result "independent". An
> independent set requires the human steps above.

## 3. Answer-level evaluation — `evaluate_answers.py`

Retrieval hit rate does **not** establish answer correctness. `evaluate_answers.py`
adds an answer-level harness. It is primarily a **human** scoring framework;
deterministic citation checks run automatically.

```bash
# Deterministic citation checks only (no key, no generation):
python eval/evaluate_answers.py --citation-check-only

# Generate answers for manual scoring (needs GROQ_API_KEY):
python eval/evaluate_answers.py --generate
```

### Rubric (human, authoritative)

Score each item in `answer_eval_set.json`:

| Field | Scale | Meaning |
|---|---|---|
| `correctness` | 0 / 1 / 2 | 0 wrong, 1 partially right, 2 fully correct vs the Laws |
| `faithfulness` | 0 / 1 / 2 | 0 contradicts/invents beyond context, 1 minor drift, 2 fully supported by retrieved text |
| `citation_correctness` | 0 / 1 / 2 | 0 cites wrong pages, 1 mixed, 2 all cited pages support the claims |
| `citation_completeness` | 0 / 1 / 2 | 0 key claims uncited, 1 partial, 2 all substantive claims cited |
| `refusal_behavior` | pass / fail / n/a | for out-of-domain/ambiguous items: did it refuse appropriately? |
| `notes` | text | human explanation |

The set includes **in-domain, out-of-domain, ambiguous, and follow-up** items.
Capture the generated answer, retrieved sources, extracted citations, and model
name per item. Use temperature 0 for reproducibility.

### Deterministic citation checks (automated)

`evaluate_answers.py` extracts `[p. N]` citations and:

- confirms each cited page was actually supplied in the retrieved context,
- flags citations to pages that were **not** retrieved,
- counts substantive answers that contain **no** citation,
- distinguishes refusal responses (no citation expected).

These checks flag suspicious answers; they do **not** prove correctness.

### Optional LLM-as-judge (provisional only)

If a pinned LLM judge is added, its scores are **automated/provisional**, stored
separately from human scores, never mixed in, and require manual review before
being reported. We do not use LLM-as-judge as ground truth.

**Status:** the harness and annotation template are committed; human scores are
**pending human evaluation** and are not fabricated.
