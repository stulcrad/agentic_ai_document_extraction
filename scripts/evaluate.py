"""
Run every test document through the running service and score the extractions against the gold labels.
"""

import argparse
import json
import re
import unicodedata
from datetime import UTC, datetime
from pathlib import Path

import httpx

TEST_SET_DIR = Path("data/test_set")
RESULTS_DIR = Path("results")
OUTCOMES = ["correct", "correct_null", "missed", "wrong", "hallucinated"]
SUMMARY_KEYS = [
    "documents",
    "documents_failed",
    "fields_scored",
    "field_accuracy",
    "recall_when_stated",
    "hallucination_rate_when_absent",
    "confidently_wrong",
    "flagged_unverified",
    "flagged_and_actually_wrong",
    "flagged_but_correct",
]


def norm(value: object) -> str:
    """Casefold, replace punctuation with spaces and collapse whitespace."""
    str_value = str(value)

    # Step 1: NFKC-normalise
    str_value = unicodedata.normalize("NFKC", str_value)

    # Step 2: Casefold
    str_value = str_value.casefold()

    # Step 3: Replace punctuation with spaces
    str_value = re.sub(r"[^\w\s]", " ", str_value)

    # Step 4: Collapse whitespace
    str_value = " ".join(str_value.split())

    return str_value


def matches(field: str, pred: object, gold: object) -> bool:
    """Compare a non-null prediction with a non-null gold value, leniently per field type."""
    # Parties: check that every gold name is contained in some predicted name or vice versa.
    if field == "parties":
        pred_names = [norm(p["name"]) for p in pred if p.get("name")]
        gold_names = [norm(g) for g in gold]
        if len(pred_names) != len(gold_names):
            return False
        for g in gold_names:
            if not any(g in p or p in g for p in pred_names):
                return False
        return True

    # Governing law: check that every gold jurisdiction is contained in the prediction.
    if field == "governing_law":
        pred_norm = norm(pred)
        gold_norms = [norm(g) for g in gold.split(";")]
        return all(g in pred_norm for g in gold_norms)

    # Renewal term: check for containment in either direction.
    if field == "renewal_term":
        pred_norm = norm(pred)
        gold_norm = norm(gold)
        return pred_norm in gold_norm or gold_norm in pred_norm

    # Everything else: exact equality.
    return norm(pred) == norm(gold)


def classify(field: str, pred: object, gold: object) -> str:
    """Return one of OUTCOMES for a single field."""
    # An empty parties list means the same as null.
    if pred == []:
        pred = None

    # Prediction empty and gold empty -> correct_null
    if pred is None and gold is None:
        return "correct_null"

    # Prediction empty and gold present -> missed
    if pred is None:
        return "missed"

    # Prediction present and gold empty -> hallucinated
    if gold is None:
        return "hallucinated"

    # Both present -> correct or wrong, decided by matches()
    return "correct" if matches(field, pred, gold) else "wrong"


def rate(numerator: int, denominator: int) -> float | None:
    """Ratio rounded to three decimals, or None when the denominator is zero."""
    return round(numerator / denominator, 3) if denominator else None


def cell(value: object) -> str:
    """Compact text for a markdown table cell."""
    if value in (None, []):
        return "—"
    if isinstance(value, list):
        return "; ".join(v["name"] if isinstance(v, dict) else str(v) for v in value)
    return " ".join(str(value).replace("|", "/").split())


def to_markdown(summary: dict, documents: list[dict], rows: list[dict],
                fields: list[str]) -> str:
    """Render the summary, per-field, per-document and error tables."""
    lines = [
        "# Evaluation results",
        "",
        f"Model `{summary['model']}`, service `{summary['service_url']}`, run {summary['run_at']}.",
        "",
        "| metric | value |",
        "|---|---|",
    ]
    lines += [f"| {key} | {summary[key]} |" for key in SUMMARY_KEYS]

    lines += ["", "## Per field", "", "| field | " + " | ".join(OUTCOMES) + " | confidently wrong |"]
    lines.append("|---" * (len(OUTCOMES) + 2) + "|")
    for field in fields:
        field_rows = [r for r in rows if r["field"] == field]
        counts = " | ".join(str(sum(r["outcome"] == o for r in field_rows)) for o in OUTCOMES)
        lines.append(f"| {field} | {counts} | {sum(r['confidently_wrong'] for r in field_rows)} |")

    lines += ["", "## Per document", "", "| document | right / 7 | seconds | prompt tokens | truncated |"]
    lines.append("|---|---|---|---|---|")
    for d in documents:
        if "error" in d:
            lines.append(f"| {d['doc_id']} | failed: {cell(d['error'])} | | | |")
        else:
            lines.append(
                f"| {d['doc_id']} | {d['fields_right']} | {d['elapsed_seconds']} "
                f"| {d['prompt_tokens']} | {d['truncated']} |"
            )

    lines += ["", "## Fields that were not right", ""]
    lines += [
        "| document | field | gold | predicted | status | reason | outcome |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        if r["outcome"] not in ("correct", "correct_null"):
            lines.append(
                f"| {r['doc_id']} | {r['field']} | {cell(r['gold'])} | {cell(r['predicted_value'])} "
                f"| {r['status']} | {cell(r['reason'])} | {r['outcome']} |"
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:8000", help="base URL of the running service")
    parser.add_argument("--timeout", type=float, default=900.0, help="seconds allowed per document")
    args = parser.parse_args()

    labels_path = TEST_SET_DIR / "labels.json"
    with open(labels_path, encoding="utf-8") as f:
        labels = json.load(f)

    # 1. POST each document to /extract; one row per field, one entry per document.
    rows: list[dict] = []
    documents: list[dict] = []
    model_name = None
    with httpx.Client(base_url=args.url, timeout=args.timeout) as client:
        fields = list(client.get("/schema").json()["body"]["properties"])
        for doc_id, info in labels.items():
            doc_path = TEST_SET_DIR / f"{doc_id}.txt"
            if not doc_path.exists():
                print(f"Missing document file for {doc_id}, skipping.")
                documents.append({"doc_id": doc_id, "error": "missing .txt file"})
                continue
            doc_text = doc_path.read_text(encoding="utf-8")

            print(f"Processing document: {doc_id} ({len(doc_text)} chars)", flush=True)
            try:
                response = client.post("/extract", json={"document": doc_text})
                response.raise_for_status()
            except httpx.HTTPStatusError as e:
                print(f"  request failed: HTTP {e.response.status_code} {e.response.text[:200]}")
                documents.append({"doc_id": doc_id, "error": f"HTTP {e.response.status_code}"})
                continue
            except httpx.HTTPError as e:
                print(f"  request failed: {type(e).__name__}: {e}")
                documents.append({"doc_id": doc_id, "error": type(e).__name__})
                continue

            extraction = response.json()
            model_name = extraction["model_name"]
            doc_rows = []
            for field in fields:
                result = extraction["results"][field]
                gold_value = info["gold"][field]
                outcome = classify(field, result["value"], gold_value)
                doc_rows.append(
                    {
                        "doc_id": doc_id,
                        "field": field,
                        "gold": gold_value,
                        "predicted_value": result["value"],
                        "predicted_quote": result["quote"],
                        "status": result["status"],
                        "reason": result["reason"],
                        "outcome": outcome,
                        "confidently_wrong": (
                            outcome in ("wrong", "hallucinated") and result["status"] == "found_and_verified"
                        ),
                    }
                )
            rows.extend(doc_rows)
            documents.append(
                {
                    "doc_id": doc_id,
                    "elapsed_seconds": round(extraction["elapsed_ms"] / 1000, 1),
                    "prompt_tokens": extraction["prompt_tokens"],
                    "truncated": extraction["truncated"],
                    "fields_right": sum(r["outcome"] in ("correct", "correct_null") for r in doc_rows),
                }
            )
            done = documents[-1]
            print(f"  {done['fields_right']}/7 right in {done['elapsed_seconds']} s", flush=True)

    # 2. Summarise.
    counts = {outcome: sum(r["outcome"] == outcome for r in rows) for outcome in OUTCOMES}
    flagged = [r for r in rows if r["status"] == "unverified"]
    summary = {
        "model": model_name,
        "service_url": args.url,
        "run_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "documents": len(labels),
        "documents_failed": sum("error" in d for d in documents),
        "fields_scored": len(rows),
        "field_accuracy": rate(counts["correct"] + counts["correct_null"], len(rows)),
        "recall_when_stated": rate(counts["correct"], counts["correct"] + counts["missed"] + counts["wrong"]),
        "hallucination_rate_when_absent": rate(
            counts["hallucinated"], counts["correct_null"] + counts["hallucinated"]
        ),
        "confidently_wrong": sum(r["confidently_wrong"] for r in rows),
        "flagged_unverified": len(flagged),
        "flagged_and_actually_wrong": sum(r["outcome"] in ("wrong", "hallucinated") for r in flagged),
        "flagged_but_correct": sum(r["outcome"] == "correct" for r in flagged),
        "outcomes": counts,
    }

    # 3. Write results.json (everything) and results.md (tables for the README).
    RESULTS_DIR.mkdir(exist_ok=True)
    report = {"summary": summary, "documents": documents, "fields": rows}
    report_json = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    (RESULTS_DIR / "results.json").write_text(report_json, encoding="utf-8")
    markdown = to_markdown(summary, documents, rows, fields)
    (RESULTS_DIR / "results.md").write_text(markdown, encoding="utf-8")
    print("\n" + markdown)


if __name__ == "__main__":
    main()
