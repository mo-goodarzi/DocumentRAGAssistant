"""Walk the 10 questions, show retrieved chunks, record your verdict.

    python eval/manual_review.py              # start (or resume) the review
    python eval/manual_review.py --report     # rebuild the markdown from saved answers

Judgements are saved to eval/manual_recall.json after every question, so you can
stop halfway and pick it up later. The markdown report is generated from that.

The verdict is YOURS. This script only removes the copy-paste tedium.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import config          # noqa: E402
from retrieve import retrieve   # noqa: E402

EVAL_DIR = ROOT / "eval"
QUESTIONS_FILE = EVAL_DIR / "questions.json"
ANSWERS_FILE = EVAL_DIR / "manual_recall.json"
REPORT_FILE = EVAL_DIR / "manual_recall.md"

DIAGNOSES = {
    "1": "vocabulary mismatch — my words are not the document's words",
    "2": "split answer — the provision spans chunks, I got part of it",
    "3": "crowding — near-duplicate passages pushed the right one past rank 5",
    "4": "extraction damage — the chunk text itself is mangled (table/annex)",
    "5": "single-document tunnel — needed two sources, retrieved only one",
    "6": "other (type your own)",
}

RULE = "=" * 78


def load_json(path: Path, default):
    if path.exists():
        return json.loads(path.read_text())
    return default


def prompt_verdict(q: dict) -> dict:
    """Ask for hit/miss and, on a miss, a diagnosis."""
    while True:
        raw = input("\n  Is a passage that ACTUALLY answers this in the top 5? "
                    "[y/n/s=skip/q=quit] ").strip().lower()
        if raw in {"y", "n", "s", "q"}:
            break
        print("  please answer y, n, s or q")

    if raw == "q":
        raise KeyboardInterrupt
    if raw == "s":
        return {}

    record = {"hit": raw == "y", "question": q["question"], "category": q["category"]}

    if raw == "n":
        print("\n  Why did it fail?")
        for key, label in DIAGNOSES.items():
            print(f"    {key}. {label}")
        choice = input("  > ").strip()
        if choice == "6" or choice not in DIAGNOSES:
            record["diagnosis"] = input("  diagnosis: ").strip()
        else:
            record["diagnosis"] = DIAGNOSES[choice]
    else:
        best = input("  which source/page carried the answer? (optional) ").strip()
        if best:
            record["answered_by"] = best

    extra = input("  any note? (enter to skip) ").strip()
    if extra:
        record["note"] = extra

    return record


def review() -> None:
    questions = load_json(QUESTIONS_FILE, [])
    if not questions:
        sys.exit(f"no questions found at {QUESTIONS_FILE}")

    answers = load_json(ANSWERS_FILE, {})

    todo = [q for q in questions if q["id"] not in answers]
    if not todo:
        print("all questions already judged — use --report to regenerate the markdown")
        return

    print(f"{len(todo)} question(s) left to judge. Ctrl-C or 'q' saves and exits.\n")

    try:
        for q in todo:
            print(f"\n{RULE}\n{q['id'].upper()}  [{q['category']}]\n{q['question']}\n{RULE}")

            sources = retrieve(q["question"], k=5)
            if not sources:
                print("  (nothing retrieved)")

            for s in sources:
                print(f"\n[{s.n}] distance {s.distance:.3f}   {s.source}  p.{s.page}")
                print("-" * 78)
                print(s.text)

            record = prompt_verdict(q)
            if record:
                record["retrieved"] = [
                    {"rank": s.n, "source": s.source, "page": s.page,
                     "distance": round(s.distance, 4)}
                    for s in sources
                ]
                answers[q["id"]] = record
                ANSWERS_FILE.write_text(json.dumps(answers, indent=2))
                print("  saved.")

    except KeyboardInterrupt:
        print("\n\nstopped — progress saved.")

    write_report(questions, answers)


def write_report(questions: list, answers: dict) -> None:
    judged = [q for q in questions if q["id"] in answers]
    if not judged:
        print("nothing judged yet, no report written")
        return

    hits = sum(1 for q in judged if answers[q["id"]]["hit"])
    recall = hits / len(judged)

    by_cat: dict[str, list[bool]] = {}
    for q in judged:
        by_cat.setdefault(q["category"], []).append(answers[q["id"]]["hit"])

    lines = [
        "# Manual recall@5 — baseline",
        "",
        f"Run: {datetime.now():%Y-%m-%d %H:%M}",
        "",
        "Judged by hand: for each question, is a passage that actually answers it "
        "present in the top 5 retrieved chunks?",
        "",
        "## Config",
        "",
        "```json",
        json.dumps(config.as_dict(), indent=2),
        "```",
        "",
        "## Result",
        "",
        f"**recall@5 = {hits}/{len(judged)} = {recall:.2f}**",
        "",
        "| Category | Hits | Total | Recall |",
        "|---|---|---|---|",
    ]

    for cat, results in by_cat.items():
        lines.append(
            f"| {cat} | {sum(results)} | {len(results)} | "
            f"{sum(results) / len(results):.2f} |"
        )

    lines += ["", "## Per question", ""]

    for q in judged:
        a = answers[q["id"]]
        mark = "HIT" if a["hit"] else "MISS"
        lines += [
            f"### {q['id'].upper()} — {mark}",
            "",
            f"> {q['question']}",
            "",
            f"*Category:* {q['category']}",
            "",
        ]
        retrieved = ", ".join(
            f"{r['source']} p.{r['page']} ({r['distance']})" for r in a["retrieved"]
        )
        lines += [f"*Retrieved:* {retrieved}", ""]
        if a["hit"] and a.get("answered_by"):
            lines += [f"*Answered by:* {a['answered_by']}", ""]
        if not a["hit"]:
            lines += [f"*Diagnosis:* {a.get('diagnosis', '(none recorded)')}", ""]
        if a.get("note"):
            lines += [f"*Note:* {a['note']}", ""]

    lines += [
        "## What this means for step 8",
        "",
        "_Fill in after judging: which failure mode dominates, and which single "
        "change is most likely to fix it?_",
        "",
    ]

    REPORT_FILE.write_text("\n".join(lines))
    print(f"\nrecall@5 = {hits}/{len(judged)} = {recall:.2f}")
    print(f"report written to {REPORT_FILE}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", action="store_true",
                        help="regenerate markdown from saved judgements")
    args = parser.parse_args()

    if args.report:
        write_report(load_json(QUESTIONS_FILE, []), load_json(ANSWERS_FILE, {}))
    else:
        review()
