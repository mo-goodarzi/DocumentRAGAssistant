"""Run the full pipeline over the testset and score it.

    python eval/run_eval.py                     # full run, saves timestamped results
    python eval/run_eval.py --label baseline    # tag the run
    python eval/run_eval.py --limit 5           # smoke test on 5 questions first
    python eval/run_eval.py --no-ragas          # page-match recall only, no LLM judge

Every run writes eval/results/<timestamp>_<label>.json containing the scores AND
the config that produced them. Without the config a score is not reproducible and
the step 8 comparison means nothing.

Cost warning: RAGAS is an LLM judge — it makes several calls per question per
metric. A 35-question run is a few thousand calls' worth of tokens. Use --limit
first to confirm the plumbing works before paying for the full run.
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import config                        # noqa: E402
from generate import answer          # noqa: E402

EVAL_DIR = ROOT / "eval"
TESTSET = EVAL_DIR / "testset.json"
RESULTS_DIR = EVAL_DIR / "results"


# --------------------------------------------------------------------------
# Pipeline run
# --------------------------------------------------------------------------

def run_pipeline(testset: list[dict]) -> list[dict]:
    """Answer every question, keeping the retrieved chunks for scoring."""
    rows = []

    for i, item in enumerate(testset, start=1):
        print(f"  [{i}/{len(testset)}] {item['id']}: {item['question'][:60]}...")

        result = answer(item["question"])

        rows.append({
            **item,
            "response": result.text,
            "retrieved_contexts": [s.text for s in result.sources],
            "retrieved_meta": [
                {"source": s.source, "page": s.page, "distance": round(s.distance, 4)}
                for s in result.sources
            ],
            "cost_usd": result.cost_usd,
        })

    return rows


# --------------------------------------------------------------------------
# Page-match recall — computed by us, no LLM involved
# --------------------------------------------------------------------------

def page_match_recall(rows: list[dict], window: int = 1) -> dict:
    """Did a chunk from the expected page make it into the retrieved set?

    This is deliberately dumb and deterministic. It is the check on the LLM
    judge: when this and RAGAS context recall disagree sharply, one of them is
    wrong and it is worth finding out which.

    `window` allows +/- N pages, because a provision that starts on p.42 often
    continues onto p.43 and either chunk is a legitimate hit.
    """
    scored = [r for r in rows if r.get("answerable") and r.get("expected_page")]
    if not scored:
        return {"n": 0, "recall": None,
                "note": "no expected_page values filled in — cannot compute"}

    hits = []
    for r in scored:
        want_source = r["expected_source"]
        want_page = r["expected_page"]
        hit = any(
            m["source"] == want_source and abs(m["page"] - want_page) <= window
            for m in r["retrieved_meta"]
        )
        hits.append(hit)
        r["page_match_hit"] = hit

    return {
        "n": len(scored),
        "recall": round(sum(hits) / len(scored), 4),
        "window_pages": window,
    }


def refusal_rate(rows: list[dict]) -> dict:
    """On unanswerable questions, did the system actually decline?

    Crude keyword check — read the responses yourself too. It exists to catch
    the case where refusal behaviour silently regresses after a prompt change.
    """
    unanswerable = [r for r in rows if not r.get("answerable", True)]
    if not unanswerable:
        return {"n": 0, "refusal_rate": None}

    markers = ("do not", "does not", "not addressed", "no information",
               "not contain", "insufficient", "cannot", "not specify",
               "not mention", "not provide")

    refused = [any(m in r["response"].lower() for m in markers) for r in unanswerable]
    for r, ok in zip(unanswerable, refused):
        r["refused"] = ok

    return {"n": len(unanswerable),
            "refusal_rate": round(sum(refused) / len(unanswerable), 4)}


# --------------------------------------------------------------------------
# RAGAS
# --------------------------------------------------------------------------

def run_ragas(rows: list[dict]) -> dict:
    """Score with RAGAS. Requires ragas >= 0.2 (SingleTurnSample API).

    Unanswerable questions are excluded: context recall decomposes the
    reference into claims and checks them against retrieved chunks, and
    'this is not addressed in these documents' has no such claims. Including
    them produces a meaningless number that drags the average around.
    """
    from openai import OpenAI
    from ragas import EvaluationDataset, SingleTurnSample, evaluate
    from ragas.llms import llm_factory
    from ragas.metrics import (
        Faithfulness,
        LLMContextPrecisionWithReference,
        LLMContextRecall,
        ResponseRelevancy,
    )

    scored_rows = [r for r in rows if r.get("answerable", True)]
    print(f"\n  scoring {len(scored_rows)} answerable questions "
          f"({len(rows) - len(scored_rows)} unanswerable excluded)")

    samples = [
        SingleTurnSample(
            user_input=r["question"],
            retrieved_contexts=r["retrieved_contexts"],
            response=r["response"],
            reference=r["ground_truth"],
        )
        for r in scored_rows
    ]

    # The judge model is a variable too — a change here moves scores without
    # anything in the pipeline changing. It goes in the saved config.
    judge = llm_factory(config.JUDGE_MODEL, client=OpenAI())

    result = evaluate(
        dataset=EvaluationDataset(samples=samples),
        metrics=[
            LLMContextRecall(),
            LLMContextPrecisionWithReference(),
            Faithfulness(),
            ResponseRelevancy(),
        ],
        llm=judge,
    )

    df = result.to_pandas()

    # Attach per-question scores back onto the rows for failure analysis.
    metric_cols = [c for c in df.columns
                   if c not in {"user_input", "retrieved_contexts",
                                "response", "reference"}]
    for row, (_, scores) in zip(scored_rows, df.iterrows()):
        row["ragas"] = {c: (None if scores[c] != scores[c] else float(scores[c]))
                        for c in metric_cols}   # NaN check via self-inequality

    return {
        col: round(float(df[col].mean(skipna=True)), 4)
        for col in metric_cols
    }


# --------------------------------------------------------------------------

def category_breakdown(rows: list[dict]) -> dict:
    """Per-category context recall. This is where step 8 direction comes from —
    an aggregate hides the gap between plain-language and vocabulary questions."""
    out: dict[str, list[float]] = {}

    for r in rows:
        scores = r.get("ragas") or {}
        value = scores.get("context_recall")
        if value is None:
            continue
        out.setdefault(r["category"], []).append(value)

    return {
        cat: {"n": len(vals), "context_recall": round(sum(vals) / len(vals), 4)}
        for cat, vals in out.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", default="run", help="tag for the results file")
    parser.add_argument("--limit", type=int, help="only run the first N questions")
    parser.add_argument("--no-ragas", action="store_true",
                        help="skip the LLM judge (free, retrieval metrics only)")
    parser.add_argument("--page-window", type=int, default=1)
    args = parser.parse_args()

    if not TESTSET.exists():
        sys.exit(f"no testset at {TESTSET}")

    testset = json.loads(TESTSET.read_text())
    if args.limit:
        testset = testset[: args.limit]

    print(f"running pipeline over {len(testset)} questions")
    rows = run_pipeline(testset)

    summary = {
        "page_match": page_match_recall(rows, window=args.page_window),
        "refusal": refusal_rate(rows),
        "pipeline_cost_usd": round(sum(r["cost_usd"] for r in rows), 4),
    }

    if not args.no_ragas:
        print("\nscoring with RAGAS (this is the slow, paid part)")
        try:
            summary["ragas"] = run_ragas(rows)
            summary["by_category"] = category_breakdown(rows)
        except ImportError:
            print("  ragas not installed — pip install 'ragas>=0.2'")
        except Exception as exc:
            print(f"  RAGAS failed: {exc}")
            print("  retrieval metrics below are still valid.")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"{stamp}_{args.label}.json"

    out_path.write_text(json.dumps({
        "label": args.label,
        "timestamp": stamp,
        "n_questions": len(rows),
        "config": {**config.as_dict(), "judge_model": config.JUDGE_MODEL},
        "summary": summary,
        "rows": rows,
    }, indent=2))

    print("\n" + "=" * 60)
    print(json.dumps(summary, indent=2))
    print("=" * 60)
    print(f"\nsaved to {out_path}")


if __name__ == "__main__":
    main()
