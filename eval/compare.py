"""Compare two eval runs and write the step 8 before/after report.

    python eval/compare.py results/2026..._baseline.json results/2026..._articles.json

Writes eval/results/comparison.md — the table that goes in your README, plus the
per-question flips, which are what you actually cite when someone asks what
changed.
"""

import argparse
import json
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent / "results"
REPORT = RESULTS_DIR / "comparison.md"


def load(path: Path) -> dict:
    p = path if path.is_absolute() else Path(__file__).resolve().parent / path
    return json.loads(p.read_text())


def config_diff(a: dict, b: dict) -> list[tuple[str, str, str]]:
    keys = sorted(set(a["config"]) | set(b["config"]))
    return [
        (k, str(a["config"].get(k, "—")), str(b["config"].get(k, "—")))
        for k in keys
        if a["config"].get(k) != b["config"].get(k)
    ]


def fmt(value) -> str:
    return "—" if value is None else f"{value:.3f}"


def delta(before, after) -> str:
    if before is None or after is None:
        return "—"
    d = after - before
    arrow = "▲" if d > 0.001 else ("▼" if d < -0.001 else "=")
    return f"{arrow} {d:+.3f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline")
    parser.add_argument("improved")
    args = parser.parse_args()

    a, b = load(Path(args.baseline)), load(Path(args.improved))

    lines = [
        "# Retrieval experiment — before / after",
        "",
        f"**Baseline:** `{a['label']}` ({a['timestamp']}, {a['n_questions']} questions)",
        f"**Variant:** `{b['label']}` ({b['timestamp']}, {b['n_questions']} questions)",
        "",
    ]

    # --- what actually changed --------------------------------------------
    changes = config_diff(a, b)
    lines += ["## What changed", ""]
    if not changes:
        lines += ["No config difference recorded. If the code changed without a "
                  "config value changing, note it here manually — otherwise this "
                  "result is not reproducible.", ""]
    else:
        lines += ["| Setting | Before | After |", "|---|---|---|"]
        lines += [f"| {k} | {before} | {after} |" for k, before, after in changes]
        lines += [""]
        if len(changes) > 1:
            lines += ["> More than one setting changed. The result cannot be "
                      "attributed to any single one of them.", ""]

    # --- metrics -----------------------------------------------------------
    lines += ["## Results", "", "| Metric | Before | After | Δ |", "|---|---|---|---|"]

    ra = a["summary"].get("ragas", {})
    rb = b["summary"].get("ragas", {})
    for metric in sorted(set(ra) | set(rb)):
        lines.append(
            f"| {metric} | {fmt(ra.get(metric))} | {fmt(rb.get(metric))} | "
            f"{delta(ra.get(metric), rb.get(metric))} |"
        )

    pa = a["summary"]["page_match"].get("recall")
    pb = b["summary"]["page_match"].get("recall")
    lines.append(f"| page_match_recall | {fmt(pa)} | {fmt(pb)} | {delta(pa, pb)} |")

    fa = a["summary"]["refusal"].get("refusal_rate")
    fb = b["summary"]["refusal"].get("refusal_rate")
    lines.append(f"| refusal_rate | {fmt(fa)} | {fmt(fb)} | {delta(fa, fb)} |")

    lines += [""]

    # --- per category ------------------------------------------------------
    ca = a["summary"].get("by_category", {})
    cb = b["summary"].get("by_category", {})
    if ca or cb:
        lines += ["## Context recall by category", "",
                  "| Category | n | Before | After | Δ |", "|---|---|---|---|---|"]
        for cat in sorted(set(ca) | set(cb)):
            va = ca.get(cat, {}).get("context_recall")
            vb = cb.get(cat, {}).get("context_recall")
            n = cb.get(cat, ca.get(cat, {})).get("n", "—")
            lines.append(f"| {cat} | {n} | {fmt(va)} | {fmt(vb)} | {delta(va, vb)} |")
        lines += [""]

    # --- flips -------------------------------------------------------------
    rows_a = {r["id"]: r for r in a["rows"]}
    rows_b = {r["id"]: r for r in b["rows"]}

    fixed, broken = [], []
    for qid in rows_a.keys() & rows_b.keys():
        ha = rows_a[qid].get("page_match_hit")
        hb = rows_b[qid].get("page_match_hit")
        if ha is False and hb is True:
            fixed.append(rows_b[qid])
        elif ha is True and hb is False:
            broken.append(rows_b[qid])

    lines += ["## Questions that flipped", ""]
    lines += [f"Fixed: **{len(fixed)}**  ·  Regressed: **{len(broken)}**", ""]

    for title, group in (("Fixed", fixed), ("Regressed", broken)):
        if not group:
            continue
        lines += [f"### {title}", ""]
        for r in group:
            before_pages = ", ".join(
                f"p.{m['page']}" for m in rows_a[r["id"]]["retrieved_meta"]
            )
            after_pages = ", ".join(f"p.{m['page']}" for m in r["retrieved_meta"])
            lines += [
                f"**{r['id']}** ({r['category']}) — expected p.{r['expected_page']}",
                "",
                f"> {r['question']}",
                "",
                f"- before: {before_pages}",
                f"- after: {after_pages}",
                "",
            ]

    lines += [
        "## Interpretation",
        "",
        "_Write this yourself. Which failure mode did the change address, and does "
        "the per-category breakdown support that story? If a metric moved the wrong "
        "way, say so and explain why — a documented negative result reads better "
        "than a suspiciously clean win._",
        "",
    ]

    REPORT.write_text("\n".join(lines))
    print(f"written to {REPORT}")


if __name__ == "__main__":
    main()
