"""Ask questions from the terminal.

    python src/ask.py "What practices are prohibited?"
    python src/ask.py "..." --show          # print the full retrieved chunks
    python src/ask.py "..." --retrieval     # retrieval only, no generation
    python src/ask.py                       # interactive loop
"""

import argparse
import sys
import time

from generate import answer
from retrieve import retrieve


def print_sources(sources, show: bool) -> None:
    for s in sources:
        print(f"  [{s.n}] {s.source}  p.{s.page}   distance {s.distance:.3f}")
        if show:
            print("      " + s.text.replace("\n", "\n      "))
            print()


def run(question: str, k: int | None, show: bool, retrieval_only: bool) -> None:
    started = time.perf_counter()

    if retrieval_only:
        sources = retrieve(question, k=k)
        print(f"\nretrieved {len(sources)} chunks "
              f"in {time.perf_counter() - started:.2f}s\n")
        print_sources(sources, show=True)
        return

    result = answer(question, k=k)
    elapsed = time.perf_counter() - started

    print(f"\n{result.text}\n")
    print(f"--- {len(result.sources)} sources | {elapsed:.2f}s | "
          f"${result.cost_usd:.5f} ---")
    print_sources(result.sources, show)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("question", nargs="*")
    parser.add_argument("-k", type=int, default=None, help="chunks to retrieve")
    parser.add_argument("--show", action="store_true", help="print full chunk text")
    parser.add_argument("--retrieval", action="store_true",
                        help="skip generation, show retrieval only")
    args = parser.parse_args()

    if args.question:
        run(" ".join(args.question), args.k, args.show, args.retrieval)
        return

    print("Ask a question. Ctrl-C to quit.")
    while True:
        try:
            q = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            sys.exit("\nbye")
        if q:
            run(q, args.k, args.show, args.retrieval)


if __name__ == "__main__":
    main()