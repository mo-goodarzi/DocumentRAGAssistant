"""python src/ask.py "your question" [--show] [-k 5]"""

import argparse
from retrieve import retrieve


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    parser.add_argument("-k", type=int, default=None)
    parser.add_argument("--show", action="store_true", help="print full chunk text")
    args = parser.parse_args()

    for s in retrieve(args.question, k=args.k):
        print(f"[{s.n}] {s.distance:.3f}  {s.source} p.{s.page}")
        preview = s.text if args.show else s.text[:300].replace("\n", " ") + "..."
        print(f"     {preview}\n")


if __name__ == "__main__":
    main()