"""PDF → cleaned text, one page at a time.

Usage:
    python src/extract.py corpus/ai_act_2024_1689.pdf 40    # dump one page
    python src/extract.py corpus/ai_act_2024_1689.pdf       # page count + sample
"""

import re
import sys
from collections.abc import Iterator
from pathlib import Path

import pdfplumber

MIN_PAGE_CHARS = 80


def clean(text: str) -> str:
    """Normalize the whitespace artifacts PDF extraction leaves behind.

    Starting set — extend it with whatever you actually find in your own
    documents. Order matters: rejoin hyphenated words before collapsing
    newlines, or the line break you're keying on is already gone.
    """
    text = text.replace("\xa0", " ")            # non-breaking spaces
    text = text.replace("\ufeff", "")           # stray byte-order marks
    text = re.sub(r"-\n(\w)", r"\1", text)      # compli-\nance → compliance
    text = re.sub(r"[ \t]+", " ", text)         # runs of spaces/tabs
    text = re.sub(r" *\n *", "\n", text)        # trim around line breaks
    text = re.sub(r"\n{3,}", "\n\n", text)      # cap blank-line runs
    return text.strip()


def extract_pages(pdf_path: Path) -> Iterator[tuple[int, str]]:
    """Yield (page_number, cleaned_text) for every page with real content.

    Page numbers are 1-indexed to match what a PDF reader shows — these end
    up in citations a user may click through to verify.
    """
    with pdfplumber.open(pdf_path) as pdf:
        for page_no, page in enumerate(pdf.pages, start=1):
            text = clean(page.extract_text() or "")
            if len(text) < MIN_PAGE_CHARS:
                continue                        # covers, blanks, number-only pages
            yield page_no, text


if __name__ == "__main__":
    path = Path(sys.argv[1])

    if len(sys.argv) > 2:
        wanted = int(sys.argv[2])
        for page_no, text in extract_pages(path):
            if page_no == wanted:
                print(text)
                break
        else:
            print(f"page {wanted} not found or filtered out as empty")
    else:
        pages = list(extract_pages(path))
        print(f"{path.name}: {len(pages)} pages with content")
        if pages:
            print(f"\n--- page {pages[0][0]} ---\n{pages[0][1][:800]}")