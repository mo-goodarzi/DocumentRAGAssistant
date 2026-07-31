"""Streamlit demo UI.

    streamlit run src/app.py

Calls the pipeline directly rather than going through FastAPI — one process is
far simpler to deploy on HF Spaces, and the API remains available separately for
programmatic use. Set API_URL to proxy through it instead.
"""

import time

import streamlit as st
import tiktoken

import config
from generate import SYSTEM_PROMPT, answer_stream
from retrieve import get_collection

encoder = tiktoken.get_encoding("cl100k_base")

EXAMPLES = [
    "Which AI practices are prohibited?",
    "Can my company be fined for using face recognition in a shop?",
    "Do I have to tell people when they're talking to a chatbot?",
    "What must a provider do before placing a high-risk system on the market?",
]

st.set_page_config(
    page_title="EU Regulation Assistant",
    page_icon="§",
    layout="centered",
)

# Type is doing the work here: serif for the answer, because it is prose meant
# to be read as a document; mono for citations, because they are records meant
# to be checked. Everything else stays out of the way.
st.markdown("""
<style>
  .answer-body {
    font-family: Georgia, 'Iowan Old Style', serif;
    font-size: 1.05rem;
    line-height: 1.65;
  }
  .cite-meta {
    font-family: ui-monospace, 'SF Mono', Menlo, monospace;
    font-size: 0.78rem;
    letter-spacing: 0.01em;
    color: #5c6672;
  }
  .stat-row {
    font-family: ui-monospace, 'SF Mono', Menlo, monospace;
    font-size: 0.78rem;
    color: #5c6672;
    border-top: 1px solid #e6e8eb;
    padding-top: 0.6rem;
    margin-top: 1.2rem;
  }
  .grounding-note {
    font-size: 0.82rem;
    color: #6b7280;
    border-left: 2px solid #c8ccd2;
    padding-left: 0.7rem;
    margin: 0.5rem 0 1.2rem 0;
  }
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------------------------------
# State
# --------------------------------------------------------------------------

if "question" not in st.session_state:
    st.session_state.question = ""
if "history" not in st.session_state:
    st.session_state.history = []


def estimate_cost(question: str, sources, answer_text: str) -> float:
    """Streaming responses carry no usage data, so estimate from token counts.

    Approximate by construction — it counts the prompt we assembled rather than
    what the API billed, and ignores any per-request overhead.
    """
    prompt_text = SYSTEM_PROMPT + question + "".join(s.text for s in sources)
    prompt_tokens = len(encoder.encode(prompt_text))
    output_tokens = len(encoder.encode(answer_text))
    return (prompt_tokens * config.PRICE_IN
            + output_tokens * config.PRICE_OUT) / 1_000_000


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------

with st.sidebar:
    st.markdown("### Index")

    try:
        count = get_collection().count()
        meta = get_collection().get(include=["metadatas"])["metadatas"]
        docs = sorted({m["source"] for m in meta})
        st.markdown(
            f"<div class='cite-meta'>{count:,} chunks · {len(docs)} documents</div>",
            unsafe_allow_html=True,
        )
        for d in docs:
            n = sum(1 for m in meta if m["source"] == d)
            st.markdown(f"<div class='cite-meta'>· {d} ({n})</div>",
                        unsafe_allow_html=True)
    except Exception as exc:
        st.error(f"Index unavailable. Run `python src/ingest.py` first.\n\n`{exc}`")
        st.stop()

    st.markdown("### Retrieval")
    k = st.slider(
        "Chunks retrieved",
        min_value=1, max_value=10, value=config.TOP_K,
        help="How many passages are sent to the model. More coverage, more noise.",
    )

    st.markdown(
        f"<div class='cite-meta'>embed · {config.EMBED_MODEL}<br>"
        f"chat · {config.CHAT_MODEL}</div>",
        unsafe_allow_html=True,
    )

    if st.session_state.history:
        st.markdown("### This session")
        total = sum(h["cost"] for h in st.session_state.history)
        avg = sum(h["latency"] for h in st.session_state.history) / len(st.session_state.history)
        st.markdown(
            f"<div class='cite-meta'>{len(st.session_state.history)} queries · "
            f"${total:.4f} · {avg:.1f}s avg</div>",
            unsafe_allow_html=True,
        )


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

st.title("EU Regulation Assistant")
st.markdown(
    "<div class='grounding-note'>Answers are drawn only from the indexed "
    "documents, and every claim is numbered back to the passage it came from. "
    "Open a source to read the original text. Not legal advice.</div>",
    unsafe_allow_html=True,
)


def set_question(text: str) -> None:
    st.session_state.question = text


if not st.session_state.history:
    st.markdown("**Try one of these**")
    cols = st.columns(2)
    for i, example in enumerate(EXAMPLES):
        with cols[i % 2]:
            st.button(example, key=f"ex{i}", use_container_width=True,
                      on_click=set_question, args=(example,))

question = st.text_input(
    "Ask about the regulations",
    value=st.session_state.question,
    placeholder="What obligations apply to providers of high-risk AI systems?",
)

if question:
    started = time.perf_counter()

    try:
        stream = answer_stream(question, k=k)
        sources, _ = next(stream)          # first yield carries the sources
    except Exception as exc:
        st.error(f"Retrieval failed: {exc}")
        st.stop()

    if not sources:
        st.warning(
            "Nothing in the indexed documents matched that question. "
            "Try rephrasing, or check the sidebar for what is actually indexed."
        )
        st.stop()

    st.markdown("---")

    placeholder = st.empty()
    collected = ""

    try:
        for _, token in stream:
            if token:
                collected += token
                placeholder.markdown(
                    f"<div class='answer-body'>{collected}▌</div>",
                    unsafe_allow_html=True,
                )
        placeholder.markdown(
            f"<div class='answer-body'>{collected}</div>", unsafe_allow_html=True
        )
    except Exception as exc:
        st.error(f"Generation stopped partway: {exc}")

    latency = time.perf_counter() - started
    cost = estimate_cost(question, sources, collected)

    st.markdown(
        f"<div class='stat-row'>{latency:.1f}s · ~${cost:.5f} · "
        f"{len(sources)} passages retrieved</div>",
        unsafe_allow_html=True,
    )

    st.markdown("#### Sources")
    st.caption("Each number in the answer points to one of these. Open it to "
               "check the claim against the original wording.")

    for s in sources:
        with st.expander(f"[{s.n}]  {s.source} — page {s.page}"):
            st.markdown(
                f"<div class='cite-meta'>cosine distance {s.distance:.3f} · "
                f"chunk {s.chunk_index if hasattr(s, 'chunk_index') else '—'}</div>",
                unsafe_allow_html=True,
            )
            st.text(s.text)

    st.session_state.history.append({"cost": cost, "latency": latency})