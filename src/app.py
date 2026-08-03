"""Streamlit UI — ask the EU regulation corpus, or upload your own documents.

    streamlit run src/app.py

Uploaded documents go into a per-session Chroma collection and are deleted on
clear or after SESSION_TTL_HOURS. The demo corpus is read-only and shared.
"""

import logging
import time

import streamlit as st
import tiktoken

import config
from generate import SYSTEM_PROMPT, answer_stream
from session import (
    clear_session,
    delete_document,
    get_demo_collection,
    get_session_collection,
    new_session_id,
    session_documents,
    sweep_expired,
)
from upload import UploadRejected, ingest_upload

logging.basicConfig(level=logging.INFO)
encoder = tiktoken.get_encoding("cl100k_base")

DEMO_EXAMPLES = [
    "How long do I have to report a data breach under GDPR?",
    "What are the maximum fines for GDPR violations?",
    "At what age can a child consent to an online service without parental approval?",
    "When is a Data Protection Impact Assessment required?",
    "How must very large online platforms handle illegal content under the DSA?",
    "Are dark patterns banned on online platforms?",
]

st.set_page_config(page_title="Document RAG Assistant", page_icon="§", layout="centered")

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
# Startup — once per process, not per rerun
# --------------------------------------------------------------------------

@st.cache_resource
def startup() -> int:
    """Streamlit re-executes this whole script on every interaction, so an
    unguarded sweep would scan all collections on every keystroke."""
    return sweep_expired()


startup()


# --------------------------------------------------------------------------
# State
# --------------------------------------------------------------------------

for key, default in [
    ("session_id", None),
    ("question", ""),
    ("history", []),
    ("upload_cost", 0.0),
    ("ingested", {}),
]:
    if key not in st.session_state:
        st.session_state[key] = default

if st.session_state.session_id is None:
    st.session_state.session_id = new_session_id()

SESSION_ID = st.session_state.session_id


def estimate_cost(question: str, sources, answer_text: str) -> float:
    """Streaming carries no usage data, so estimate from token counts."""
    prompt = SYSTEM_PROMPT + question + "".join(s.text for s in sources)
    return (len(encoder.encode(prompt)) * config.PRICE_IN
            + len(encoder.encode(answer_text)) * config.PRICE_OUT) / 1_000_000


def set_question(text: str) -> None:
    st.session_state.question = text


# --------------------------------------------------------------------------
# Mode
# --------------------------------------------------------------------------

st.title("Document RAG Assistant")

mode = st.radio(
    "Source",
    ["EU regulation corpus", "My documents"],
    horizontal=True,
    label_visibility="collapsed",
)
is_demo = mode == "EU regulation corpus"


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------

with st.sidebar:
    if is_demo:
        st.markdown("### Indexed corpus")
        try:
            metas = get_demo_collection().get(include=["metadatas"])["metadatas"]
            docs = sorted({m["source"] for m in metas})
            st.markdown(f"<div class='cite-meta'>{len(metas):,} chunks · "
                        f"{len(docs)} documents</div>", unsafe_allow_html=True)
            for d in docs:
                n = sum(1 for m in metas if m["source"] == d)
                st.markdown(f"<div class='cite-meta'>· {d} ({n})</div>",
                            unsafe_allow_html=True)
        except Exception as exc:
            st.error(f"Demo index unavailable: {exc}")

    else:
        st.markdown("### Your documents")
        indexed = session_documents(SESSION_ID)

        if not indexed:
            st.markdown("<div class='cite-meta'>Nothing uploaded yet.</div>",
                        unsafe_allow_html=True)
        else:
            for name, n in indexed.items():
                col_a, col_b = st.columns([5, 1])
                col_a.markdown(f"<div class='cite-meta'>{name}<br>{n} chunks</div>",
                               unsafe_allow_html=True)
                if col_b.button("✕", key=f"del_{name}", help=f"Remove {name}"):
                    delete_document(SESSION_ID, name)
                    st.session_state.ingested.pop(name, None)
                    st.rerun()

            st.markdown(
                f"<div class='cite-meta'>indexing cost this session · "
                f"${st.session_state.upload_cost:.4f}</div>",
                unsafe_allow_html=True,
            )

            if st.button("Clear all", use_container_width=True):
                clear_session(SESSION_ID)
                st.session_state.session_id = new_session_id()
                st.session_state.ingested = {}
                st.session_state.upload_cost = 0.0
                st.session_state.history = []
                st.rerun()

        st.markdown(
            f"<div class='cite-meta'>Limits · {config.MAX_FILES_PER_SESSION} files · "
            f"{config.MAX_UPLOAD_MB:.0f}MB · {config.MAX_PAGES_PER_FILE} pages each<br>"
            f"Deleted after {config.SESSION_TTL_HOURS:.0f}h</div>",
            unsafe_allow_html=True,
        )

    st.markdown("### Retrieval")
    k = st.slider("Chunks retrieved", 1, 10, config.TOP_K,
                  help="How many passages are sent to the model. "
                       "More coverage, more noise.")

    st.markdown(f"<div class='cite-meta'>embed · {config.EMBED_MODEL}<br>"
                f"chat · {config.CHAT_MODEL}</div>", unsafe_allow_html=True)

    if st.session_state.history:
        total = sum(h["cost"] for h in st.session_state.history)
        st.markdown(f"<div class='cite-meta'>{len(st.session_state.history)} queries · "
                    f"${total:.4f}</div>", unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Source selection
# --------------------------------------------------------------------------

collection = None

if is_demo:
    st.markdown(
        "<div class='grounding-note'>Answers come only from the indexed EU "
        "regulations, and every claim is numbered back to the passage it came "
        "from. Open a source to read the original text. Not legal advice.</div>",
        unsafe_allow_html=True,
    )
    try:
        collection = get_demo_collection()
    except Exception as exc:
        st.error(f"Demo index unavailable: {exc}")
        st.stop()

else:
    st.markdown(
        "<div class='grounding-note'>Upload PDFs and ask questions about them. "
        "Documents stay in your session, are not shared, and are deleted "
        f"automatically after {config.SESSION_TTL_HOURS:.0f} hours.</div>",
        unsafe_allow_html=True,
    )

    uploaded = st.file_uploader(
        "Upload PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        help=f"Up to {config.MAX_FILES_PER_SESSION} files, "
             f"{config.MAX_UPLOAD_MB:.0f}MB and "
             f"{config.MAX_PAGES_PER_FILE} pages each. Text-based PDFs only.",
    )

    if uploaded:
        already = session_documents(SESSION_ID)
        pending = [f for f in uploaded if f.name not in already]

        for file in pending:
            bar = st.progress(0.0, text=f"Preparing {file.name}")

            def report(fraction: float, message: str, _bar=bar) -> None:
                _bar.progress(min(fraction, 1.0), text=message)

            try:
                result = ingest_upload(
                    session_id=SESSION_ID,
                    file_bytes=file.getvalue(),
                    filename=file.name,
                    existing_files=len(session_documents(SESSION_ID)),
                    on_progress=report,
                )
                bar.empty()
                st.success(
                    f"{result.filename} — {result.pages} pages, "
                    f"{result.chunks} chunks, ~${result.cost_usd:.4f}"
                )
                st.session_state.ingested[result.filename] = result
                st.session_state.upload_cost += result.cost_usd

            except UploadRejected as exc:
                bar.empty()
                st.error(str(exc))
            except Exception as exc:
                bar.empty()
                st.error(f"Could not index {file.name}: {exc}")

        if pending:
            st.rerun()

    if not session_documents(SESSION_ID):
        st.info("Upload a document to get started.")
        st.stop()

    collection = get_session_collection(SESSION_ID)


# --------------------------------------------------------------------------
# Ask
# --------------------------------------------------------------------------

if is_demo and not st.session_state.history:
    st.markdown("**Try one of these**")
    cols = st.columns(2)
    for i, example in enumerate(DEMO_EXAMPLES):
        with cols[i % 2]:
            st.button(example, key=f"ex{i}", use_container_width=True,
                      on_click=set_question, args=(example,))

question = st.text_input(
    "Ask a question",
    value=st.session_state.question,
    placeholder="What does this document say about ...?",
)

if question:
    started = time.perf_counter()

    try:
        stream = answer_stream(question, k=k, collection=collection)
        sources, _ = next(stream)
    except Exception as exc:
        st.error(f"Retrieval failed: {exc}")
        st.stop()

    if not sources:
        st.warning("Nothing in these documents matched that question.")
        st.stop()

    st.markdown("---")
    placeholder = st.empty()
    collected = ""

    try:
        for _, token in stream:
            if token:
                collected += token
                placeholder.markdown(f"<div class='answer-body'>{collected}▌</div>",
                                     unsafe_allow_html=True)
        placeholder.markdown(f"<div class='answer-body'>{collected}</div>",
                             unsafe_allow_html=True)
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
            st.markdown(f"<div class='cite-meta'>cosine distance "
                        f"{s.distance:.3f}</div>", unsafe_allow_html=True)
            st.text(s.text)

    st.session_state.history.append({"cost": cost, "latency": latency})
