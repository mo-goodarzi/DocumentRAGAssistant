"""Turn retrieved chunks into a grounded, cited answer."""

from dataclasses import dataclass

from openai import OpenAI

import config
from retrieve import Source, retrieve

client = OpenAI()

SYSTEM_PROMPT = """You answer questions about regulatory documents using ONLY the \
numbered excerpts provided.

Rules:
1. Every factual sentence must end with a citation: [1], or [2][3] if it draws on more \
than one excerpt.
2. Use only what the excerpts say. Do not add background knowledge about EU law, even \
if you are confident it is correct.
3. If the excerpts do not contain the answer, say so directly and name what is missing. \
Do not assemble a partial answer from loosely related material.
4. If the question assumes something the excerpts contradict or do not support, say that \
before answering anything else.
5. Quote the regulation's exact wording only where the precise phrasing carries legal \
weight. Otherwise paraphrase.
6. Be concise. No preamble, no restating the question, no closing summary."""


@dataclass
class Answer:
    text: str
    sources: list[Source]
    cost_usd: float


def format_context(sources: list[Source]) -> str:
    """Number the excerpts so the model has something concrete to cite.

    Numbering matches Source.n, which is also what ask.py prints — so a [3] in
    the answer maps to the third block you can see on screen.
    """
    return "\n\n".join(
        f"[{s.n}] ({s.source}, p.{s.page})\n{s.text}" for s in sources
    )


def cost_of(response) -> float:
    """USD for one chat completion. Prices in config are per million tokens."""
    u = response.usage
    return (u.prompt_tokens * config.PRICE_IN
            + u.completion_tokens * config.PRICE_OUT) / 1_000_000


def answer(question: str, k: int | None = None, collection=None) -> Answer:
    """Retrieve, then generate an answer grounded in what came back.

    temperature=0 because this is an extraction task, not a creative one: given
    the same excerpts, the same question should produce the same answer. Any
    variation here is noise that would also make the step 7 eval unrepeatable.
    """
    sources = retrieve(question, k=k, collection=collection)

    if not sources:
        return Answer(
            text="Nothing in the indexed documents matched that question.",
            sources=[],
            cost_usd=0.0,
        )

    response = client.chat.completions.create(
        model=config.CHAT_MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Excerpts:\n\n{format_context(sources)}\n\n"
                    f"Question: {question}"
                ),
            },
        ],
    )

    return Answer(
        text=response.choices[0].message.content.strip(),
        sources=sources,
        cost_usd=cost_of(response),
    )


def answer_stream(question: str, k: int | None = None, collection=None):
    """Yield (sources, token) — sources once, then text as it arrives.

    Needed by the Streamlit UI in step 10, where waiting on a full response
    feels broken. Usage data is not available when streaming, so no cost here.
    """
    sources = retrieve(question, k=k, collection=collection)
    yield sources, None

    if not sources:
        yield None, "Nothing in the indexed documents matched that question."
        return

    stream = client.chat.completions.create(
        model=config.CHAT_MODEL,
        temperature=0,
        stream=True,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Excerpts:\n\n{format_context(sources)}\n\n"
                    f"Question: {question}"
                ),
            },
        ],
    )

    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield None, delta