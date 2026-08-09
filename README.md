# Document RAG Assistant

Question answering over PDF documents, where every claim is numbered back to the
passage it came from. Ships with a built-in demo over EU regulation (GDPR, the Digital
Services Act) and a mode where you upload your own PDFs and ask questions about them
instead.

**[Live demo](https://domainragassistant-production.up.railway.app/)** · Built with Python, Chroma, OpenAI,
FastAPI, Streamlit, Docker


## What it does

Ask a question and get an answer built only from the indexed documents, with a citation
on every factual sentence. Open any citation to read the original passage and check the
claim yourself.

When the documents do not contain the answer, it says so rather than guessing — including
when the question assumes something the source material does not say.

Two modes:
- **Demo** — a pre-built index over a fixed EU regulation corpus, shared and read-only.
- **Upload** — bring your own PDFs. Each visitor gets an isolated, per-session index that
  is deleted on request and swept automatically after a TTL.

**Demo corpus:** the_digital_service_act_2022.pdf, GDPR_2016


---

## Why the citations matter

A RAG system that produces fluent, confident, unsupported prose is worse than no system
at all in a regulatory context — it is wrong in a way that is expensive to detect. Three
things in this build exist to make that failure visible:

- Every factual sentence carries a `[n]` marker tied to a specific document and page.
- The UI shows the **full retrieved passage**, not a preview, so a claim can be checked
  without leaving the page.
- The system is instructed to refuse rather than assemble an answer from loosely related
  material, and refusal behaviour is measured, not assumed.

---

## Architecture

![Architecture](docs/flowchart.png)

| Stage | Choice | Reasoning |
|---|---|---|
| Extraction | pdfplumber, page by page | Page numbers are carried through the whole pipeline so citations resolve to something a reader can verify. |
| Chunking | Fixed token windows, 500/80 overlap, within page boundaries | Chunks never span pages, so every chunk has exactly one page number. Costs a little context at page seams; buys citation precision. |
| Embedding | `text-embedding-3-small`, batched | One config constant is read by both ingestion and query, so the two cannot silently diverge into different vector spaces. |
| Vector store | Chroma, cosine space, persistent | Cosine is set explicitly at collection creation — Chroma's default is L2, which ranks differently. |
| Generation | `gpt-4o-mini`, temperature 0 | Extraction task, not a creative one. The same excerpts must produce the same answer, or the evaluation is not repeatable. |



---

## Results


Evaluated on **30 hand-written question/answer pairs** plus 5 unanswerable controls,
scored with RAGAS.

| Metric | Score |
|---|---|
| Context recall | 0.975 |
| Context precision | 0.949 |
| Faithfulness | 0.996 |
| Answer relevancy | 0.883 |
| Page-match recall@5 | 0.933 |
| Refusal rate (unanswerable) | 1.000 |
| Cost per query | $0.0005 |

**Faithfulness 0.996** — answers are almost entirely supported by the retrieved
passages. Combined with a 5/5 refusal rate on unanswerable and false-premise
questions, including one asking which article bans AI in agriculture, the system
does not invent provisions.

**Answer relevancy 0.883 is the outlier, and the interesting one.** Retrieval and
grounding are near-ceiling while relevancy trails by roughly ten points — the
answers are well-supported but not always well-aimed at what was asked. That points
at generation, not retrieval: a prompt and answer-shaping problem rather than a
search problem.

### On the high scores

These numbers are higher than a naive pipeline should produce, and the reason is the
test set rather than the system. Each question was written against a specific
provision using that provision's vocabulary, which is the case dense retrieval
handles best. 28 of 30 questions score exactly 1.0 on context recall.

A harder test set — questions phrased the way a non-lawyer would ask them, questions
whose answers span two documents, questions answered by tables in the annexes —
would produce lower and more informative scores. That is the next piece of work, and
it has to come before any retrieval change can be meaningfully measured.

### The experiment

Not yet run. With context recall at 0.975 there is no measurable headroom, so a
harder evaluation set is a prerequisite rather than an optional refinement.



---

## Running it

```bash
git clone https://github.com/mo-goodarzi/EU_Regulation_Assistant
cd EU_Regulation_Assistant

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env        # add your OPENAI_API_KEY

# add PDFs to corpus/, then build the index
python src/ingest.py

streamlit run src/app.py
```

**Docker:**

```bash
docker build -t document-rag-assistant .
docker run -p 8501:8501 -e OPENAI_API_KEY=$OPENAI_API_KEY document-rag-assistant
```

The image ships with a pre-built index rather than embedding at startup — re-embedding on
every container start costs money and minutes on each restart and cold boot. The trade-off
is that the container cannot re-index itself; `corpus/` is excluded from the image
entirely.

**API:**

```bash
uvicorn api:app --reload --app-dir src   # http://localhost:8000/docs
```

`POST /query` returns the answer, sources with page numbers, latency, and cost per query.
`GET /health` reports chunk count and both model names — the most confusing possible
failure is an index built with one embedding model being queried with another, which
produces quietly wrong results rather than an error.

---

## Evaluation

```bash
python eval/run_eval.py --limit 3 --no-ragas   # free, checks the plumbing
python eval/run_eval.py --label baseline       # full run
python eval/compare.py results/A.json results/B.json
```

Every results file records the config that produced it — chunk size, overlap, top-k, both
model names, and the judge model. A score without its config is not reproducible, and the
before/after comparison would mean nothing.

---

## Limitations

- **Not legal advice.** A retrieval system over regulatory text, nothing more.
- **No OCR.** Scanned PDFs without a text layer are skipped at ingestion.
- **English only.** In demo mode, questions outside the indexed corpus are refused by
  design rather than answered from the model's general knowledge; the same holds for
  uploaded documents in upload mode.
- **Uploaded documents are not persisted.** Session collections live on local disk and
  are not backed up — a redeploy or restart loses them, by design.
- **Evaluated on NN questions** written by one person. Small, and reflects my own sense of
  what matters in these documents.
- **Chunks do not span page boundaries**, so a provision split across a page break is
  split across chunks. Deliberate — see the architecture table — but it is a real cost.
- **Headers and footers are not stripped.** Repeated boilerplate rides along in every
  chunk, consuming context and slightly compressing the similarity score distribution.
  Measurable as a future experiment.
- **Faithfulness is judged by an LLM**, which has its own error rate. The page-match metric
  is the deterministic control, and it only checks retrieval, not the answer.

---
