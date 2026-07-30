# Manual recall@5 — baseline

Run: 2026-07-30 12:01

Judged by hand: for each question, is a passage that actually answers it present in the top 5 retrieved chunks?

## Config

```json
{
  "embed_model": "text-embedding-3-small",
  "chat_model": "gpt-4o-mini",
  "chunk_tokens": 500,
  "chunk_overlap": 80,
  "top_k": 5
}
```

## Result

**recall@5 = 1/1 = 1.00**

| Category | Hits | Total | Recall |
|---|---|---|---|
| regulatory_vocabulary | 1 | 1 | 1.00 |

## Per question

### Q01 — HIT

> What obligations apply to providers of high-risk AI systems?

*Category:* regulatory_vocabulary

*Retrieved:* the_digital_service_act_2022.pdf p.23 (0.4433), the_digital_service_act_2022.pdf p.26 (0.4721), the_digital_service_act_2022.pdf p.27 (0.4825), the_digital_service_act_2022.pdf p.64 (0.4942), the_digital_service_act_2022.pdf p.24 (0.4946)

## What this means for step 8

_Fill in after judging: which failure mode dominates, and which single change is most likely to fix it?_
