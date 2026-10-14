# 02. Embeddings and vector search

Lesson: [Embeddings and Vector Search Explained: Pick a Model With a 10-Question Test](https://www.util-kits.io.vn/en/blog/2026-10-22-embeddings-vector-search-explained-rag) · [Tiếng Việt](README.vi.md)

This is where the `ragkit/` package starts. It holds an `Embedder` interface with a local sentence-transformers implementation, a disk cache for vectors, and a 10-question smoke test (`eval/smoke.jsonl`) that measures whether the right article lands in the top 3. `compare_models.py` runs the test on `bge-m3` and `multilingual-e5-small`, with English and Vietnamese questions against the English corpus. The lesson explains the results; this page covers how to run them.

| File | Contents |
|---|---|
| `ragkit/corpus.py` | Loads MultiHop-RAG articles and questions from Hugging Face |
| `ragkit/embed.py` | `Embedder` interface, `LocalEmbedder`, `OpenAIEmbedder` (untested), on-disk vector cache |
| `ragkit/search.py` | Brute-force top-k with a matrix multiplication |
| `ragkit/smoke.py` | Loads the smoke set and computes hit@3 and the rank of the best evidence article |
| `eval/smoke.jsonl` | 10 questions: English text, Vietnamese translation, expected article URLs. Lesson 03 reuses it |
| `eval/build_smoke.py` | How the 10 questions were picked (documented, no need to rerun) |
| `intuition.py` | Cosine, dot product and L2 on four real sentences |
| `compare_models.py` | Embeds all 609 articles per config and runs the smoke test |
| `results/` | `intuition.json` and `compare.json`: my run on an RTX 5060 Ti, 2026-10-14 |

## Before you start

1. Install the dependencies from the repo root, as described in the [root README](../README.md#setup). No LLM is needed for this lesson.
2. A GPU helps. On CPU, `compare_models.py` takes several times longer (the 2048-token `bge-m3` pass is the slow one).

The first run downloads `bge-m3` (about 2.3 GB) and `multilingual-e5-small` (about 0.5 GB). Embeddings are cached under `../.cache/embeddings/`, so a second run finishes in seconds.

## Run

```bash
cd 02-embeddings
uv run python intuition.py
uv run python compare_models.py
```

Expected output of `intuition.py` (scores match to three decimals on my machine):

```text
dimension: 1024, vector norm: 1.000

cosine similarity (A/B: same trial, different words; C: football; D: Vietnamese version of A)
  A vs B: 0.595
  A vs C: 0.255
  A vs D: 0.867
  B vs C: 0.305
  B vs D: 0.607
  C vs D: 0.308

normalized vectors, A vs C: cosine 0.255, dot 0.255, L2 1.220, 2 - 2*cosine = 1.489 = L2 squared 1.489
scale A by 3: dot 0.766 (changed), cosine 0.255 (same)
```

Expected output of `compare_models.py` (each real line also ends with the embedding time in seconds, which is 0.0 when vectors come from the cache; `results/compare.json` keeps the first-run timings):

```text
bge-m3, 2048 tokens                      dim 1024  truncated 275/609  EN hit@3 1.0  VI hit@3 1.0  both@3 0.4/0.4  mean rank 1.3/1.1
bge-m3, 512 tokens                       dim 1024  truncated 609/609  EN hit@3 0.9  VI hit@3 0.9  both@3 0.4/0.4  mean rank 1.9/2.1
multilingual-e5-small, with prefixes     dim  384  truncated 609/609  EN hit@3 0.9  VI hit@3 0.6  both@3 0.3/0.3  mean rank 1.6/5.3
multilingual-e5-small, no prefixes       dim  384  truncated 609/609  EN hit@3 0.9  VI hit@3 0.6  both@3 0.3/0.3  mean rank 2.0/4.2
```

"hit@3" counts a question as solved when at least one expected article is in the top 3. "both@3" requires all expected articles, and "mean rank" is the average position of the best-placed expected article among all 609.

## Try your own model

Add a line to `CONFIGS` in `compare_models.py`. Read the model card first for the prefixes and the maximum input length:

```python
("my model", dict(model_name="org/model-name", query_prefix="query: ", doc_prefix="passage: ", max_seq_length=512)),
```

You can also add your own questions to `eval/smoke.jsonl` (same fields: `id`, `q_en`, `q_vi`, `expected_urls`).

`OpenAIEmbedder` in `ragkit/embed.py` shows the API fallback. It was written but never run, because I had no API key. Install `openai`, set `OPENAI_API_KEY`, and treat it as a starting point.

## Troubleshooting

| Symptom | Fix |
|---|---|
| CUDA out of memory in `compare_models.py` | Lower `batch_size` for the `bge-m3` 2048-token config, or close other GPU programs |
| Garbled Vietnamese in the terminal on Windows | Set `PYTHONIOENCODING=utf-8` |
| Numbers differ slightly from the lesson | GPU and library versions change the last digits. Ranks and hit counts should match |
| Stale results after changing a config | Delete `../.cache/embeddings/`. The cache key covers the model, max length and prefix, not other settings |
