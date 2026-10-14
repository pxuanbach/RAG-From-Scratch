# 03. Ingestion and chunking

Lesson: [RAG Chunking Strategies in Python: Clean and Split Real Articles, Then Measure](https://www.util-kits.io.vn/en/blog/2026-10-29-rag-chunking-strategies-python) · [Tiếng Việt](README.vi.md)

This folder extends lesson 02's `ragkit/` with a cleaning step (`corpus.py`) and four chunking strategies (`chunk.py`). `compare_chunking.py` splits the 609 cleaned articles seven ways, embeds every chunk with `bge-m3`, and scores retrieval on 150 sampled questions plus the 10-question smoke set. No LLM is needed except for the optional `astros.py --ask`.

| File | Contents |
|---|---|
| `ragkit/corpus.py` | Lesson 02's loaders plus `find_boilerplate`, `clean_body`, `load_clean_articles` |
| `ragkit/chunk.py` | `Chunk`, the strategies (`fixed`, `recursive`, `by_paragraph`), content-hash ids, `make_chunks` |
| `ragkit/smoke.py` | Lesson 02's smoke test plus `run_chunk_eval` (hit, all, evidence found, by top-k or token budget) |
| `ragkit/embed.py`, `ragkit/search.py` | Unchanged from lesson 02 |
| `eval/sample150.jsonl` | 150 questions (50 per type) with evidence sentences; `eval/build_sample.py` builds it |
| `eval/smoke.jsonl` | The 10 smoke questions from lesson 02 |
| `junk_stats.py` | Counts the junk in all 609 articles, cleans them, checks no evidence sentence was lost |
| `compare_chunking.py` | Chunks, embeds and scores every strategy, with and without the metadata header |
| `analyze.py` | Paired bootstrap intervals: which differences are bigger than noise |
| `junk_in_topk.py` | How much junk reaches the top 5 when you skip cleaning |
| `chunk_ids.py` | How many chunk ids survive a small edit, content hash versus position |
| `astros.py` | The Astros question from lesson 01 over all 609 articles |
| `results/` | `junk_stats`, `chunking`, `significance`, `junk_in_topk`, `chunk_ids`, `astros` (JSON): my run on an RTX 5060 Ti, 2026-10-14 |

## Before you start

1. Install the dependencies from the repo root, as described in the [root README](../README.md#setup).
2. A GPU helps. `compare_chunking.py` embeds about 70,000 chunks in total (first run: about 25 minutes on my GPU, a cached rerun about 4 minutes).
3. Only `astros.py --ask` needs an LLM. Set the LM Studio or Claude variables from the root README.

Embeddings are cached in `../.cache/embeddings/` (file `BAAI__bge-m3_len1024_0p.npz`, about 300 MB). Delete that file to recompute them.

## Run

```bash
cd 03-chunking
uv run python junk_stats.py
uv run python eval/build_sample.py      # optional, sample150.jsonl is already committed
uv run python compare_chunking.py
uv run python analyze.py
uv run python junk_in_topk.py
uv run python chunk_ids.py
uv run python astros.py --ask
```

Expected output of `junk_stats.py` (key lines):

```text
author_null: 64
author_empty_string: 4
author_string_None: 0
replacement_char_U+FFFD: 0
articles_with_invisible_chars: 16
articles_with_any_junk_paragraph: 213
paragraphs_total: 29011
paragraphs_after_cleaning: 28107
tokens_before: 1548729
tokens_after: 1537038
evidence_facts_found_in_clean_body: 6084
```

Expected output of `compare_chunking.py` (each real line also ends with the embedding time in seconds, 0.0 on a cache hit; `results/chunking.json` keeps the first-run timings). `hit@3` counts a question when one of the top 3 chunks comes from an expected article; `fact@k` is the share of evidence sentences found whole in the top k chunks; `fact@2000t` uses as many chunks as fit in 2,000 tokens; `intact` is the share of evidence sentences not cut by a chunk boundary:

```text
whole article (2048-token cut)    chunks    609  tok 2543.7  intact 1.000  hit@3 0.867  all@5 0.340  fact@3 0.536  fact@5 0.654  fact@2000t 0.255
fixed 256                         chunks   6305  tok  243.9  intact 0.870  hit@3 0.833  all@5 0.267  fact@3 0.271  fact@5 0.357  fact@2000t 0.428
fixed 512                         chunks   3303  tok  465.4  intact 0.944  hit@3 0.840  all@5 0.300  fact@3 0.320  fact@5 0.440  fact@2000t 0.354
fixed 512 + overlap 64            chunks   3641  tok  475.5  intact 1.000  hit@3 0.833  all@5 0.240  fact@3 0.326  fact@5 0.404  fact@2000t 0.349
recursive 256                     chunks   7420  tok  207.1  intact 1.000  hit@3 0.827  all@5 0.253  fact@3 0.290  fact@5 0.384  fact@2000t 0.495
recursive 512                     chunks   3543  tok  433.8  intact 1.000  hit@3 0.807  all@5 0.273  fact@3 0.311  fact@5 0.399  fact@2000t 0.371
paragraph (min 128, max 512)      chunks   9180  tok  167.4  intact 1.000  hit@3 0.840  all@5 0.313  fact@3 0.265  fact@5 0.362  fact@2000t 0.537
```

The script prints a second line per strategy for the `+ header` variant, one for recursive 512 on uncleaned text, and the mean rank and smoke-test result; the numbers above are the ones quoted in the lesson.

Expected output of `astros.py --ask` (key lines):

```text
fixed 256                      first chunk with the scheme ranks    2   in top 3: True    in a 2000-token context (7 chunks): True
paragraph (min 128, max 512)   first chunk with the scheme ranks    4   in top 3: False   in a 2000-token context (12 chunks): True
```

It ends with the model's answer, which changes in wording from run to run.

## Things to try

- Change `min_tokens` or `max_tokens` in `by_paragraph` and rerun `compare_chunking.py`. Only new chunk texts are embedded.
- Add a strategy to `STRATEGIES` in `ragkit/chunk.py` (for example sentence windows) and see where it lands.
- Run `analyze.py` on your own variants before you believe a 0.02 difference.

## Troubleshooting

| Symptom | Fix |
|---|---|
| CUDA out of memory | Lower `batch_size` in `compare_chunking.py` (16 for the chunk model, 4 for the whole-article baseline) |
| Garbled characters on Windows | Set `PYTHONIOENCODING=utf-8` |
| Numbers differ in the last digit | GPU and library versions change vectors slightly. Chunk counts and cleaning stats should match exactly |
| `astros.py --ask` fails to connect | Start LM Studio and set the variables from the root README, or run without `--ask` |
