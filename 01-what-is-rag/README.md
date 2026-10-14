# 01. What is RAG? Your first RAG in about 50 lines

Lesson: [What Is RAG? Build Your First RAG in About 50 Lines of Python](https://www.util-kits.io.vn/en/blog/2026-10-15-what-is-rag-build-first-rag-python) · [Tiếng Việt](README.vi.md)

`rag_50_lines.py` loads 20 news articles from MultiHop-RAG, keeps the first three paragraphs of each, and embeds the 60 paragraphs with `BAAI/bge-m3`. It then finds the three paragraphs closest to your question and asks an LLM to answer from them with `[n]` citations. The lesson explains each step; this page covers how to run it.

| File | Contents |
|---|---|
| `rag_50_lines.py` | The pipeline: load, embed, retrieve, generate |
| `experiment.py` | Asks one question three ways (no documents, documents pasted into the prompt, RAG) and writes token counts and latency to `results.json` |
| `results.json` | My run with Qwen2.5-7B-Instruct-1M in LM Studio on an RTX 5060 Ti |

## Before you start

1. Install the dependencies from the repo root, as described in the [root README](../README.md#setup).
2. Have one LLM ready. Either start the LM Studio server with a model loaded, or get an Anthropic API key.

The first run downloads the corpus (12 MB) and the `bge-m3` model (about 2.3 GB). A CPU handles 60 paragraphs in under a minute.

## Run

With LM Studio:

```bash
cd 01-what-is-rag
export ANTHROPIC_BASE_URL=http://127.0.0.1:1234
export ANTHROPIC_API_KEY=lm-studio
export RAG_MODEL=qwen2.5-7b-instruct-1m   # the model id LM Studio shows
uv run python rag_50_lines.py
```

With Claude, set only `ANTHROPIC_API_KEY` to your key. `RAG_MODEL` defaults to `claude-haiku-5-5`.

In PowerShell, set variables with `$env:ANTHROPIC_BASE_URL = "http://127.0.0.1:1234"` and so on.

Expected output (your wording will differ, the scores should match):

```text
0.649  TechCrunch: SBF’s trial starts soon, but how did he — and FTX — get here?
0.556  TechCrunch: SBF’s trial starts soon, but how did he — and FTX — get here?
0.454  TechCrunch: SBF’s trial starts soon, but how did he — and FTX — get here?

FTX was founded in 2019 by Sam Bankman-Fried, and at its peak, its valuation hit $32 billion [1].
```

## Ask your own question

Pass the question as arguments:

```bash
uv run python rag_50_lines.py "Who won the 2023 Nobel Prize in Physics?"
```

None of the 20 articles covers the Nobel Prize, so the model should answer that it doesn't know. Two more to try:

- `"How did the Houston Astros cheat in 2017?"` retrieves the right article, yet the model says it doesn't know. The script keeps only the first three paragraphs of each article, and those paragraphs don't describe the scheme. Lesson 03 fixes this with real chunking.
- A question about an article outside the first 20 finds nothing useful. Change `[:20]` in the script to load more.

`bge-m3` is multilingual, so questions in other languages work too. A Vietnamese version of the FTX question still retrieved the TechCrunch article (score 0.643) in my run. Lesson 02 measures cross-language search properly.

## Reproduce the experiment

```bash
uv run python experiment.py --retrieval-only   # no LLM needed, prints the top 3
uv run python experiment.py                    # full run, needs an LLM
```

Both commands overwrite `results.json`. On a local model, the step that pastes all 20 articles into the prompt fails with a context-length error, and the script records that error. Raise the context length in LM Studio if you want that row to succeed.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `Connection error` | Start the LM Studio server and check that it listens on port 1234 |
| Model not found | Set `RAG_MODEL` to an id from `curl http://127.0.0.1:1234/v1/models` |
| Garbled `’` or `—` when you pipe output on Windows | Set `PYTHONIOENCODING=utf-8` |
