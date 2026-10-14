# RAG From Scratch

Companion code for the **RAG From Scratch** series on [UtilKits](https://www.util-kits.io.vn/en/blog). Twelve hands-on lessons take you from a 50-line script to a RAG service you can ship. Every lesson builds something that runs, and every number in the posts comes from running this code.

[Tiếng Việt](README.vi.md)

## Lessons

| # | Lesson | Level | Status |
|---|---|---|---|
| 01 | [What Is RAG? Build Your First RAG in About 50 Lines](01-what-is-rag) | Beginner | ready |
| 02 | [Embeddings and Vector Search](02-embeddings) | Beginner | coming soon |
| 03 | [Ingestion and Chunking](03-chunking) | Beginner | coming soon |
| 04 | [Storing Vectors in Postgres with pgvector](04-pgvector) | Beginner | coming soon |
| 05 | [Grounded Answers with Citations](05-grounded-generation) | Beginner | coming soon |
| 06 | [Evaluating RAG](06-evaluation) | Intermediate | coming soon |
| 07 | [Hybrid Search and Reranking](07-hybrid-search-reranking) | Intermediate | coming soon |
| 08 | [Query Transformation](08-query-transformation) | Intermediate | coming soon |
| 09 | [Contextual Retrieval and Parent-Document Indexing](09-smarter-indexing) | Intermediate | coming soon |
| 10 | [Agentic RAG](10-agentic-rag) | Advanced | coming soon |
| 11 | [RAG Security](11-rag-security) | Advanced | coming soon |
| 12 | [Ship It: RAG as a Service](12-ship-it) | Advanced | coming soon |

Finish 01–05 and you can build a working RAG app. Finish 06–09 and you can measure and improve one. Finish 10–12 and you can run one in production.

## Setup

You need Python 3.10+ and [uv](https://docs.astral.sh/uv/) (or pip).

```bash
git clone https://github.com/pxuanbach/RAG-From-Scratch.git
cd RAG-From-Scratch
uv venv && uv pip install -r requirements.txt
```

The first run downloads the `BAAI/bge-m3` embedding model, about 2.3 GB.

## Run on a free local model or on Claude

The code calls the LLM through the `anthropic` SDK, so the same script runs on Claude or on [LM Studio](https://lmstudio.ai/docs/developer/anthropic-compat), which exposes an Anthropic-compatible endpoint.

```bash
# Local, free (LM Studio running a model such as qwen2.5-7b-instruct-1m)
export ANTHROPIC_BASE_URL=http://127.0.0.1:1234
export ANTHROPIC_API_KEY=lm-studio
export RAG_MODEL=qwen2.5-7b-instruct-1m

# Or Claude: set only a real key; RAG_MODEL defaults to claude-haiku-5-5
export ANTHROPIC_API_KEY=sk-ant-...
```

Then run a lesson from its folder:

```bash
cd 01-what-is-rag
uv run python rag_50_lines.py
```

On Windows PowerShell, use `$env:ANTHROPIC_BASE_URL = "http://127.0.0.1:1234"` instead of `export`.

## Data

The lessons use [MultiHop-RAG](https://huggingface.co/datasets/yixuantt/MultiHopRAG) (Tang & Yang, 2024): 609 English news articles from late 2023 and 2,556 questions with answers and supporting evidence. The scripts download it from Hugging Face on first run; this repo does not redistribute it. The dataset is licensed ODC-BY 1.0, and the article text remains the property of its publishers.
