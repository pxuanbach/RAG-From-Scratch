# RAG From Scratch

Sample code for the RAG From Scratch series on [UtilKits](https://www.util-kits.io.vn/en/blog). The 12 lessons start with a 50-line script and end with a RAG service you can deploy. Each folder holds one lesson's code and a README that explains how to run it. The posts explain the ideas, and the numbers in them come from running this code.

[Tiếng Việt](README.vi.md)

## Lessons

| # | Lesson | Level | Status |
|---|---|---|---|
| 01 | [What is RAG? Your first RAG in about 50 lines](01-what-is-rag) | Beginner | ready |
| 02 | [Embeddings and vector search](02-embeddings) | Beginner | coming soon |
| 03 | [Ingestion and chunking](03-chunking) | Beginner | coming soon |
| 04 | [Storing vectors in Postgres with pgvector](04-pgvector) | Beginner | coming soon |
| 05 | [Grounded answers with citations](05-grounded-generation) | Beginner | coming soon |
| 06 | [Evaluating RAG](06-evaluation) | Intermediate | coming soon |
| 07 | [Hybrid search and reranking](07-hybrid-search-reranking) | Intermediate | coming soon |
| 08 | [Query transformation](08-query-transformation) | Intermediate | coming soon |
| 09 | [Contextual retrieval and parent-document indexing](09-smarter-indexing) | Intermediate | coming soon |
| 10 | [Agentic RAG](10-agentic-rag) | Advanced | coming soon |
| 11 | [RAG security](11-rag-security) | Advanced | coming soon |
| 12 | [Ship it: RAG as a service](12-ship-it) | Advanced | coming soon |

Lessons 01 to 05 get a pipeline answering questions. Lessons 06 to 09 measure it and fix what the measurements expose, and 10 to 12 add agents, security and deployment.

## Setup

You need Python 3.10+ and [uv](https://docs.astral.sh/uv/) (or pip).

```bash
git clone https://github.com/pxuanbach/RAG-From-Scratch.git
cd RAG-From-Scratch
uv venv && uv pip install -r requirements.txt
```

The first run downloads the `BAAI/bge-m3` embedding model, about 2.3 GB.

## Run on a free local model or on Claude

The code calls the LLM through the `anthropic` SDK. [LM Studio](https://lmstudio.ai/docs/developer/anthropic-compat) exposes an Anthropic-compatible endpoint, so the same script runs on Claude or on a local model.

```bash
# Local, free (LM Studio running a model such as qwen2.5-7b-instruct-1m)
export ANTHROPIC_BASE_URL=http://127.0.0.1:1234
export ANTHROPIC_API_KEY=lm-studio
export RAG_MODEL=qwen2.5-7b-instruct-1m

# Or Claude: set only a real key; RAG_MODEL defaults to claude-haiku-5-5
export ANTHROPIC_API_KEY=sk-ant-...
```

Then run a lesson from its folder. Each lesson's README lists its commands and expected output.

```bash
cd 01-what-is-rag
uv run python rag_50_lines.py
```

On Windows PowerShell, use `$env:ANTHROPIC_BASE_URL = "http://127.0.0.1:1234"` instead of `export`.

## Data

The lessons use [MultiHop-RAG](https://huggingface.co/datasets/yixuantt/MultiHopRAG) (Tang & Yang, 2024): 609 English news articles from late 2023 and 2,556 questions with answers and supporting evidence. The scripts download it from Hugging Face on first run; this repo does not redistribute it. The dataset is licensed ODC-BY 1.0, and the article text remains the property of its publishers.

## License

The code is released under the [MIT License](LICENSE). The MultiHop-RAG dataset keeps its own ODC-BY 1.0 license.
