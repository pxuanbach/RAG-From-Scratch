# What Is RAG? Build Your First RAG in About 50 Lines of Python

Read on the blog: [https://www.util-kits.io.vn/en/blog/2026-10-15-what-is-rag-build-first-rag-python](https://www.util-kits.io.vn/en/blog/2026-10-15-what-is-rag-build-first-rag-python) · [Tiếng Việt](README.vi.md)

*Part 1 of RAG From Scratch. This post: what RAG is, and a working pipeline you can read line by line.*

> **Level:** Beginner (post 1 of 5 in the Beginner stage)
> **For:** developers who know basic Python and have used ChatGPT or Claude, but have never built an LLM app. PMs and founders can read the concept sections and skip the code.
> **You need:** Python 3.10+, `uv` or `pip`, and either LM Studio (free, local) or an Anthropic API key.
> **After this post you can:** explain retrieve, augment, generate in your own words, run a RAG script without a framework, and measure the token cost of one question.

I asked a local 7B model a simple question: what is the hotel cap per night in Tokyo under the Acme Robotics travel policy. Acme Robotics is a company I made up for this test, so no model has seen its policy. The model replied that it had no information about internal policies and suggested I contact HR.

Then I pasted a four-sentence policy document into the prompt and asked again. The answer came back in a third of a second: "240 USD [1]".

That second request is retrieval-augmented generation (RAG) in miniature. This series builds the version where code finds the right document among thousands and pastes it in for you.

## What RAG does

A [RAG](https://www.util-kits.io.vn/en/glossary#rag) system answers a question in three steps:

1. **Retrieve:** search your documents for the passages most related to the question.
2. **Augment:** put those passages into the prompt, next to the question.
3. **Generate:** the LLM writes an answer based on the passages it was given.

The search step uses embeddings: each passage becomes a list of numbers (a vector), and passages with similar meaning end up with similar vectors. Post 2 covers how that works. For now, treat `embed()` as a function that turns text into a vector you can compare.

## RAG, fine-tuning, or a bigger prompt

There are three common ways to give an LLM knowledge it lacks:

| Approach | Good for | Weak at |
|---|---|---|
| Fine-tuning | Teaching a style, format, or task | Facts that change often, citing sources |
| Put every document in the prompt | A handful of short documents | Cost and speed once documents pile up, context limits |
| RAG | Large or changing document sets, answers with citations | Answers that depend on documents the search step misses |

Fine-tuning shapes *how* a model responds. RAG supplies *what* it should know at question time. The experiment below shows where the "bigger prompt" option breaks down.

## The data: 609 real news articles

The series uses [MultiHop-RAG](https://huggingface.co/datasets/yixuantt/MultiHopRAG), a public dataset built for testing RAG systems. It contains 609 English news articles published between September and December 2023, about 1.25 million tokens in total, plus 2,556 questions that come with answers and the articles that support them. Later posts use those questions to measure the pipeline. The dataset is licensed ODC-BY 1.0.

This post keeps things small: the first 20 articles, three paragraphs from each, 60 passages in total.

Install the four packages:

```bash
uv venv && uv pip install sentence-transformers huggingface_hub anthropic numpy
```

The first run downloads the `BAAI/bge-m3` embedding model, about 2.3 GB. A GPU helps but a CPU handles 60 passages fine.

## The code

```python
# rag_50_lines.py
import json
import os
import sys

import anthropic
import numpy as np
from huggingface_hub import hf_hub_download
from sentence_transformers import SentenceTransformer

# 1. Load 20 news articles and keep the first 3 real paragraphs of each
path = hf_hub_download("yixuantt/MultiHopRAG", "corpus.json", repo_type="dataset")
articles = json.load(open(path, encoding="utf-8"))[:20]
docs = []
for a in articles:
    paragraphs = [p.strip() for p in a["body"].split("\n\n") if len(p.strip()) > 80]
    for p in paragraphs[:3]:
        docs.append({"title": a["title"], "source": a["source"], "text": p})

# 2. Embed every paragraph once (normalized, so dot product = cosine similarity)
model = SentenceTransformer("BAAI/bge-m3")
doc_vecs = model.encode([d["text"] for d in docs], normalize_embeddings=True)


def retrieve(question, k=3):
    q_vec = model.encode([question], normalize_embeddings=True)[0]
    scores = doc_vecs @ q_vec
    top = np.argsort(-scores)[:k]
    return [(docs[i], float(scores[i])) for i in top]


SYSTEM = (
    "Answer using only the documents provided. Cite documents as [1], [2]. "
    "If the documents do not contain the answer, say you don't know."
)


def ask(question):
    hits = retrieve(question)
    context = "\n\n".join(
        f"[{n}] {d['title']} ({d['source']})\n{d['text']}" for n, (d, _) in enumerate(hits, 1)
    )
    # Reads ANTHROPIC_API_KEY / ANTHROPIC_BASE_URL, so the same code runs on Claude or LM Studio
    response = anthropic.Anthropic().messages.create(
        model=os.environ.get("RAG_MODEL", "claude-haiku-5-5"),
        max_tokens=2000,
        system=SYSTEM,
        messages=[{"role": "user", "content": f"<documents>\n{context}\n</documents>\n\n{question}"}],
    )
    return "".join(b.text for b in response.content if b.type == "text"), hits


if __name__ == "__main__":
    answer, hits = ask(" ".join(sys.argv[1:]) or "When was FTX founded and what was it worth at its peak?")
    for d, score in hits:
        print(f"{score:.3f}  {d['source']}: {d['title']}")
    print("\n" + answer)
```

The script maps onto the three steps:

- **Retrieve** is `retrieve()`. It embeds the question, scores it against all 60 passage vectors with one matrix multiplication, and keeps the top 3. Because the vectors are normalized, the dot product equals cosine similarity.
- **Augment** is the `context` string in `ask()`. Each passage gets a number so the model can cite it.
- **Generate** is the `messages.create` call. The system prompt tells the model to stay inside the documents and to say so when they don't contain the answer.

The `len(p) > 80` filter is a crude way to skip newsletter sign-up boxes and short captions in the raw articles. Post 3 replaces it with real cleaning and chunking.

## Run it on a free local model

[LM Studio](https://www.util-kits.io.vn/en/blog/2026-09-01-local-ai-lm-studio-guide) exposes an Anthropic-compatible `/v1/messages` endpoint, so the same code runs locally by changing three environment variables:

```bash
export ANTHROPIC_BASE_URL=http://127.0.0.1:1234
export ANTHROPIC_API_KEY=lm-studio
export RAG_MODEL=qwen2.5-7b-instruct-1m
uv run python rag_50_lines.py
```

Unset them and set a real `ANTHROPIC_API_KEY` to run the same script on Claude Haiku 5.5. My output with the local model:

```text
0.649  TechCrunch: SBF's trial starts soon, but how did he — and FTX — get here?
0.556  TechCrunch: SBF's trial starts soon, but how did he — and FTX — get here?
0.454  TechCrunch: SBF's trial starts soon, but how did he — and FTX — get here?

FTX was founded in 2019 by Sam Bankman-Fried, and at its peak, it was valued at $32 billion [1].
```

All three passages came from the one article that holds the answer, and the answer matches it.

## Three ways to answer the same question

I asked "When was FTX founded and what was it worth at its peak?" three ways, using Qwen2.5-7B-Instruct-1M in LM Studio on an RTX 5060 Ti. The cost column estimates what the same token counts would cost on Claude Haiku 5.5 ($0.10 per million input tokens, $0.50 per million output tokens). The token counts come from Qwen's tokenizer, so treat cost as approximate.

| Approach | Input tokens | Latency | Answer | Est. cost on Haiku 5.5 |
|---|---|---|---|---|
| No documents | 40 | 0.74 s | Wrong founding year (2017), plus a co-founder name that appears in none of the articles | $0.000025 |
| All 20 articles in the prompt | 39,965 | failed | Over the 13,056-token context the local model was loaded with | n/a |
| 5 articles in the prompt | 10,535 | 4.0 s | Correct (2019, $32 billion) | $0.0011 |
| RAG, top 3 passages | 311 | 0.74 s | Correct, with citations | $0.000051 |

From the table:

1. **The model without documents sounded sure and got it wrong.** It mixed the 2017 founding of Alameda Research with FTX, which the TechCrunch article dates to 2019. The $32 billion figure was right, which makes the wrong year harder to spot.
2. **Stuffing documents runs out of room early.** Twenty articles produced a 39,965-token prompt, three times the context LM Studio had allocated. You can raise that limit, but a bigger context window needs more GPU memory.
3. **When stuffing does fit, RAG still uses about 34 times fewer tokens** for the same correct answer (311 vs 10,535), and returned in under a fifth of the time.
4. **The full corpus doesn't fit at all.** All 609 articles come to roughly 1.25 million tokens, more than Claude Haiku 5.5's 1 million token context. If it fit, each question would cost around $0.62 at the long-prompt rate. RAG stays near 300 tokens per question however large the corpus grows.

## What this script gets wrong

The script works on this question and will mislead you on harder ones. RAG reduces hallucination but adds a new failure: when retrieval returns the wrong passages, the model builds a wrong answer on them and cites them as proof. Later posts close these gaps:

- Passages are cut by a crude filter instead of a chunking strategy (post 3).
- Vectors live in memory and get recomputed on every run (post 4).
- Nothing measures whether retrieval found the right article (post 6).
- Searches for exact names and numbers can miss (post 7).

The local test also showed a practical limit: the [context window](https://www.util-kits.io.vn/en/glossary#context-window) your hardware can afford is smaller than the one a model advertises. Long prompts also cost answer quality, as the [context compression post](https://www.util-kits.io.vn/en/blog/2026-10-12-context-compression-coding-agents) explains.

Next, post 2 opens up `embed()`: what an embedding is, how to pick a model, and whether a Vietnamese question can find an English article.

## Check yourself

1. Why does RAG reduce hallucination without removing it?
2. Your documents change every day. Would you pick RAG or fine-tuning, and why?
3. In `rag_50_lines.py`, which lines do the retrieve step and which do the augment step?

## Sources

1. [MultiHop-RAG dataset card](https://huggingface.co/datasets/yixuantt/MultiHopRAG) on Hugging Face: corpus, questions, ODC-BY license.
2. [Tang & Yang, MultiHop-RAG: Benchmarking Retrieval-Augmented Generation for Multi-Hop Queries](https://arxiv.org/abs/2401.15391): article count, date range and average article length.
3. [BAAI/bge-m3 model card](https://huggingface.co/BAAI/bge-m3): the embedding model used here.
4. [LM Studio docs: Anthropic-compatible endpoints](https://lmstudio.ai/docs/developer/anthropic-compat): the `/v1/messages` endpoint.
5. [Anthropic pricing](https://platform.claude.com/docs/en/about-claude/pricing): Claude Haiku 5.5 rates used for the cost estimates.
6. Further reading: [Lewis et al., Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401), the paper that introduced the term.
