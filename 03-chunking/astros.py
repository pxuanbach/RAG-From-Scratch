"""Part 3: the Astros question from lesson 01, now over all 609 cleaned articles.

Usage: uv run python astros.py [--ask]      (writes results/astros.json)
For each chunking strategy: where does the first chunk that describes the scheme (camera, trash can) rank, and
is it inside the top 3 or inside a 2,000-token context? With --ask, the best strategy's chunks that fit 2,000 tokens
go to the LLM (LM Studio or Claude, read from the .env in the repo root).
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

from ragkit.chunk import DEFAULT, STRATEGIES, make_chunks
from ragkit.corpus import load_clean_articles
from ragkit.embed import LocalEmbedder

QUESTION = "How did the Houston Astros cheat in 2017?"
SCHEME = ("trash can", "camera")  # the scheme is described in four paragraphs in the middle of the article
BUDGET = 2000

articles = load_clean_articles()
emb = LocalEmbedder("BAAI/bge-m3", max_seq_length=1024, batch_size=16)
q_vec = emb.embed_query(QUESTION)
report = {"question": QUESTION, "budget_tokens": BUDGET, "strategies": []}
best = None
for name, strategy in STRATEGIES.items():
    chunks = make_chunks(articles, strategy)
    order = np.argsort(-(emb.embed_documents([c.text for c in chunks]) @ q_vec))
    has = [any(w in chunks[i].raw_text.lower() for w in SCHEME) for i in order[:300]]
    rank = has.index(True) + 1 if True in has else None
    sizes = np.cumsum([chunks[i].meta["tokens"] for i in order[:300]])
    in_budget = max(1, int((sizes <= BUDGET).sum()))
    row = {"strategy": name, "first_scheme_chunk_rank": rank, "chunks_in_budget": in_budget,
           "scheme_in_top3": bool(rank and rank <= 3), "scheme_in_budget": bool(rank and rank <= in_budget),
           "top3": [{"source": chunks[i].meta["source"], "chunk_index": chunks[i].meta["chunk_index"],
                     "tokens": chunks[i].meta["tokens"]} for i in order[:3]]}
    report["strategies"].append(row)
    print(f"{name:30s} first chunk with the scheme ranks {str(rank):>4}   in top 3: {row['scheme_in_top3']!s:5}   "
          f"in a {BUDGET}-token context ({in_budget} chunks): {row['scheme_in_budget']}")
    if name == DEFAULT:
        best = [chunks[i] for i in order[:in_budget]]

if "--ask" in sys.argv:
    import anthropic
    from dotenv import load_dotenv
    load_dotenv()
    system = ("Answer using only the documents provided. Cite documents as [1], [2]. "
              "If the documents do not contain the answer, say you don't know.")
    context = "\n\n".join(f"[{n}] {c.meta['title']} ({c.meta['source']})\n{c.raw_text}" for n, c in enumerate(best, 1))
    model = os.environ.get("RAG_MODEL", "claude-haiku-5-5")
    r = anthropic.Anthropic().messages.create(model=model, max_tokens=500, system=system, messages=[
        {"role": "user", "content": f"<documents>\n{context}\n</documents>\n\n{QUESTION}"}])
    report.update(model=model, ask_strategy=DEFAULT, answer="".join(b.text for b in r.content if b.type == "text"))
    print(f"\n{model}, {DEFAULT}, {len(best)} chunks:\n{report['answer']}")
Path("results").mkdir(exist_ok=True)
Path("results/astros.json").write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
