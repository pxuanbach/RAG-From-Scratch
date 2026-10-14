"""Experiment for post 1: closed-book vs stuff-everything vs RAG top-3.

Usage:
    uv run python experiment.py --retrieval-only  # no API key needed
    uv run python experiment.py                   # needs Anthropic credentials

Writes results.json. Prices: Claude Haiku 5.5, $0.10 / $0.50 per MTok
(prompts up to 100K tokens), $0.50 / $2.50 beyond (Anthropic pricing page, checked 2026-10-06).
"""
import json
import os
import sys
import time
from pathlib import Path

from rag_50_lines import SYSTEM, articles, docs, retrieve

QUESTION = "When was FTX founded and what was it worth at its peak?"
INTERNAL_DOC = (
    "Acme Robotics travel policy (internal, 2026 revision). Employees book flights "
    "through the Wayfare portal. Economy class for flights under 6 hours, premium "
    "economy above 6 hours. Hotel cap: 180 USD per night, 240 USD in Tokyo and London. "
    "Receipts must be uploaded within 10 days."
)
INTERNAL_Q = "What is the hotel cap per night in Tokyo under the Acme Robotics travel policy?"
MODEL = os.environ.get("RAG_MODEL", "claude-haiku-5-5")
CORPUS_TOKENS_FULL = 609 * 2046.5  # MultiHop-RAG paper, Table 2


def price(input_tokens, output_tokens):
    long = input_tokens > 100_000
    return input_tokens * (0.50 if long else 0.10) / 1e6 + output_tokens * (2.50 if long else 0.50) / 1e6


def run(client, user_content, system=SYSTEM):
    t0 = time.perf_counter()
    r = client.messages.create(model=MODEL, max_tokens=2000, system=system,
                               messages=[{"role": "user", "content": user_content}])
    latency = time.perf_counter() - t0
    text = "".join(b.text for b in r.content if b.type == "text")
    u = r.usage
    # cost: what the same token counts would cost on Claude Haiku 5.5 (local run itself is free;
    # token counts come from the local model's tokenizer, so treat cost as an estimate)
    return {"answer": text, "input_tokens": u.input_tokens, "output_tokens": u.output_tokens,
            "latency_s": round(latency, 2),
            "haiku_5_5_cost_usd_est": round(price(u.input_tokens, u.output_tokens), 6)}


def main():
    hits = retrieve(QUESTION)
    out = {"model": MODEL, "question": QUESTION, "n_paragraphs": len(docs),
           "top3": [{"score": round(s, 3), "source": d["source"], "title": d["title"], "text": d["text"]}
                    for d, s in hits]}
    if "--retrieval-only" not in sys.argv:
        import anthropic
        client = anthropic.Anthropic()
        closed = "Answer the question. If you are not sure, say so."
        out["hook_closed_book"] = run(client, INTERNAL_Q, system=closed)
        out["hook_with_doc"] = run(client, f"<documents>\n[1] {INTERNAL_DOC}\n</documents>\n\n{INTERNAL_Q}")
        out["a_closed_book"] = run(client, QUESTION, system=closed)
        def stuff(subset):
            full = "\n\n".join(f"[{i+1}] {a['title']} ({a['source']})\n{a['body']}" for i, a in enumerate(subset))
            return f"<documents>\n{full}\n</documents>\n\n{QUESTION}"
        try:
            out["b_stuff_20_articles"] = run(client, stuff(articles))
        except anthropic.APIStatusError as e:  # local model's context window is smaller than the prompt
            out["b_stuff_20_articles"] = {"error": str(e)[:400]}
        out["b_stuff_5_articles"] = run(client, stuff(articles[7:12]))  # 5 articles incl. the FTX one
        ctx = "\n\n".join(f"[{n}] {d['title']} ({d['source']})\n{d['text']}" for n, (d, _) in enumerate(hits, 1))
        out["c_rag_top3"] = run(client, f"<documents>\n{ctx}\n</documents>\n\n{QUESTION}")
        out["b_full_corpus_estimate"] = {
            "input_tokens": int(CORPUS_TOKENS_FULL),
            "note": "609 articles x 2,046.5 avg tokens; exceeds Haiku 5.5's 1M context, not run",
            "cost_usd_if_it_fit": round(price(CORPUS_TOKENS_FULL, 200), 4),
        }
    Path(__file__).with_name("results.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
