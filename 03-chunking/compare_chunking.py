"""Part 3: chunk the cleaned corpus five ways, embed every chunk, and score retrieval.

Usage: uv run python compare_chunking.py     (writes results/chunking.json; embeddings are cached)
No LLM needed. Questions: the 10-question smoke set from lesson 02 and 150 sampled questions.
A chunk hit counts when one of the top-k CHUNKS comes from an expected article, so one long
article cannot fill the top k with copies of itself for free.
"""
import gc
import json
import time
from datetime import date
from pathlib import Path

import numpy as np

from ragkit.chunk import STRATEGIES, Chunk, chunk_id, make_chunks
from ragkit.corpus import article_text, load_articles, load_clean_articles
from ragkit.embed import LocalEmbedder
from ragkit.smoke import load_sample, normalize, run_chunk_eval, smoke_with_facts

KS = (3, 5)
BUDGETS = (2000,)  # tokens of context handed to the LLM
raw_articles = load_articles()
articles = load_clean_articles()
sets = {"smoke10": smoke_with_facts(), "sample150": load_sample()}
facts = {(f["url"], normalize(f["fact"])) for qs in sets.values() for q in qs for f in q["facts"]}


def facts_intact(chunks: list[Chunk]) -> float:
    """Share of evidence sentences that sit whole inside one chunk (not cut by a boundary)."""
    by_url: dict[str, list[str]] = {}
    for c in chunks:
        by_url.setdefault(c.meta["url"], []).append(normalize(c.raw_text))
    return sum(any(f in t for t in by_url.get(u, [])) for u, f in facts) / len(facts)


def evaluate(label, chunks, embedder, **extra):
    t0 = time.perf_counter()
    vecs = embedder.embed_documents([c.text for c in chunks])
    embed_s = time.perf_counter() - t0
    tokens = np.array([c.meta["tokens"] for c in chunks])
    row = {"label": label, "chunks": len(chunks), "mean_tokens": round(float(tokens.mean()), 1),
           "median_tokens": float(np.median(tokens)), "max_tokens": int(tokens.max()),
           "under_32_tokens": int((tokens < 32).sum()), "facts_intact": round(facts_intact(chunks), 4),
           "unique_ids": len({c.id for c in chunks}), "embed_seconds_this_run": round(embed_s, 1),
           "float32_mb": round(len(chunks) * embedder.dim * 4 / 1e6, 1), **extra}
    for name, qs in sets.items():
        r = run_chunk_eval(embedder, chunks, vecs, qs, KS, BUDGETS)
        row[name] = {k: v for k, v in r.items() if k != "rows"}
        if name == "sample150":
            row["sample150_rows"] = r["rows"]
    s = row["sample150"]
    print(f"{label:46s} chunks {row['chunks']:6d}  tok {row['mean_tokens']:6.1f}  intact {row['facts_intact']:.3f}  "
          f"hit@3 {s['hit@3']:.3f}  all@5 {s['all@5']:.3f}  fact@3 {s['fact@3']:.3f}  fact@5 {s['fact@5']:.3f}  "
          f"fact@2000t {s['fact@2000t']:.3f}  rank {s['mean_rank']:5.1f}  | smoke hit@3 {row['smoke10']['hit@3']:.1f} ({row['embed_seconds_this_run']} s)", flush=True)
    return row


results = {"date": date.today().isoformat(), "model": "BAAI/bge-m3", "n_articles": len(articles),
           "k": list(KS), "questions": {k: len(v) for k, v in sets.items()}, "configs": []}

# Baseline from lesson 02: each article embedded whole, cut at 2048 tokens (now on cleaned text).
whole = [Chunk(chunk_id(a["url"], article_text(a)), article_text(a), article_text(a),
               {"url": a["url"], "tokens": 0}) for a in articles]
emb = LocalEmbedder("BAAI/bge-m3", max_seq_length=2048, batch_size=4)
from ragkit.chunk import count_tokens  # noqa: E402
for c, n in zip(whole, count_tokens([c.raw_text for c in whole])):
    c.meta["tokens"] = n
results["configs"].append(evaluate("whole article (2048-token cut)", whole, emb, strategy="whole", prepend=False,
                                   cleaned=True, truncated_articles=sum(c.meta["tokens"] > 2048 for c in whole)))
del emb
gc.collect()

emb = LocalEmbedder("BAAI/bge-m3", max_seq_length=1024, batch_size=16)  # room for a 512-token chunk + header
for name, strategy in STRATEGIES.items():
    for prepend in (False, True):
        label = f"{name}{' + header' if prepend else ''}"
        results["configs"].append(evaluate(label, make_chunks(articles, strategy, prepend), emb,
                                           strategy=name, prepend=prepend, cleaned=True))

# Does cleaning matter? Same strategy and size, on the raw text.
raw_uncleaned = [dict(a, date=a["published_at"][:10]) for a in raw_articles]
results["configs"].append(evaluate("recursive 512 + header, NOT cleaned", make_chunks(raw_uncleaned, STRATEGIES["recursive 512"], True),
                                   emb, strategy="recursive 512", prepend=True, cleaned=False))

out = Path("results/chunking.json")
if out.exists():  # a cached rerun reports ~0 s; keep the first-run timings
    old = {c["label"]: c for c in json.loads(out.read_text(encoding="utf-8"))["configs"]}
    for c in results["configs"]:
        c["embed_seconds_first_run"] = old.get(c["label"], {}).get("embed_seconds_first_run", c["embed_seconds_this_run"])
else:
    for c in results["configs"]:
        c["embed_seconds_first_run"] = c["embed_seconds_this_run"]
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(results, indent=1, ensure_ascii=False), encoding="utf-8")
