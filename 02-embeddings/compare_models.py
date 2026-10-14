"""Part 2: embed every article with each model config, then run the 10-question smoke test.

Usage: uv run python compare_models.py       (writes results/compare.json; embeddings are cached)
Each article is embedded whole (title + body), cut at the model's max tokens. Chunking is lesson 03.
"""
import json
import time
from datetime import date
from pathlib import Path

import numpy as np

from ragkit.corpus import article_text, load_articles
from ragkit.embed import LocalEmbedder
from ragkit.smoke import load_smoke, run_smoke

CONFIGS = [
    # name shown in the table, constructor kwargs
    ("bge-m3, 2048 tokens", dict(model_name="BAAI/bge-m3", max_seq_length=2048, batch_size=4)),
    ("bge-m3, 512 tokens", dict(model_name="BAAI/bge-m3", max_seq_length=512, batch_size=16)),
    ("multilingual-e5-small, with prefixes", dict(model_name="intfloat/multilingual-e5-small", max_seq_length=512,
                                                  query_prefix="query: ", doc_prefix="passage: ", batch_size=16)),
    ("multilingual-e5-small, no prefixes", dict(model_name="intfloat/multilingual-e5-small", max_seq_length=512,
                                                batch_size=16)),
]

articles = load_articles()
texts = [article_text(a) for a in articles]
urls = [a["url"] for a in articles]
smoke = load_smoke()
results = {"date": date.today().isoformat(), "n_articles": len(articles), "n_questions": len(smoke), "k": 3,
           "configs": []}

for label, kw in CONFIGS:
    emb = LocalEmbedder(**kw)
    t0 = time.perf_counter()
    doc_vecs = emb.embed_documents(texts)
    embed_s = time.perf_counter() - t0  # ~0 on a cache hit; see embed_seconds_first_run in the committed file
    tok = emb.model.tokenizer
    n_tokens = [len(tok(t, add_special_tokens=False)["input_ids"]) for t in texts]
    max_len = emb.model.max_seq_length
    row = {"label": label, "model": emb.name, "dim": emb.dim, "max_seq_length": max_len,
           "articles_truncated": int(sum(n > max_len for n in n_tokens)),
           "mb_per_1k_vectors_float32": round(emb.dim * 4 * 1000 / 1e6, 2),
           "embed_seconds_this_run": round(embed_s, 1)}
    for lang in ("q_en", "q_vi"):
        r = run_smoke(emb, doc_vecs, urls, smoke, lang)
        row[lang] = {"hit_at_3": r["hit_at_k"], "both_at_3": r["all_at_k"],
                     "mean_rank": round(r["mean_rank"], 1), "median_rank": r["median_rank"],
                     "ranks": [x["rank"] for x in r["rows"]], "missed": [x["id"] for x in r["rows"] if not x["hit"]]}
    results["configs"].append(row)
    print(f"{label:40s} dim {row['dim']:4d}  truncated {row['articles_truncated']:3d}/{len(articles)}  "
          f"EN hit@3 {row['q_en']['hit_at_3']:.1f}  VI hit@3 {row['q_vi']['hit_at_3']:.1f}  "
          f"both@3 {row['q_en']['both_at_3']:.1f}/{row['q_vi']['both_at_3']:.1f}  "
          f"mean rank {row['q_en']['mean_rank']}/{row['q_vi']['mean_rank']}  "
          f"({row['embed_seconds_this_run']} s)")
    del emb

out = Path("results/compare.json")
if out.exists():  # keep first-run timings: a cached run reports ~0 s
    old = {c["label"]: c for c in json.loads(out.read_text(encoding="utf-8"))["configs"]}
    for c in results["configs"]:
        c["embed_seconds_first_run"] = old.get(c["label"], {}).get("embed_seconds_first_run", c["embed_seconds_this_run"])
else:
    for c in results["configs"]:
        c["embed_seconds_first_run"] = c["embed_seconds_this_run"]
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
