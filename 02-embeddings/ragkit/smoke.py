"""The smoke test: 10 questions, did the right article show up in the top 3?"""
import json
from pathlib import Path

import numpy as np

SMOKE_PATH = Path(__file__).resolve().parents[1] / "eval" / "smoke.jsonl"


def load_smoke(path=SMOKE_PATH) -> list[dict]:
    return [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]


def run_smoke(embedder, doc_vecs: np.ndarray, urls: list[str], questions: list[dict],
              field: str, k: int = 3) -> dict:
    """field is 'q_en' or 'q_vi'. A question is a hit when ANY expected article is in the top k."""
    from ragkit.search import top_k

    rows = []
    for q in questions:
        ranked = [urls[i] for i, _ in top_k(embedder.embed_query(q[field]), doc_vecs, len(urls))]
        got = ranked[:k]
        found = [u for u in q["expected_urls"] if u in got]
        # rank (1 = best) of the best-placed expected article among all articles
        rank = min(ranked.index(u) for u in q["expected_urls"]) + 1
        rows.append({"id": q["id"], "hit": bool(found), "all": len(found) == len(q["expected_urls"]),
                     "rank": rank, "top_urls": got})
    n = len(rows)
    return {"hit_at_k": sum(r["hit"] for r in rows) / n, "all_at_k": sum(r["all"] for r in rows) / n,
            "median_rank": float(np.median([r["rank"] for r in rows])),
            "mean_rank": float(np.mean([r["rank"] for r in rows])), "n": n, "k": k, "rows": rows}
