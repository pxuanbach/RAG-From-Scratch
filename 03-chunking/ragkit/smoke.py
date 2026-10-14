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


# ---------------------------------------------------------------- chunk-level evaluation (lesson 03)
import re  # noqa: E402

from ragkit.corpus import INVISIBLE, load_questions  # noqa: E402

SAMPLE_PATH = Path(__file__).resolve().parents[1] / "eval" / "sample150.jsonl"


def load_sample(path=SAMPLE_PATH) -> list[dict]:
    """150 questions (50 per type) with the evidence sentences. Built by eval/build_sample.py."""
    return [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]


def smoke_with_facts() -> list[dict]:
    """The 10 smoke questions from lesson 02, now carrying their evidence sentences too."""
    all_q = load_questions()
    return [dict(q, facts=[{"url": e["url"], "fact": e["fact"]} for e in all_q[q["src_index"]]["evidence_list"]])
            for q in load_smoke()]


def normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s.translate(INVISIBLE)).strip()


def run_chunk_eval(embedder, chunks, chunk_vecs: np.ndarray, questions: list[dict], ks=(3, 5),
                   budgets=(2000,)) -> dict:
    """Rank chunks for each question (English text in q_en) and score the top chunks.

    hit    some chunk in the top comes from an expected article
    all    every expected article has a chunk in the top
    fact   share of evidence sentences that appear verbatim in a top chunk of their article
    rank   position of the first chunk from an expected article (1 = best)
    The top is either the best k chunks (hit@3 ...) or as many chunks as fit in a token budget
    (hit@2000t ...), which is how a prompt is limited in practice.
    """
    urls = [c.meta["url"] for c in chunks]
    texts = [normalize(c.raw_text) for c in chunks]
    sizes = np.array([c.meta["tokens"] for c in chunks])
    qvecs = np.stack([embedder.embed_query(q["q_en"]) for q in questions])
    scores = qvecs @ chunk_vecs.T

    def score(top, q):
        got, expected = {urls[i] for i in top}, set(q["expected_urls"])
        found = sum(any(urls[i] == f["url"] and normalize(f["fact"]) in texts[i] for i in top) for f in q["facts"])
        return {"hit": bool(got & expected), "all": expected <= got, "fact": found / len(q["facts"]),
                "ctx_tokens": int(sizes[top].sum()), "n_chunks": len(top)}

    rows = []
    for q, sc in zip(questions, scores):
        order = np.argsort(-sc)
        expected = set(q["expected_urls"])
        row = {"id": q["id"], "type": q.get("type"), "rank": next(r for r, i in enumerate(order, 1) if urls[i] in expected)}
        for k in ks:
            row.update({f"{m}@{k}": v for m, v in score(order[:k], q).items()})
        cum = np.cumsum(sizes[order[:200]])
        for b in budgets:  # always keep the best chunk, even if it alone is over budget
            row.update({f"{m}@{b}t": v for m, v in score(order[:max(1, int((cum <= b).sum()))], q).items()})
        rows.append(row)
    out = {"n": len(rows)}
    for key in rows[0]:
        if "@" in key:
            out[key] = round(float(np.mean([r[key] for r in rows])), 4)
    out["mean_rank"] = round(float(np.mean([r["rank"] for r in rows])), 2)
    out["median_rank"] = float(np.median([r["rank"] for r in rows]))
    k = max(ks)
    out["by_type"] = {t: {m: round(float(np.mean([r[f"{m}@{k}"] for r in rows if r["type"] == t])), 3)
                          for m in ("hit", "all", "fact")} for t in sorted({r["type"] for r in rows if r["type"]})}
    out["rows"] = rows
    return out
