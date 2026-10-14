"""Part 3: are the differences in results/chunking.json bigger than noise?

With 150 questions a hit rate near 0.85 has a standard error of about 0.03, so two strategies 0.02 apart are tied.
This script resamples the questions (paired bootstrap, 2,000 draws) and prints a 95% interval for each difference.
Usage: uv run python analyze.py      (reads results/chunking.json, writes results/significance.json)
"""
import json
from pathlib import Path

import numpy as np

data = json.loads(Path("results/chunking.json").read_text(encoding="utf-8"))
rows = {c["label"]: {r["id"]: r for r in c["sample150_rows"]} for c in data["configs"]}
ids = sorted(next(iter(rows.values())))
rng = np.random.default_rng(0)
draws = rng.integers(0, len(ids), size=(2000, len(ids)))


def diff(a_labels, b_labels, metric):
    """Mean of metric over a minus b, averaged over paired labels; returns (mean, lo, hi)."""
    per_q = np.array([np.mean([float(rows[a][i][metric]) - float(rows[b][i][metric]) for a, b in zip(a_labels, b_labels)])
                      for i in ids])
    boots = per_q[draws].mean(axis=1)
    return round(float(per_q.mean()), 4), round(float(np.percentile(boots, 2.5)), 4), round(float(np.percentile(boots, 97.5)), 4)


strategies = ["fixed 256", "fixed 512 + overlap 64", "recursive 256", "recursive 512", "paragraph (min 128, max 512)"]
with_h = [f"{s} + header" for s in strategies]
out = {}
R512, PARA, WHOLE = "recursive 512", "paragraph (min 128, max 512)", "whole article (2048-token cut)"
tests = {
    "header vs no header (pooled over 5 strategies)": (with_h, strategies),
    "cleaned vs raw text (recursive 512 + header)": (["recursive 512 + header"], ["recursive 512 + header, NOT cleaned"]),
    "recursive 256 vs fixed 256 (same size, different cut points)": (["recursive 256"], ["fixed 256"]),
    "recursive 512 vs fixed 512": ([R512], ["fixed 512"]),
    "paragraph vs recursive 512": ([PARA], [R512]),
    "paragraph vs fixed 256": ([PARA], ["fixed 256"]),
    "recursive 512 vs whole article": ([R512], [WHOLE]),
    "paragraph vs whole article": ([PARA], [WHOLE]),
}
for name, (a, b) in tests.items():
    out[name] = {m: diff(a, b, m) for m in ("hit@3", "all@5", "fact@5", "fact@2000t")}
    print(name)
    for m, (mean, lo, hi) in out[name].items():
        flag = "" if lo <= 0 <= hi else "   <- outside noise"
        print(f"   {m:11s} {mean:+.3f}  [{lo:+.3f}, {hi:+.3f}]{flag}")
Path("results/significance.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
