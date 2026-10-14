"""Part 1: what do embeddings look like, and how do cosine, dot product and L2 relate?

Usage: uv run python intuition.py        (writes results/intuition.json)
"""
import json
from pathlib import Path

import numpy as np

from ragkit.embed import LocalEmbedder

SENTENCES = {
    "A": "In January, Bankman-Fried pleaded not guilty to all counts, which include wire fraud, "
         "conspiracy to commit money laundering and conspiracy to misuse customer funds.",  # TechCrunch
    "B": "Sam Bankman-Fried, the founder of failed cryptocurrency exchange FTX, is on trial for "
         "seven counts of wire fraud and conspiracy.",  # The Verge
    "C": "The Raiders were abysmal on the offensive side of the ball, outside of a second-quarter "
         "touchdown drive led by Josh Jacobs.",  # Sporting News
    "D": "Hồi tháng 1, Bankman-Fried đã không nhận tội với mọi cáo buộc, gồm gian lận chuyển tiền "
         "điện tử, âm mưu rửa tiền và âm mưu lạm dụng tiền của khách hàng.",  # my translation of A
}

emb = LocalEmbedder("BAAI/bge-m3", use_cache=False)
ids = list(SENTENCES)
v = emb.embed_documents(list(SENTENCES.values()))
print(f"dimension: {emb.dim}, vector norm: {np.linalg.norm(v[0]):.3f}\n")

print("cosine similarity (A/B: same trial, different words; C: football; D: Vietnamese version of A)")
cos = v @ v.T
for i in range(len(ids)):
    for j in range(i + 1, len(ids)):
        print(f"  {ids[i]} vs {ids[j]}: {cos[i, j]:.3f}")

# Dot product, cosine and L2 on normalized vectors give the same ranking.
a, b = v[0], v[2]
l2 = float(np.linalg.norm(a - b))
print(f"\nnormalized vectors, A vs C: cosine {a @ b:.3f}, dot {np.dot(a, b):.3f}, "
      f"L2 {l2:.3f}, 2 - 2*cosine = {2 - 2 * (a @ b):.3f} = L2 squared {l2 ** 2:.3f}")

# Without normalizing, dot product depends on vector length, cosine does not.
big = 3 * a
cos_big = float(big @ b / (np.linalg.norm(big) * np.linalg.norm(b)))
print(f"scale A by 3: dot {big @ b:.3f} (changed), cosine {cos_big:.3f} (same)")

Path("results").mkdir(exist_ok=True)
Path("results/intuition.json").write_text(json.dumps({
    "model": "BAAI/bge-m3", "dim": emb.dim, "sentences": SENTENCES,
    "cosine": {f"{ids[i]}-{ids[j]}": round(float(cos[i, j]), 3) for i in range(4) for j in range(i + 1, 4)},
    "A_vs_C": {"cosine": round(float(a @ b), 3), "l2": round(l2, 3), "l2_squared": round(l2 ** 2, 3)},
    "scaled_dot": round(float(big @ b), 3),
}, indent=2, ensure_ascii=False), encoding="utf-8")
