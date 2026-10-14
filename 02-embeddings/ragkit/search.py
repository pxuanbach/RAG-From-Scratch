import numpy as np


def top_k(query_vec: np.ndarray, doc_vecs: np.ndarray, k: int = 3) -> list[tuple[int, float]]:
    """Brute-force search. With normalized vectors the dot product is the cosine similarity."""
    scores = doc_vecs @ query_vec
    idx = np.argsort(-scores)[:k]
    return [(int(i), float(scores[i])) for i in idx]
