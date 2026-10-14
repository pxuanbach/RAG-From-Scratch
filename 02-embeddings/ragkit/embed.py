"""The Embedder interface plus a local implementation and an API one.

Queries and documents go through separate methods because some models want
different prefixes for each (E5 wants "query: " and "passage: "). Keeping the
two apart now means a later lesson can swap models without touching callers.
"""
import hashlib
import os
from pathlib import Path
from typing import Protocol

import numpy as np

CACHE_DIR = Path(os.environ.get("RAG_CACHE", Path(__file__).resolve().parents[2] / ".cache"))


class Embedder(Protocol):
    name: str
    dim: int

    def embed_documents(self, texts: list[str]) -> np.ndarray: ...
    def embed_query(self, text: str) -> np.ndarray: ...


class DiskCache:
    """Maps sha256(text) -> vector in one .npz file per embedder config."""

    def __init__(self, tag: str):
        self.path = CACHE_DIR / "embeddings" / f"{tag}.npz"
        self.vecs: dict[str, np.ndarray] = {}
        if self.path.exists():
            with np.load(self.path) as f:
                self.vecs = {k: f[k] for k in f.files}

    @staticmethod
    def key(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(self.path, **self.vecs)


def _embed_with_cache(cache: DiskCache | None, texts, encode):
    if cache is None:
        return encode(texts)
    keys = [cache.key(t) for t in texts]
    missing = [i for i, k in enumerate(keys) if k not in cache.vecs]
    if missing:
        new = encode([texts[i] for i in missing])
        for i, v in zip(missing, new):
            cache.vecs[keys[i]] = v
        cache.save()
    return np.stack([cache.vecs[k] for k in keys])


class LocalEmbedder:
    """Any sentence-transformers model. Vectors come back L2-normalized."""

    def __init__(self, model_name: str, *, query_prefix: str = "", doc_prefix: str = "",
                 max_seq_length: int | None = None, batch_size: int = 8, use_cache: bool = True):
        import torch
        from sentence_transformers import SentenceTransformer

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name, device=device)
        if max_seq_length:
            self.model.max_seq_length = max_seq_length  # longer texts are cut off silently
        self.name = model_name
        get_dim = getattr(self.model, "get_embedding_dimension", None) or self.model.get_sentence_embedding_dimension
        self.dim = get_dim()
        self.query_prefix, self.doc_prefix, self.batch_size = query_prefix, doc_prefix, batch_size
        tag = f"{model_name.replace('/', '__')}_len{self.model.max_seq_length}_{len(doc_prefix)}p"
        self.cache = DiskCache(tag) if use_cache else None

    def _encode(self, texts):
        return self.model.encode(texts, batch_size=self.batch_size, normalize_embeddings=True,
                                 show_progress_bar=len(texts) > 50)

    def embed_documents(self, texts):
        return _embed_with_cache(self.cache, [self.doc_prefix + t for t in texts], self._encode)

    def embed_query(self, text):
        return self._encode([self.query_prefix + text])[0]  # queries are short, never cached


class OpenAIEmbedder:
    """API fallback for machines that cannot run a local model.

    UNTESTED: written against the OpenAI embeddings API but never run, because
    I had no key. Needs `pip install openai` and OPENAI_API_KEY.
    """

    def __init__(self, model_name: str = "text-embedding-3-small", dim: int = 1536):
        from openai import OpenAI

        self.client, self.name, self.dim = OpenAI(), model_name, dim
        self.cache = DiskCache(f"openai__{model_name}_{dim}")

    def _encode(self, texts):
        out = []
        for i in range(0, len(texts), 256):
            r = self.client.embeddings.create(model=self.name, input=texts[i:i + 256], dimensions=self.dim)
            out += [d.embedding for d in r.data]
        v = np.array(out, dtype=np.float32)
        return v / np.linalg.norm(v, axis=1, keepdims=True)  # normalize so dot product = cosine

    def embed_documents(self, texts):
        return _embed_with_cache(self.cache, texts, self._encode)

    def embed_query(self, text):
        return self._encode([text])[0]
