import json
import os
import sys

import anthropic
import numpy as np
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download
from sentence_transformers import SentenceTransformer

load_dotenv()  # reads ANTHROPIC_* and RAG_MODEL from .env in the repo root

# 1. Load 20 news articles and keep the first 3 real paragraphs of each
path = hf_hub_download("yixuantt/MultiHopRAG", "corpus.json", repo_type="dataset")
articles = json.load(open(path, encoding="utf-8"))[:20]
docs = []
for a in articles:
    paragraphs = [p.strip() for p in a["body"].split("\n\n") if len(p.strip()) > 80]
    for p in paragraphs[:3]:
        docs.append({"title": a["title"], "source": a["source"], "text": p})

# 2. Embed every paragraph once (normalized, so dot product = cosine similarity)
model = SentenceTransformer("BAAI/bge-m3")
doc_vecs = model.encode([d["text"] for d in docs], normalize_embeddings=True)


def retrieve(question, k=3):
    q_vec = model.encode([question], normalize_embeddings=True)[0]
    scores = doc_vecs @ q_vec
    top = np.argsort(-scores)[:k]
    return [(docs[i], float(scores[i])) for i in top]


SYSTEM = (
    "Answer using only the documents provided. Cite documents as [1], [2]. "
    "If the documents do not contain the answer, say you don't know."
)


def ask(question):
    hits = retrieve(question)
    context = "\n\n".join(
        f"[{n}] {d['title']} ({d['source']})\n{d['text']}" for n, (d, _) in enumerate(hits, 1)
    )
    # Reads ANTHROPIC_API_KEY / ANTHROPIC_BASE_URL, so the same code runs on Claude or LM Studio
    response = anthropic.Anthropic().messages.create(
        model=os.environ.get("RAG_MODEL", "claude-haiku-5-5"),
        max_tokens=2000,
        system=SYSTEM,
        messages=[{"role": "user", "content": f"<documents>\n{context}\n</documents>\n\n{question}"}],
    )
    return "".join(b.text for b in response.content if b.type == "text"), hits


if __name__ == "__main__":
    answer, hits = ask(" ".join(sys.argv[1:]) or "When was FTX founded and what was it worth at its peak?")
    for d, score in hits:
        print(f"{score:.3f}  {d['source']}: {d['title']}")
    print("\n" + answer)
