"""Part 3: does skipping the cleaning step let boilerplate into the retrieved chunks?

Chunks the RAW articles (recursive 512 + header), searches with the 150 sample questions, and counts how many
chunks contain a junk paragraph, in the whole index and in the top 5. Reuses cached embeddings from compare_chunking.py.
Usage: uv run python junk_in_topk.py     (writes results/junk_in_topk.json)
"""
import json
from pathlib import Path

import numpy as np

from ragkit.chunk import STRATEGIES, make_chunks
from ragkit.corpus import find_boilerplate, junk_reason, load_articles
from ragkit.embed import LocalEmbedder
from ragkit.smoke import load_sample

raw = load_articles()
bp = find_boilerplate(raw)
chunks = make_chunks([dict(a, date=a["published_at"][:10]) for a in raw], STRATEGIES["recursive 512"], prepend=True)
has_junk = np.array([any(junk_reason(p.strip(), bp) for p in c.raw_text.split("\n\n") if p.strip()) for c in chunks])

emb = LocalEmbedder("BAAI/bge-m3", max_seq_length=1024, batch_size=16)
vecs = emb.embed_documents([c.text for c in chunks])
qs = load_sample()
top5 = [np.argsort(-(vecs @ emb.embed_query(q["q_en"])))[:5] for q in qs]
hits = [int(has_junk[t].sum()) for t in top5]
out = {"strategy": "recursive 512 + header, raw text", "chunks": len(chunks), "chunks_with_junk": int(has_junk.sum()),
       "share_of_index": round(float(has_junk.mean()), 3), "retrieved_chunks": 5 * len(qs),
       "retrieved_with_junk": int(sum(hits)), "share_of_retrieved": round(sum(hits) / (5 * len(qs)), 3),
       "questions_with_a_junk_chunk_in_top5": int(sum(h > 0 for h in hits))}
Path("results/junk_in_topk.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
print(json.dumps(out, indent=1))
