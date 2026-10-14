"""Part 3: why chunk ids come from content, not position.

Edit the first paragraph of every article (as if a publisher fixed a typo), re-chunk, and count how many
chunk ids survive. Compares content-hash ids with position ids ("<url>#<index>").
Usage: uv run python chunk_ids.py        (writes results/chunk_ids.json, no GPU needed)
"""
import json
from pathlib import Path

from ragkit.chunk import STRATEGIES, make_chunks
from ragkit.corpus import load_clean_articles

articles = load_clean_articles()
edited = [dict(a, body=a["body"].replace("\n\n", " (corrected)\n\n", 1)) for a in articles]  # one tiny edit each

rows = []
for name, strategy in STRATEGIES.items():
    before, after = make_chunks(articles, strategy), make_chunks(edited, strategy)
    kept_hash = len({c.id for c in before} & {c.id for c in after})
    pos = lambda cs: {(c.meta["url"], c.meta["chunk_index"]): c.raw_text for c in cs}
    pb, pa = pos(before), pos(after)
    shared = pb.keys() & pa.keys()  # position ids that still exist after the edit...
    changed = sum(pb[k] != pa[k] for k in shared)  # ...but now point at different text
    rows.append({"strategy": name, "chunks_before": len(before), "chunks_after": len(after),
                 "ids_unchanged_content_hash": kept_hash, "position_ids_still_exist": len(shared),
                 "position_ids_with_changed_text": changed,
                 "share_content_ids_kept": round(kept_hash / len(before), 3)})
    print(f"{name:30s} before {len(before):6d}  content ids kept {kept_hash:6d} ({kept_hash / len(before):.1%})  "
          f"position ids still exist {len(shared):6d}, {changed} of them now point at different text")
Path("results").mkdir(exist_ok=True)
Path("results/chunk_ids.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
