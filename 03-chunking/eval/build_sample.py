"""Pick a bigger question set for comparing chunking strategies: 50 per question type, 150 total.

The 10-question smoke set from lesson 02 is too easy to rank strategies (bge-m3 scored 10/10).
Retrieval metrics need no LLM, so 150 questions cost nothing. Fixed seed, smoke questions excluded.
Run from the lesson folder: uv run python eval/build_sample.py
"""
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ragkit.corpus import load_questions
from ragkit.smoke import load_smoke

smoke_ids = {q["src_index"] for q in load_smoke()}
rng = random.Random(0)
questions = load_questions()
out = []
for qtype in ("inference_query", "comparison_query", "temporal_query"):  # null_query has no evidence
    pool = [i for i, x in enumerate(questions) if x["question_type"] == qtype and i not in smoke_ids]
    for i in sorted(rng.sample(pool, 50)):
        x = questions[i]
        out.append({"id": f"q{len(out) + 1:03d}", "type": qtype, "src_index": i, "q_en": x["query"],
                    "expected_urls": [e["url"] for e in x["evidence_list"]],
                    "facts": [{"url": e["url"], "fact": e["fact"]} for e in x["evidence_list"]]})
path = Path(__file__).with_name("sample150.jsonl")
path.write_text("\n".join(json.dumps(o, ensure_ascii=False) for o in out) + "\n", encoding="utf-8")
print(f"wrote {len(out)} questions to {path}")
