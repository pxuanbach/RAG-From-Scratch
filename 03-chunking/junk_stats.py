"""Part 3: count the junk in all 609 articles, clean them, and check the cleaning did no harm.

Usage: uv run python junk_stats.py      (writes results/junk_stats.json, no GPU or LLM needed)
"""
import json
import re
from collections import Counter
from datetime import date
from pathlib import Path

from transformers import AutoTokenizer

from ragkit.corpus import (INVISIBLE, JUNK_PARAGRAPH_RULES, clean_article, find_boilerplate, junk_reason,
                           load_articles, load_questions)

raw = load_articles()
boilerplate = find_boilerplate(raw)
clean = [clean_article(a, boilerplate) for a in raw]
tok = AutoTokenizer.from_pretrained("BAAI/bge-m3")
n_tokens = lambda texts: sum(len(x) for x in tok(texts, add_special_tokens=False)["input_ids"])


def arts_where(pred):
    return sum(1 for a in raw if pred(a))


reasons_par, reasons_art = Counter(), {}
for a in raw:
    seen = set()
    for p in a["body"].translate(INVISIBLE).split("\n\n"):
        r = junk_reason(p.strip(), boilerplate) if p.strip() else None
        if r:
            reasons_par[r] += 1
            seen.add(r)
    for r in seen:
        reasons_art[r] = reasons_art.get(r, 0) + 1

paragraphs_total = sum(len(a["body"].split("\n\n")) for a in raw)
paragraphs_kept = sum(len(a["body"].split("\n\n")) if a["body"] else 0 for a in clean)
tokens_before, tokens_after = n_tokens([a["body"] for a in raw]), n_tokens([a["body"] for a in clean])

ws = lambda s: re.sub(r"\s+", " ", s.translate(INVISIBLE)).strip()  # a few facts contain a BOM too
by_url = {a["url"]: ws(a["body"]) for a in clean}
facts = [(e["url"], ws(e["fact"])) for q in load_questions() for e in q["evidence_list"]]
raw_by_url = {a["url"]: ws(a["body"]) for a in raw}
facts_raw_ok = sum(f in raw_by_url[u] for u, f in facts)
facts_clean_ok = sum(f in by_url[u] for u, f in facts)

out = {
    "date": date.today().isoformat(), "n_articles": len(raw),
    "author_null": arts_where(lambda a: a["author"] is None),
    "author_empty_string": arts_where(lambda a: a["author"] == ""),
    "author_string_None": arts_where(lambda a: a["author"] == "None"),
    "replacement_char_U+FFFD": arts_where(lambda a: "\ufffd" in a["body"]),
    "mojibake_Ã_or_â€": arts_where(lambda a: re.search("Ã|â€", a["body"]) is not None),
    "articles_with_invisible_chars": arts_where(lambda a: any(ord(c) in INVISIBLE for c in a["body"])),
    "invisible_char_count": sum(1 for a in raw for c in a["body"] if ord(c) in INVISIBLE),
    "articles_with_empty_paragraphs_(3+ newlines)": arts_where(lambda a: "\n\n\n" in a["body"]),
    "junk_paragraphs_by_rule": dict(reasons_par),
    "articles_with_junk_by_rule": reasons_art,
    "articles_with_any_junk_paragraph": sum(
        1 for a in raw if any(junk_reason(p.strip(), boilerplate) for p in a["body"].split("\n\n") if p.strip())),
    "repeated_boilerplate_distinct_paragraphs": len(boilerplate),
    "repeated_boilerplate_top": [p[:90] for p, _ in sorted(
        ((p, sum(p in {x.strip() for x in a["body"].split("\n\n")} for a in raw)) for p in boilerplate),
        key=lambda t: -t[1])[:12]],
    "paragraphs_total": paragraphs_total, "paragraphs_after_cleaning": paragraphs_kept,
    "tokens_before": tokens_before, "tokens_after": tokens_after,
    "articles_emptied": sum(1 for a in clean if not a["body"]),
    "evidence_facts_total": len(facts),
    "evidence_facts_found_in_raw_body": facts_raw_ok,
    "evidence_facts_found_in_clean_body": facts_clean_ok,
}
Path("results").mkdir(exist_ok=True)
Path("results/junk_stats.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
for k, v in out.items():
    print(f"{k}: {v}")
