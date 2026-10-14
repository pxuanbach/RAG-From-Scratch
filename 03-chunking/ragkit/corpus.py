"""Load MultiHop-RAG (downloaded from Hugging Face on first use)."""
import json
import re
from collections import Counter

from huggingface_hub import hf_hub_download


def _load(filename: str):
    path = hf_hub_download("yixuantt/MultiHopRAG", filename, repo_type="dataset")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_articles() -> list[dict]:
    """609 articles: title, body, source, url, author, category, published_at."""
    return _load("corpus.json")


def load_questions() -> list[dict]:
    """2,556 questions: query, answer, question_type, evidence_list[].url ..."""
    return _load("MultiHopRAG.json")


def article_text(article: dict) -> str:
    """The text we embed for a whole article: title, then body."""
    return f"{article['title']}\n\n{article['body']}"


# ---------------------------------------------------------------- cleaning (lesson 03)
# Counts below come from scanning all 609 articles with junk_stats.py.
INVISIBLE = dict.fromkeys(map(ord, "\ufeff\u200b\u200c\u200d\u00ad\u2060"))  # BOM, zero-width, soft hyphen

# Paragraphs that match any of these are dropped whole.
JUNK_PARAGRAPH_RULES = {
    "table_of_contents": re.compile(r"^Table of Contents\b"),
    "newsletter_template": re.compile(r"\{\{|^skip past newsletter promotion", re.I),
    "update_note": re.compile(r"^UPDATE: [A-Z][a-z]{2,4}\.? \d{1,2}, \d{4}"),  # "UPDATE: Nov. 27, 2023, 5:00 a.m. EST ..."
    "live_blog_stamp": re.compile(r"^Updated at \d"),
}


def find_boilerplate(articles: list[dict], min_articles: int = 5) -> set[str]:
    """Paragraphs that appear verbatim in at least `min_articles` different articles.

    Real text is almost never repeated word for word across articles, so a repeated
    paragraph ('Advertisement', 'CLICK HERE TO SIGN UP ...') is page furniture.
    """
    seen: Counter = Counter()
    for a in articles:
        seen.update({p.strip() for p in a["body"].translate(INVISIBLE).split("\n\n") if p.strip()})
    return {p for p, n in seen.items() if n >= min_articles}


def junk_reason(paragraph: str, boilerplate: set[str]) -> str | None:
    for name, rule in JUNK_PARAGRAPH_RULES.items():
        if rule.search(paragraph):
            return name
    return "repeated_boilerplate" if paragraph in boilerplate else None


def clean_body(body: str, boilerplate: set[str]) -> str:
    """Drop junk paragraphs, remove invisible characters, keep the \n\n paragraph breaks."""
    paragraphs = (p.strip() for p in body.translate(INVISIBLE).split("\n\n"))
    return "\n\n".join(p for p in paragraphs if p and not junk_reason(p, boilerplate))


def clean_article(a: dict, boilerplate: set[str]) -> dict:
    """Same fields as the raw article, plus `date`. An empty or missing author becomes None."""
    out = dict(a, body=clean_body(a["body"], boilerplate))
    out["author"] = (a.get("author") or "").strip() or None
    out["date"] = a["published_at"][:10]
    return out


def load_clean_articles() -> list[dict]:
    raw = load_articles()
    boilerplate = find_boilerplate(raw)
    return [clean_article(a, boilerplate) for a in raw]
