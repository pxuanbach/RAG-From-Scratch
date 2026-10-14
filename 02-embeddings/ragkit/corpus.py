"""Load MultiHop-RAG (downloaded from Hugging Face on first use)."""
import json

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
