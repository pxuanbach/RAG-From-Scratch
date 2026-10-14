"""Turn articles into chunks: four strategies, all sized in tokens of the embedding model.

A Chunk keeps two texts. `text` is what gets embedded (optionally with a title | source | date
header in front). `raw_text` is what you show the user and put in a prompt as a citation.
"""
import hashlib
import re
from dataclasses import dataclass, field
from functools import lru_cache

from transformers import AutoTokenizer

MODEL = "BAAI/bge-m3"


@dataclass
class Chunk:
    id: str          # hash of url + raw_text: edit the text and the id changes, move it and it does not
    text: str        # embedded text (with the metadata header if prepend=True)
    raw_text: str    # original passage, for display and citations
    meta: dict = field(default_factory=dict)  # url, title, source, category, published_at, chunk_index, tokens


@lru_cache(maxsize=1)
def tokenizer():
    return AutoTokenizer.from_pretrained(MODEL, model_max_length=10**6)  # silence the 8192 warning


def count_tokens(texts: list[str]) -> list[int]:
    return [len(ids) for ids in tokenizer()(texts, add_special_tokens=False)["input_ids"]]


# ---------------------------------------------------------------- the four strategies
# Each takes an article body and returns a list of passages (strings).

def fixed(text: str, size: int = 256, overlap: int = 0) -> list[str]:
    """Cut every `size` tokens, stepping `size - overlap`. Ignores sentences and paragraphs."""
    enc = tokenizer()(text, add_special_tokens=False, return_offsets_mapping=True)
    offsets, n = enc["offset_mapping"], len(enc["input_ids"])
    out, start = [], 0
    while start < n:
        end = min(start + size, n)
        out.append(text[offsets[start][0]:offsets[end - 1][1]])
        if end == n:
            break
        start += size - overlap
    return out


PARAGRAPH, LINE, SENTENCE = re.compile(r"\n\n"), re.compile(r"\n"), re.compile(r"(?<=[.!?])\s+")
JOIN = {PARAGRAPH: "\n\n", LINE: "\n", SENTENCE: " "}


def _atoms(text: str, max_tokens: int, levels: tuple) -> list[tuple[str, str]]:
    """Split on the coarsest separator, and only split a piece further if it is still too long."""
    if count_tokens([text])[0] <= max_tokens:
        return [(text, "")]
    if not levels:  # one enormous sentence: last resort, cut by tokens
        return [(p, " ") for p in fixed(text, max_tokens)]
    level, rest = levels[0], levels[1:]
    out = []
    for piece in filter(str.strip, level.split(text)):
        pieces = _atoms(piece.strip(), max_tokens, rest)
        out += [(t, JOIN[level] if i == 0 else j) for i, (t, j) in enumerate(pieces)]
    return out


def _pack(atoms: list[tuple[str, str]], max_tokens: int, min_tokens: int = 0) -> list[str]:
    """Glue neighbouring atoms together until the next one would push past max_tokens.

    With min_tokens > 0 a chunk is closed as soon as it reaches that size, so long paragraphs
    stay alone and only short ones get merged.
    """
    sizes = count_tokens([t for t, _ in atoms])
    chunks, cur, cur_size = [], "", 0
    for (text, join), size in zip(atoms, sizes):
        if cur and (cur_size + size > max_tokens or cur_size >= min_tokens > 0):
            chunks.append(cur)
            cur, cur_size = "", 0
        cur = f"{cur}{join}{text}" if cur else text
        cur_size += size
    return chunks + [cur] if cur else chunks


def recursive(text: str, max_tokens: int = 512) -> list[str]:
    """Paragraphs first, then lines, then sentences; pack the pieces up to max_tokens."""
    return _pack(_atoms(text, max_tokens, (PARAGRAPH, LINE, SENTENCE)), max_tokens)


def by_paragraph(text: str, max_tokens: int = 512, min_tokens: int = 128) -> list[str]:
    """One paragraph per chunk. Short paragraphs merge forward, over-long ones split at sentences."""
    return _pack(_atoms(text, max_tokens, (PARAGRAPH, SENTENCE)), max_tokens, min_tokens)


DEFAULT = "paragraph (min 128, max 512)"  # the strategy later lessons use

STRATEGIES = {
    "fixed 256": lambda t: fixed(t, 256),
    "fixed 512": lambda t: fixed(t, 512),
    "fixed 512 + overlap 64": lambda t: fixed(t, 512, 64),
    "recursive 256": lambda t: recursive(t, 256),
    "recursive 512": lambda t: recursive(t, 512),
    "paragraph (min 128, max 512)": lambda t: by_paragraph(t, 512, 128),
}


# ---------------------------------------------------------------- articles -> chunks
def chunk_id(url: str, raw_text: str) -> str:
    return hashlib.sha1(f"{url}\n{raw_text}".encode("utf-8")).hexdigest()[:16]


def header(a: dict) -> str:
    return f"{a['title']} | {a['source']} | {a.get('date') or a['published_at'][:10]}"


def make_chunks(articles: list[dict], strategy, prepend: bool = False) -> list[Chunk]:
    chunks = []
    for a in articles:
        passages = strategy(a["body"])
        sizes = count_tokens(passages)
        for i, (raw, n) in enumerate(zip(passages, sizes)):
            meta = {"url": a["url"], "title": a["title"], "source": a["source"], "category": a["category"],
                    "published_at": a["published_at"], "chunk_index": i, "tokens": n}
            chunks.append(Chunk(chunk_id(a["url"], raw), f"{header(a)}\n\n{raw}" if prepend else raw, raw, meta))
    return chunks
