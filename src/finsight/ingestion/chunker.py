"""Section-aware, token-budgeted chunking with overlap and contextual headers.

* paragraphs are packed greedily up to `max_tokens`
* tables are atomic; oversized tables are split by rows with the header row repeated
* each chunk keeps `overlap_tokens` of trailing context from the previous chunk
* the embedded text is prefixed with a contextual header ("AAPL 10-K FY2024 | Item 7 ...")
  which measurably improves retrieval for multi-company indexes
"""

from __future__ import annotations

import hashlib
import re
import uuid
from collections.abc import Iterable
from dataclasses import dataclass
from functools import lru_cache

from finsight.models import Chunk, Section

NAMESPACE = uuid.UUID("6f1c2b1e-6a51-4a53-9a4e-2f7f3f1b9c11")


@lru_cache(maxsize=1)
def _encoder():
    try:
        import tiktoken

        return tiktoken.get_encoding("cl100k_base")
    except Exception:  # offline / no BPE files -> heuristic
        return None


def count_tokens(text: str) -> int:
    enc = _encoder()
    if enc is not None:
        return len(enc.encode(text, disallowed_special=()))
    return int(len(text.split()) * 1.3) + 1


@dataclass
class _Block:
    text: str
    is_table: bool


def _blocks(text: str) -> list[_Block]:
    out: list[_Block] = []
    buf: list[str] = []
    in_table = False
    for line in text.split("\n"):
        if line == "[TABLE]":
            if buf:
                out.extend(_Block(p, False) for p in buf)
                buf = []
            in_table, tbl = True, []
            continue
        if line == "[/TABLE]" and in_table:
            in_table = False
            if tbl:
                out.append(_Block("\n".join(tbl), True))
            continue
        if in_table:
            tbl.append(line)
        else:
            buf.append(line)
    if buf:
        out.extend(_Block(p, False) for p in buf)
    return out


def _split_long(text: str, max_tokens: int) -> Iterable[str]:
    """Split an oversized paragraph on sentence boundaries."""
    sents = re.split(r"(?<=[.!?])\s+(?=[A-Z(])", text)
    cur: list[str] = []
    for s in sents:
        if cur and count_tokens(" ".join([*cur, s])) > max_tokens:
            yield " ".join(cur)
            cur = []
        cur.append(s)
    if cur:
        yield " ".join(cur)


def _split_table(table: str, max_tokens: int) -> Iterable[str]:
    rows = table.split("\n")
    header, body = rows[0], rows[1:]
    cur = [header]
    for r in body:
        if len(cur) > 1 and count_tokens("\n".join([*cur, r])) > max_tokens:
            yield "\n".join(cur)
            cur = [header]
        cur.append(r)
    if len(cur) > 1:
        yield "\n".join(cur)


def _tail(text: str, overlap_tokens: int) -> str:
    """Sentence-aligned overlap: the last whole sentences of the previous chunk within budget."""
    if "[/TABLE]" in text[-20:]:
        return ""
    sents = re.split(r"(?<=[.!?])\s+", text.replace("\n", " "))
    out: list[str] = []
    for s in reversed(sents):
        if count_tokens(" ".join([s, *out])) > overlap_tokens:
            break
        out.insert(0, s)
    return " ".join(out) if len(out) < len(sents) else ""


def chunk_section(section: Section, max_tokens: int = 450, overlap_tokens: int = 60) -> list[str]:
    units: list[tuple[str, bool]] = []
    for b in _blocks(section.text):
        if b.is_table:
            units.extend((t, True) for t in (_split_table(b.text, max_tokens) if count_tokens(b.text) > max_tokens else [b.text]))
        elif count_tokens(b.text) > max_tokens:
            units.extend((t, False) for t in _split_long(b.text, max_tokens))
        else:
            units.append((b.text, False))

    chunks: list[str] = []
    cur: list[str] = []
    cur_tok = 0
    for text, is_table in units:
        rendered = f"[TABLE]\n{text}\n[/TABLE]" if is_table else text
        t = count_tokens(rendered)
        if cur and cur_tok + t > max_tokens:
            chunks.append("\n".join(cur))
            tail = _tail(chunks[-1], overlap_tokens) if not is_table else ""
            cur, cur_tok = ([tail] if tail else []), count_tokens(tail) if tail else 0
        cur.append(rendered)
        cur_tok += t
    if cur:
        chunks.append("\n".join(cur))
    return [c for c in chunks if len(c.strip()) > 40]


def chunk_filing(sections: list[Section], meta: dict, max_tokens: int = 450, overlap_tokens: int = 60) -> list[Chunk]:
    """Turn parsed sections into Chunks with rich, filterable metadata."""
    out: list[Chunk] = []
    idx = 0
    for sec in sections:
        for body in chunk_section(sec, max_tokens, overlap_tokens):
            header = f"{meta['ticker']} {meta['form']} FY{meta.get('fiscal_year')} | {sec.title}"
            cid = str(uuid.uuid5(NAMESPACE, f"{meta['accession']}:{idx}"))
            out.append(
                Chunk(
                    id=cid,
                    text=f"{header}\n{body}",
                    body=body,
                    metadata={
                        **meta,
                        "section": sec.key,
                        "section_title": sec.title,
                        "chunk_index": idx,
                        "token_count": count_tokens(body),
                        "has_table": "[TABLE]" in body,
                        "content_hash": hashlib.sha256(body.encode()).hexdigest()[:16],
                    },
                )
            )
            idx += 1
    return out
