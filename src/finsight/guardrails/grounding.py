"""Hallucination mitigation: post-generation verification of an answer against its evidence.

1. citation validity  - cited ids must exist in the context that was actually provided
2. quote fidelity     - each quote must fuzzy-match (>= 85) text inside the cited chunk
3. numeric fidelity   - every number in the answer must appear in the evidence (the most
                        damaging hallucination in finance is a wrong figure)
4. groundedness score - share of answer sentences whose content words are supported by evidence
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from rapidfuzz import fuzz

from finsight.retrieval.embeddings import tokenize

NUM = re.compile(r"(?<![\w.])\$?\(?-?\d[\d,]*(?:\.\d+)?\)?\s?(?:%|percent|billion|million|thousand|bn|m|b)?", re.I)
CITE = re.compile(r"\[([A-Za-z0-9:_.\-]+)\]")
SENT = re.compile(r"(?<=[.!?])\s+")


@dataclass
class GroundingResult:
    groundedness: float
    invalid_citations: list[str] = field(default_factory=list)
    bad_quotes: list[str] = field(default_factory=list)
    unsupported_numbers: list[str] = field(default_factory=list)
    cited_ids: list[str] = field(default_factory=list)

    def passes(self, min_groundedness: float) -> bool:
        return self.groundedness >= min_groundedness and not self.unsupported_numbers and bool(self.cited_ids)


def _norm_num(s: str) -> str | None:
    core = re.sub(r"[^\d.]", "", s)
    core = core.rstrip(".")
    if not core or core == ".":
        return None
    try:
        v = float(core)
    except ValueError:
        return None
    return f"{v:.6g}"


def _numbers(text: str) -> set[str]:
    out = set()
    for m in NUM.finditer(text):
        n = _norm_num(m.group(0))
        if n is not None:
            out.add(n)
    return out


def check_grounding(answer: str, evidence: dict[str, str], quotes: list[tuple[str, str]] | None = None) -> GroundingResult:
    """`evidence` maps id -> text; `quotes` is [(id, quote)] from structured output (optional)."""
    cited = list(dict.fromkeys(CITE.findall(answer) + [cid for cid, _ in quotes or []]))
    invalid = [c for c in cited if c not in evidence]
    valid = [c for c in cited if c in evidence]

    bad_quotes = []
    for cid, q in quotes or []:
        if cid in evidence and q and fuzz.partial_ratio(q.lower(), evidence[cid].lower()) < 85:
            bad_quotes.append(cid)

    support_text = " ".join(evidence[c] for c in valid) if valid else ""
    all_evidence = " ".join(evidence.values())
    ev_nums = _numbers(all_evidence)
    years = {str(y) for y in range(1990, 2101)}
    clean_answer = CITE.sub(" ", answer)
    unsupported = sorted(n for n in _numbers(clean_answer) if n not in ev_nums and n not in years and float(n) > 10)

    sup_tokens = set(tokenize(support_text))
    sents = [s for s in SENT.split(clean_answer) if len(tokenize(s)) >= 3]
    if not sents:
        score = 1.0 if valid else 0.0
    else:
        supported = 0
        for s in sents:
            toks = set(tokenize(s))
            if toks and len(toks & sup_tokens) / len(toks) >= 0.6:
                supported += 1
        score = supported / len(sents)
    return GroundingResult(round(score, 3), invalid, bad_quotes, unsupported, valid)
