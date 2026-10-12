"""Prompt-injection detection for direct (user input) and indirect (retrieved documents /
tool output) attacks. Weighted heuristics are fast, explainable and cheap; the score is
calibrated against the adversarial set in `eval/datasets/adversarial.jsonl`."""

from __future__ import annotations

import base64
import re
from dataclasses import dataclass, field

RULES: list[tuple[str, float, re.Pattern]] = [
    ("override_instructions", 0.7, re.compile(
        r"\b(ignore|disregard|forget|override|bypass)\b.{0,40}\b(previous|prior|above|earlier|all|system|your)\b.{0,30}"
        r"\b(instructions?|prompts?|rules?|guidelines?|context|directives?)\b", re.I | re.S)),
    ("reveal_system_prompt", 0.7, re.compile(
        r"\b(reveal|show|print|repeat|output|leak|display|tell me)\b.{0,40}\b(system|hidden|initial|original)\s*(prompt|instructions?|message)", re.I | re.S)),
    ("role_hijack", 0.5, re.compile(
        r"\b(you are now|act as|pretend (to be|you are)|from now on,? you|new persona|roleplay as)\b", re.I)),
    ("jailbreak_keywords", 0.6, re.compile(r"\b(DAN|do anything now|developer mode|jailbreak|unfiltered mode|no restrictions)\b", re.I)),
    ("fake_delimiters", 0.6, re.compile(r"(</?(system|assistant|instructions?|source|sources)>|\[/?INST\]|<\|im_(start|end)\|>|###\s*system)", re.I)),
    ("exfiltration", 0.6, re.compile(
        r"\b(send|post|upload|exfiltrate|forward)\b.{0,40}\b(https?://|webhook|e-?mail|api key|credentials?|password)", re.I | re.S)),
    ("tool_coercion", 0.6, re.compile(r"\b(call|invoke|execute|run)\b.{0,20}\b(tool|function|ingest_company_filings)\b.{0,40}\b(without|no)\b.{0,20}\b(approval|confirmation)", re.I | re.S)),
    ("new_instruction", 0.4, re.compile(r"\b(new|updated|additional|real)\s+(instructions?|rules?|directives?|task)\s*:", re.I)),
    ("addressed_to_ai", 0.5, re.compile(
        r"\b(note|notice|message|instructions?|attention)\s+(to|for)\s+(all\s+)?(ai|llms?|language models?|assistants?|chatbots?|ai systems?)\b", re.I)),
    ("secrets_request", 0.4, re.compile(r"\b(api[_ -]?key|secret key|access token|env(ironment)? variables?)\b", re.I)),
]
B64 = re.compile(r"\b[A-Za-z0-9+/]{40,}={0,2}\b")


@dataclass
class InjectionResult:
    score: float
    flags: list[str] = field(default_factory=list)

    def is_attack(self, threshold: float) -> bool:
        return self.score >= threshold


def _decoded_payloads(text: str) -> list[str]:
    out = []
    for m in B64.findall(text):
        try:
            dec = base64.b64decode(m + "=" * (-len(m) % 4)).decode("utf-8")
            if dec.isprintable():
                out.append(dec)
        except Exception:
            continue
    return out


def detect_injection(text: str) -> InjectionResult:
    flags: list[str] = []
    score = 0.0
    candidates = [text, *_decoded_payloads(text)]
    for name, weight, pat in RULES:
        if any(pat.search(c) for c in candidates):
            flags.append(name)
            score = 1 - (1 - score) * (1 - weight)  # noisy-OR combination
    if len(candidates) > 1 and flags:
        flags.append("encoded_payload")
        score = min(1.0, score + 0.15)
    return InjectionResult(round(score, 3), flags)


_SENT = re.compile(r"((?<=[.!?:])[ \t]+|\n+)")


def sanitize_untrusted(text: str, threshold: float) -> tuple[str, int]:
    """Remove injected sentences from untrusted content (retrieved chunks, tool output) while keeping
    the legitimate remainder. Returns (clean_text, removed_count); clean_text is "" when the whole
    passage is adversarial."""
    if not detect_injection(text).is_attack(threshold):
        return text, 0
    parts = _SENT.split(text)  # odd indexes are separators -> paragraph / table layout is preserved
    kept: list[str] = []
    removed = 0
    for i in range(0, len(parts), 2):
        seg, sep = parts[i], parts[i + 1] if i + 1 < len(parts) else ""
        if seg.strip() and detect_injection(seg).is_attack(threshold * 0.8):
            removed += 1
            continue
        kept.append(seg + sep)
    clean = "".join(kept).strip()
    if detect_injection(clean).is_attack(threshold):  # attack spread across sentences -> drop passage
        return "", removed + len(kept)
    return clean, removed
