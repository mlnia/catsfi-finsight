"""PII detection & masking (applied to user input before it reaches an LLM provider or the logs,
and to model output). Checksum validation (Luhn, IBAN mod-97, Turkish national ID) keeps false
positives low on numeric-heavy financial text."""

from __future__ import annotations

import re

EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
PHONE = re.compile(r"(?<![\w$-])(?:\+\d{1,3}[ .-]?)?(?:\(?\d{3}\)?[ .-]?)\d{3}[ .-]?\d{2,4}(?:[ .-]?\d{2})?(?![\w%]|-\d)")
SSN = re.compile(r"\b(?!000|666|9\d\d)\d{3}-(?!00)\d{2}-(?!0000)\d{4}\b")
CARD = re.compile(r"\b\d(?:[ -]?\d){12,18}\b")
IBAN = re.compile(r"\b[A-Z]{2}\d{2}(?:[ ]?[A-Z0-9]{4}){3,7}(?:[ ]?[A-Z0-9]{1,3})?\b")
TCKN = re.compile(r"\b[1-9]\d{10}\b")


def _luhn(num: str) -> bool:
    d = [int(c) for c in num if c.isdigit()]
    if not 13 <= len(d) <= 19:
        return False
    s = 0
    for i, x in enumerate(reversed(d)):
        if i % 2:
            x = x * 2 - 9 if x > 4 else x * 2
        s += x
    return s % 10 == 0


def _iban_ok(s: str) -> bool:
    s = s.replace(" ", "")
    r = s[4:] + s[:4]
    return int("".join(str(int(c, 36)) for c in r)) % 97 == 1


def _tckn_ok(s: str) -> bool:
    d = [int(c) for c in s]
    return ((sum(d[0:9:2]) * 7 - sum(d[1:8:2])) % 10 == d[9]) and (sum(d[:10]) % 10 == d[10])


def mask_pii(text: str) -> tuple[str, list[str]]:
    found: list[str] = []

    def sub(pattern: re.Pattern, label: str, check=None):
        nonlocal text

        def rep(m: re.Match) -> str:
            if check and not check(m.group(0)):
                return m.group(0)
            found.append(label)
            return f"[{label}]"

        text = pattern.sub(rep, text)

    sub(EMAIL, "EMAIL")
    sub(IBAN, "IBAN", _iban_ok)
    sub(CARD, "CARD_NUMBER", _luhn)
    sub(SSN, "SSN")
    sub(TCKN, "TR_ID", _tckn_ok)
    sub(PHONE, "PHONE", lambda s: sum(c.isdigit() for c in s) >= 10 and not re.fullmatch(r"(19|20)\d{2}.*", s.strip()))
    return text, sorted(set(found))
