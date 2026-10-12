"""SEC filing parser: (inline-XBRL) HTML -> clean text split into Item sections.

* drops hidden XBRL headers, scripts and styles
* keeps financial tables, flattened to `cell | cell` rows inside [TABLE] ... [/TABLE] markers
* detects "Item 1A. Risk Factors"-style headings and resolves table-of-contents duplicates by
  keeping the longest span for every item
"""

from __future__ import annotations

import re
import warnings

from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

from finsight.models import Section

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

PARSER_VERSION = "1.4"

BLOCK_TAGS = ["p", "div", "br", "li", "h1", "h2", "h3", "h4", "h5", "h6", "tr", "section", "article"]

TEN_K_ITEMS = {
    "1": "Business", "1A": "Risk Factors", "1B": "Unresolved Staff Comments", "1C": "Cybersecurity",
    "2": "Properties", "3": "Legal Proceedings", "4": "Mine Safety Disclosures",
    "5": "Market for Registrant's Common Equity, Related Stockholder Matters and Issuer Purchases of Equity Securities",
    "6": "[Reserved]", "7": "Management's Discussion and Analysis of Financial Condition and Results of Operations",
    "7A": "Quantitative and Qualitative Disclosures About Market Risk",
    "8": "Financial Statements and Supplementary Data",
    "9": "Changes in and Disagreements with Accountants on Accounting and Financial Disclosure",
    "9A": "Controls and Procedures", "9B": "Other Information",
    "9C": "Disclosure Regarding Foreign Jurisdictions that Prevent Inspections",
    "10": "Directors, Executive Officers and Corporate Governance", "11": "Executive Compensation",
    "12": "Security Ownership of Certain Beneficial Owners and Management and Related Stockholder Matters",
    "13": "Certain Relationships and Related Transactions, and Director Independence",
    "14": "Principal Accountant Fees and Services", "15": "Exhibit and Financial Statement Schedules",
    "16": "Form 10-K Summary",
}
TEN_Q_ITEMS = {
    "I": {"1": "Financial Statements", "2": "Management's Discussion and Analysis of Financial Condition and Results of Operations",
          "3": "Quantitative and Qualitative Disclosures About Market Risk", "4": "Controls and Procedures"},
    "II": {"1": "Legal Proceedings", "1A": "Risk Factors", "2": "Unregistered Sales of Equity Securities and Use of Proceeds",
           "3": "Defaults Upon Senior Securities", "4": "Mine Safety Disclosures", "5": "Other Information", "6": "Exhibits"},
}

ITEM_RE = re.compile(r"^\s*item\s*(\d{1,2}[a-c]?)\s*[\.:\-–—|]?\s*(.{0,160})$", re.IGNORECASE)
PART_RE = re.compile(r"^\s*part\s+(I{1,2}|IV|III)\b\s*[\.:\-–—|]?", re.IGNORECASE)
TOC_LINE = re.compile(r"\|\s*\d{1,3}\s*$")
_WS = re.compile(r"[ \t  ​]+")


def html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "head", "noscript"]):
        tag.decompose()
    for tag in soup.find_all(["ix:header", "ix:hidden"]):
        tag.decompose()
    for tag in soup.find_all(style=re.compile(r"display\s*:\s*none", re.I)):
        tag.decompose()
    for s in soup.find_all(string=True):  # source line breaks inside a text node are just spaces
        if "\n" in s or "\r" in s:
            s.replace_with(s.replace("\r", " ").replace("\n", " "))

    for table in soup.find_all("table"):
        rows = []
        for tr in table.find_all("tr"):
            cells = [_WS.sub(" ", td.get_text(" ", strip=True)) for td in tr.find_all(["td", "th"])]
            cells = [c for c in cells if c and c not in {"$", "%", ")"}]
            if cells:
                rows.append(" | ".join(cells))
        # single-row tables are layout tables (headings, page numbers) -> plain line
        replacement = ("\n[TABLE]\n" + "\n".join(rows) + "\n[/TABLE]\n") if len(rows) > 1 else ("\n" + " ".join(rows) + "\n")
        table.replace_with(soup.new_string(replacement))

    for br in soup.find_all("br"):
        br.replace_with("\n")
    for tag in soup.find_all(BLOCK_TAGS):
        tag.append("\n")

    text = soup.get_text("")
    lines = [_WS.sub(" ", ln).strip() for ln in text.splitlines()]
    out: list[str] = []
    for ln in lines:
        if not ln:
            continue
        if re.fullmatch(r"(page\s*)?\d{1,3}|table of contents", ln, re.I):  # page numbers / running headers
            continue
        out.append(ln)
    return "\n".join(out)


def _title(form: str, part: str | None, item: str) -> str:
    if form.startswith("10-Q") and part in TEN_Q_ITEMS:
        name = TEN_Q_ITEMS[part].get(item, "")
        return f"Part {part}, Item {item}. {name}".strip()
    return f"Item {item}. {TEN_K_ITEMS.get(item, '')}".strip()


def split_sections(text: str, form: str = "10-K") -> list[Section]:
    lines = text.split("\n")
    heads: list[tuple[int, str, str]] = []  # (line_idx, key, title)
    part: str | None = None
    for i, ln in enumerate(lines):
        if len(ln) > 200:
            continue
        pm = PART_RE.match(ln)
        if pm:
            part = pm.group(1).upper()
        m = ITEM_RE.match(ln.replace("[TABLE]", ""))
        if not m or TOC_LINE.search(ln):  # table-of-contents rows end with a page number
            continue
        item = m.group(1).upper()
        if form.startswith("10-K") and item not in TEN_K_ITEMS:
            continue
        key = f"part_{part.lower()}_item_{item.lower()}" if form.startswith("10-Q") and part else f"item_{item.lower()}"
        if heads and heads[-1][1] == key:  # running page headers repeat the current item -> same section
            continue
        heads.append((i, key, _title(form, part, item)))

    if not heads:
        return [Section(key="document", title="Full document", text=text)]

    best: dict[str, tuple[int, int, str]] = {}
    for n, (start, key, title) in enumerate(heads):
        end = heads[n + 1][0] if n + 1 < len(heads) else len(lines)
        size = sum(len(x) for x in lines[start:end])
        if key not in best or size > _span(lines, best[key][0], best[key][1]):
            best[key] = (start, end, title)

    sections = []
    for key, (start, end, title) in sorted(best.items(), key=lambda kv: kv[1][0]):
        body = "\n".join(lines[start + 1 : end]).strip()
        if len(body) < 200:  # TOC remnants, "[Reserved]", "Not applicable."
            continue
        sections.append(Section(key=key, title=title, text=body))
    return sections or [Section(key="document", title="Full document", text=text)]


def _span(lines: list[str], start: int, end: int) -> int:
    return sum(len(x) for x in lines[start:end])


def parse_filing(html: str, form: str) -> list[Section]:
    return split_sections(html_to_text(html), form)
