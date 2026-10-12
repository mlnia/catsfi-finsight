"""Audited annual fundamentals from SEC XBRL `companyfacts` - structured numbers the agent can
cite instead of letting an LLM read figures out of prose (a major hallucination source)."""

from __future__ import annotations

from datetime import date

CONCEPT_ALIASES: dict[str, list[str]] = {
    "Revenues": ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet",
                 "RevenueFromContractWithCustomerIncludingAssessedTax"],
    "NetIncomeLoss": ["NetIncomeLoss", "ProfitLoss"],
    "OperatingIncomeLoss": ["OperatingIncomeLoss"],
    "GrossProfit": ["GrossProfit"],
    "EarningsPerShareDiluted": ["EarningsPerShareDiluted"],
    "ResearchAndDevelopmentExpense": ["ResearchAndDevelopmentExpense"],
    "CashAndCashEquivalentsAtCarryingValue": ["CashAndCashEquivalentsAtCarryingValue"],
    "Assets": ["Assets"],
    "Liabilities": ["Liabilities"],
    "NetCashProvidedByUsedInOperatingActivities": ["NetCashProvidedByUsedInOperatingActivities"],
}
SUPPORTED_METRICS = sorted(CONCEPT_ALIASES)


def fmt_value(v: float, unit: str | None) -> str:
    if unit and unit.startswith("USD") and "/" not in unit and abs(v) >= 1e6:
        return f"${v / 1e9:,.2f}B" if abs(v) >= 1e9 else f"${v / 1e6:,.1f}M"
    if unit and "/" in unit:
        return f"${v:,.2f}"
    return f"{v:,.2f}"


def _days(a: str, b: str) -> int:
    return (date.fromisoformat(b) - date.fromisoformat(a)).days


def annual_series(facts: dict, metric: str, years: int = 3) -> list[dict]:
    gaap = facts.get("facts", {}).get("us-gaap", {})
    best: list[dict] = []
    for concept in CONCEPT_ALIASES.get(metric, [metric]):
        node = gaap.get(concept)
        if not node:
            continue
        for unit, items in node.get("units", {}).items():
            by_end: dict[str, dict] = {}
            for it in items:
                if not str(it.get("form", "")).startswith("10-K") or it.get("fp") != "FY":
                    continue
                if "start" in it and not 350 <= _days(it["start"], it["end"]) <= 380:
                    continue  # skip quarterly / cumulative sub-periods inside the 10-K
                cur = by_end.get(it["end"])
                if cur is None or it.get("filed", "") > cur.get("filed", ""):
                    by_end[it["end"]] = it
            rows = [
                {"fy": int(end[:4]), "period_end": end, "value": float(it["val"]), "unit": unit, "concept": concept,
                 "accession": it.get("accn"), "filed": it.get("filed")}
                for end, it in sorted(by_end.items())
            ][-years:]
            if rows and (not best or rows[-1]["period_end"] > best[-1]["period_end"]):
                best = rows
    return best
