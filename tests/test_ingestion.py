from finsight.ingestion.chunker import chunk_filing, chunk_section, count_tokens
from finsight.ingestion.parser import html_to_text, parse_filing, split_sections
from finsight.models import Section

from .conftest import FIXTURE_HTML


def test_html_to_text_drops_hidden_xbrl_and_keeps_tables():
    text = html_to_text(FIXTURE_HTML)
    assert "dei:EntityRegistrantName" not in text
    assert "[TABLE]" in text and "Total net revenue | 2,418 | 2,049 | 1,756" in text
    assert "$ |" not in text  # stray currency cells removed


def test_text_node_line_breaks_are_spaces():
    text = html_to_text("<p>Gross margin\nimproved to 46.5%.</p><p>Next</p>")
    assert "Gross margin improved to 46.5%." in text.split("\n")


def test_sections_ignore_toc_and_merge_running_headers():
    secs = parse_filing(FIXTURE_HTML, "10-K")
    keys = [s.key for s in secs]
    assert keys == ["item_1", "item_1a", "item_7", "item_8"]
    risk = next(s for s in secs if s.key == "item_1a")
    assert risk.title == "Item 1A. Risk Factors"
    # the running header "Item 1A" mid-section must not truncate the section
    assert "Kestrel Semiconductor" in risk.text and "convertible senior notes" in risk.text


def test_10q_parts_produce_distinct_keys():
    text = "\n".join(["PART I", "Item 1. Financial Statements", "x " * 150, "Item 2. Management's Discussion", "y " * 150,
                      "PART II", "Item 1. Legal Proceedings", "z " * 150, "Item 1A. Risk Factors", "w " * 150])
    keys = [s.key for s in split_sections(text, "10-Q")]
    assert keys == ["part_i_item_1", "part_i_item_2", "part_ii_item_1", "part_ii_item_1a"]


def test_fallback_when_no_items():
    secs = split_sections("just some text " * 50, "10-K")
    assert secs[0].key == "document"


def test_chunks_respect_budget_and_keep_tables_whole():
    secs = parse_filing(FIXTURE_HTML, "10-K")
    item8 = next(s for s in secs if s.key == "item_8")
    chunks = chunk_section(item8, max_tokens=120, overlap_tokens=20)
    table_chunks = [c for c in chunks if "[TABLE]" in c]
    assert len(table_chunks) == 1 and "[/TABLE]" in table_chunks[0]
    for c in chunk_section(next(s for s in secs if s.key == "item_7"), 120, 20):
        assert count_tokens(c) <= 120 + 40  # budget + sentence-aligned overlap


def test_oversized_table_is_split_with_header_repeated():
    rows = ["Metric | 2025 | 2024"] + [f"Line item {i} | {i * 10} | {i * 9}" for i in range(80)]
    sec = Section(key="item_8", title="Item 8.", text="[TABLE]\n" + "\n".join(rows) + "\n[/TABLE]")
    parts = chunk_section(sec, max_tokens=100, overlap_tokens=10)
    assert len(parts) > 2
    assert all(p.split("\n")[1] == "Metric | 2025 | 2024" for p in parts)


def test_chunk_metadata_and_deterministic_ids():
    secs = parse_filing(FIXTURE_HTML, "10-K")
    meta = {"ticker": "NWRB", "form": "10-K", "fiscal_year": 2025, "accession": "acc-1"}
    a, b = chunk_filing(secs, meta, 120, 20), chunk_filing(secs, meta, 120, 20)
    assert [c.id for c in a] == [c.id for c in b]
    assert a[0].text.startswith("NWRB 10-K FY2025 | Item 1. Business")
    assert {"section", "section_title", "chunk_index", "content_hash", "token_count"} <= set(a[0].metadata)
    assert len({c.id for c in a}) == len(a)


def test_ingest_is_idempotent(container, fake_sec):
    first = container.ingestion.ingest_ticker("NWRB")
    second = container.ingestion.ingest_ticker("NWRB")
    assert first.filings[0].status == "indexed" and first.chunks > 5
    assert second.filings[0].status == "skipped" and fake_sec.downloads == 1
    n = container.store.count()
    container.ingestion.ingest_ticker("NWRB", force=True)
    assert container.store.count() == n  # force re-index replaces, never duplicates


def test_unknown_ticker_raises(container):
    import pytest

    with pytest.raises(KeyError):
        container.ingestion.ingest_ticker("ZZZZ")
