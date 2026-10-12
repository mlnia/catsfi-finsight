import json

import pytest

from finsight.llm.base import LLMProvider, LLMResponse, StructuredOutputError, ToolCall, parse_json
from finsight.llm.pricing import cost_usd
from finsight.llm.providers import AnthropicProvider, OpenAIChatProvider
from finsight.models import RAGAnswer, Usage
from finsight.prompts.registry import get_prompt, list_prompts


class ScriptedLLM(LLMProvider):
    name, model = "scripted", "gpt-4o-mini"

    def __init__(self, outputs):
        self.outputs, self.calls = list(outputs), []

    def _chat(self, messages, tools, schema, temperature, max_tokens):
        self.calls.append(messages)
        return LLMResponse(text=self.outputs.pop(0), usage=Usage(input_tokens=1000, output_tokens=100))


def test_parse_json_handles_fences_and_prose():
    assert parse_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json('Sure! {"a": [1, 2]} hope this helps') == {"a": [1, 2]}
    with pytest.raises(ValueError):
        parse_json("no json here")


def test_structured_output_repair_loop():
    good = json.dumps({"answer": "ok [c1]", "citations": [{"chunk_id": "c1", "quote": "x"}], "confidence": "high"})
    llm = ScriptedLLM(['{"answer": 5, "confidence": "certain"}', good])
    ans, usage = llm.generate_structured([{"role": "user", "content": "q"}], RAGAnswer)
    assert ans.answer == "ok [c1]" and len(llm.calls) == 2
    assert "failed validation" in llm.calls[1][-1]["content"]
    assert usage.input_tokens == 2000 and usage.cost_usd == pytest.approx(2 * cost_usd("gpt-4o-mini", 1000, 100))


def test_structured_output_gives_up():
    with pytest.raises(StructuredOutputError):
        ScriptedLLM(["nope"] * 3).generate_structured([{"role": "user", "content": "q"}], RAGAnswer, max_retries=2)


def test_openai_wire_format_for_tool_turns():
    msgs = [{"role": "user", "content": "hi"},
            {"role": "assistant", "content": "", "tool_calls": [ToolCall("t1", "search_filings", {"query": "x"})]},
            {"role": "tool", "tool_call_id": "t1", "name": "search_filings", "content": "{}"}]
    wire = OpenAIChatProvider._to_wire(msgs)
    assert wire[1]["tool_calls"][0]["function"] == {"name": "search_filings", "arguments": '{"query": "x"}'}
    assert wire[2] == {"role": "tool", "tool_call_id": "t1", "content": "{}"}


def test_anthropic_wire_format_merges_parallel_tool_results():
    msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "", "tool_calls": [ToolCall("a", "x", {}), ToolCall("b", "y", {})]},
            {"role": "tool", "tool_call_id": "a", "content": "1"}, {"role": "tool", "tool_call_id": "b", "content": "2"}]
    system, wire = AnthropicProvider._to_wire(msgs)
    assert system == "sys"
    assert [b["type"] for b in wire[1]["content"]] == ["tool_use", "tool_use"]
    assert len(wire) == 3 and [b["tool_use_id"] for b in wire[2]["content"]] == ["a", "b"]


def test_pricing():
    assert cost_usd("gpt-4o-mini-2024-07-18", 1_000_000, 0) == 0.15
    assert cost_usd("unknown-model", 10, 10) == 0.0


def test_prompt_registry_versions():
    latest = get_prompt("rag_answer")
    assert latest.ref == "rag_answer@v2"
    v1 = get_prompt("rag_answer@v1")
    assert v1.sha != latest.sha and len(latest.sha) == 12
    msgs = latest.render(context="<source id=\"c1\">x</source>", question="why?")
    assert msgs[0]["role"] == "system" and "<question>why?</question>" in msgs[1]["content"]
    with pytest.raises(KeyError):
        get_prompt("rag_answer@v99")
    assert {"rag_answer@v1", "rag_answer@v2", "agent_system@v1"} <= {p["ref"] for p in list_prompts()}
