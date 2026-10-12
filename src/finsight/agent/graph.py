"""LangGraph research agent with human-in-the-loop approval.

    START -> guard_input --blocked--> finalize -> END
                  |
                  v
               agent  <-----------------+
                  |  tool calls?         |
          +-------+--------+             |
          | gated |        | safe        |
          v       |        v             |
   human_approval-+----> tools ----------+
     (interrupt)
                  | no tool calls / step budget hit
                  v
              finalize (citation + numeric grounding, PII masking)

State is checkpointed per `thread_id`, which gives multi-turn memory and lets a run pause at
`human_approval` (interrupt) and resume later via the API with the reviewer's decision.
"""

from __future__ import annotations

import json
import operator
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from typing import Annotated, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from finsight.agent.tools import Tool, evidence_from_tool
from finsight.config import Settings
from finsight.guardrails.grounding import check_grounding
from finsight.guardrails.injection import detect_injection, sanitize_untrusted
from finsight.guardrails.pii import mask_pii
from finsight.llm.base import LLMProvider, ToolCall
from finsight.observability.metrics import GUARDRAIL_EVENTS
from finsight.observability.tracing import span
from finsight.prompts.registry import get_prompt


def _merge(a: dict, b: dict) -> dict:
    return {**a, **b}


def _add_usage(a: dict, b: dict) -> dict:
    return {k: round(a.get(k, 0) + b.get(k, 0), 6) for k in {"input_tokens", "output_tokens", "cost_usd"}}


class AgentState(TypedDict, total=False):
    messages: Annotated[list[dict], operator.add]
    evidence: Annotated[dict[str, str], _merge]
    usage: Annotated[dict, _add_usage]
    trace: Annotated[list[dict], operator.add]
    pending: list[dict]
    approval: dict | None
    steps: int
    guard: dict
    final: dict | None


def _to_provider(messages: list[dict]) -> list[dict]:
    out = []
    for m in messages:
        if m.get("tool_calls"):
            m = {**m, "tool_calls": [ToolCall(tc["id"], tc["name"], tc["arguments"]) for tc in m["tool_calls"]]}
        out.append(m)
    return out


def build_agent_graph(llm: LLMProvider, tools: dict[str, Tool], s: Settings, checkpointer=None):
    prompt = get_prompt(s.agent_prompt)
    specs = [t.spec for t in tools.values()]

    def guard_input(state: AgentState) -> dict:
        # PII was already masked by AgentService before the turn entered the graph / checkpointer.
        inj = detect_injection(state["messages"][-1]["content"])
        guard = {"input_flags": inj.flags, "pii_masked": (state.get("guard") or {}).get("pii_masked", []), "action": "pass"}
        if inj.is_attack(s.injection_threshold) and s.guardrail_mode == "block":
            GUARDRAIL_EVENTS.labels("prompt_injection", "block").inc()
            guard["action"] = "blocked"
        return {"guard": guard, "steps": 0, "final": None, "pending": [], "approval": None}

    def after_guard(state: AgentState) -> str:
        return "finalize" if state["guard"]["action"] == "blocked" else "agent"

    def agent(state: AgentState) -> dict:
        msgs = state["messages"]
        system = prompt.render(today=date.today().isoformat())
        with span("agent.step", {"step": state.get("steps", 0)}):
            resp = llm.chat(_to_provider(system + msgs), tools=specs, temperature=s.llm_temperature,
                            max_tokens=s.llm_max_tokens, op="agent")
        calls = [{"id": tc.id, "name": tc.name, "arguments": tc.arguments} for tc in resp.tool_calls]
        msg = {"role": "assistant", "content": resp.text, **({"tool_calls": calls} if calls else {})}
        return {"messages": [msg], "pending": calls, "steps": state.get("steps", 0) + 1,
                "usage": resp.usage.model_dump()}

    def after_agent(state: AgentState) -> str:
        if not state.get("pending"):
            return "finalize"
        if state["steps"] > s.agent_max_steps:
            return "finalize"
        if any(tools.get(c["name"]) and tools[c["name"]].requires_approval for c in state["pending"]):
            return "human_approval"
        return "tools"

    def human_approval(state: AgentState) -> Command:
        gated = [c for c in state["pending"] if tools.get(c["name"]) and tools[c["name"]].requires_approval]
        decision = interrupt({"type": "approval_required", "tool_calls": gated})
        return Command(goto="tools", update={"approval": decision if isinstance(decision, dict) else {"approved": bool(decision)}})

    def run_tools(state: AgentState) -> dict:
        approval = state.get("approval") or {}
        results: list[dict] = []
        evidence: dict[str, str] = {}
        trace: list[dict] = []

        def run_one(c: dict) -> tuple[dict, dict]:
            tool = tools.get(c["name"])
            if tool is None:
                return c, {"error": f"unknown tool {c['name']}"}
            if tool.requires_approval and not approval.get("approved"):
                return c, {"error": f"Rejected by human reviewer: {approval.get('reason') or 'not approved'}"}
            return c, tool(c["arguments"])

        with ThreadPoolExecutor(max_workers=4) as ex:
            outs = list(ex.map(run_one, state["pending"]))
        for c, out in outs:
            text = json.dumps(out, default=str)
            if c["name"] == "search_filings":  # indirect injection inside tool output
                for r in out.get("results", []):
                    clean, removed = sanitize_untrusted(r["text"], s.injection_threshold)
                    if removed:
                        r["text"] = clean or "[quarantined: possible prompt injection]"
                        GUARDRAIL_EVENTS.labels("indirect_injection", "sanitize").inc()
                text = json.dumps(out, default=str)
            results.append({"role": "tool", "tool_call_id": c["id"], "name": c["name"], "content": text[:12000]})
            evidence.update(evidence_from_tool(c["name"], out))
            trace.append({"tool": c["name"], "arguments": c["arguments"], "ok": "error" not in out,
                          "error": out.get("error"), "approved": approval.get("approved") if tools.get(c["name"]) and tools[c["name"]].requires_approval else None})
        return {"messages": results, "evidence": evidence, "trace": trace, "pending": [], "approval": None}

    def finalize(state: AgentState) -> dict:
        guard = dict(state.get("guard") or {})
        if guard.get("action") == "blocked":
            return {"final": {"answer": "This request was blocked by the input safety policy (possible prompt injection).",
                              "guardrails": guard}}
        last = next((m for m in reversed(state["messages"]) if m["role"] == "assistant"), {"content": ""})
        answer = last.get("content") or ""
        if state.get("pending"):
            answer = (answer + "\n\n" if answer else "") + "Stopped: step budget exhausted before the research was complete."
        evidence = state.get("evidence") or {}
        g = check_grounding(answer, evidence) if evidence else None
        if g:
            guard.update(groundedness=g.groundedness, invalid_citations=g.invalid_citations,
                         unsupported_numbers=g.unsupported_numbers)
            if g.invalid_citations or g.unsupported_numbers:
                GUARDRAIL_EVENTS.labels("ungrounded_answer", "flag").inc()
                guard["action"] = "flagged"
                answer += "\n\n⚠️ Some statements could not be verified against tool outputs: " + \
                          ", ".join(g.invalid_citations + g.unsupported_numbers)
        if s.mask_pii:
            answer, pii = mask_pii(answer)
            guard["pii_masked"] = sorted(set(guard.get("pii_masked", [])) | set(pii))
        return {"final": {"answer": answer, "guardrails": guard}}

    g = StateGraph(AgentState)
    g.add_node("guard_input", guard_input)
    g.add_node("agent", agent)
    g.add_node("human_approval", human_approval, destinations=("tools",))
    g.add_node("tools", run_tools)
    g.add_node("finalize", finalize)
    g.add_edge(START, "guard_input")
    g.add_conditional_edges("guard_input", after_guard, ["agent", "finalize"])
    g.add_conditional_edges("agent", after_agent, ["human_approval", "tools", "finalize"])
    g.add_edge("tools", "agent")
    g.add_edge("finalize", END)
    return g.compile(checkpointer=checkpointer or InMemorySaver())

