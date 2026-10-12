"""Provider-agnostic LLM interface: chat, tool/function calling and structured outputs.

Messages use a neutral schema:
  {"role": "system"|"user"|"assistant"|"tool", "content": str,
   "tool_calls": [ToolCall...]  (assistant only), "tool_call_id": str (tool only), "name": str}
Each provider adapter translates to its native wire format.
"""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from finsight.llm.pricing import cost_usd
from finsight.models import Usage
from finsight.observability.metrics import LLM_COST, LLM_LATENCY, LLM_TOKENS
from finsight.observability.tracing import span

T = TypeVar("T", bound=BaseModel)


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]  # JSON schema


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class LLMResponse:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    model: str = ""
    raw: Any = None


class StructuredOutputError(RuntimeError):
    pass


class LLMProvider(ABC):
    name: str = "base"
    model: str = ""

    @abstractmethod
    def _chat(self, messages: list[dict], tools: list[ToolSpec] | None, schema: type[BaseModel] | None,
              temperature: float, max_tokens: int) -> LLMResponse: ...

    def chat(self, messages: list[dict], tools: list[ToolSpec] | None = None, schema: type[BaseModel] | None = None,
             temperature: float = 0.0, max_tokens: int = 1200, op: str = "chat") -> LLMResponse:
        with span(f"llm.{op}", {"llm.provider": self.name, "llm.model": self.model}) as sp:
            t0 = time.perf_counter()
            resp = self._chat(messages, tools, schema, temperature, max_tokens)
            dt = time.perf_counter() - t0
            resp.usage.cost_usd = cost_usd(self.model, resp.usage.input_tokens, resp.usage.output_tokens)
            LLM_LATENCY.labels(self.name, op).observe(dt)
            LLM_TOKENS.labels(self.name, self.model, "input").inc(resp.usage.input_tokens)
            LLM_TOKENS.labels(self.name, self.model, "output").inc(resp.usage.output_tokens)
            LLM_COST.labels(self.name, self.model).inc(resp.usage.cost_usd)
            sp.set_attribute("llm.input_tokens", resp.usage.input_tokens)
            sp.set_attribute("llm.output_tokens", resp.usage.output_tokens)
            sp.set_attribute("llm.cost_usd", resp.usage.cost_usd)
            sp.set_attribute("llm.tool_calls", len(resp.tool_calls))
            return resp

    def generate_structured(self, messages: list[dict], schema: type[T], max_retries: int = 2,
                            **kw) -> tuple[T, Usage]:
        """Ask for `schema`-shaped JSON; validate with Pydantic and repair on failure."""
        usage = Usage()
        msgs = list(messages)
        last_err: Exception | None = None
        for _ in range(max_retries + 1):
            resp = self.chat(msgs, schema=schema, op="structured", **kw)
            usage = usage + resp.usage
            try:
                return schema.model_validate(parse_json(resp.text)), usage
            except (ValidationError, ValueError) as e:
                last_err = e
                msgs = [*msgs, {"role": "assistant", "content": resp.text},
                        {"role": "user", "content": f"Your output failed validation: {e}. Return ONLY valid JSON for the schema."}]
        raise StructuredOutputError(str(last_err))


def parse_json(text: str) -> Any:
    t = text.strip()
    if t.startswith("```"):
        t = t.strip("`")
        t = t[t.find("\n") + 1 :] if "\n" in t else t
    start = min((i for i in (t.find("{"), t.find("[")) if i >= 0), default=-1)
    if start < 0:
        raise ValueError("no JSON object found")
    end = max(t.rfind("}"), t.rfind("]"))
    return json.loads(t[start : end + 1])
