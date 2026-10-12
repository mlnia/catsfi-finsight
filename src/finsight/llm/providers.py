"""Concrete LLM adapters: OpenAI, Azure OpenAI, any OpenAI-compatible endpoint (Ollama, vLLM,
Gemini, Groq...), and Anthropic. Each supports tool calling and schema-constrained output."""

from __future__ import annotations

import json
import uuid

from pydantic import BaseModel

from finsight.config import Settings
from finsight.llm.base import LLMProvider, LLMResponse, ToolCall
from finsight.models import Usage


class OpenAIChatProvider(LLMProvider):
    def __init__(self, client, model: str, name: str = "openai", json_schema_mode: bool = True):
        self._c, self.model, self.name, self._schema_mode = client, model, name, json_schema_mode

    @staticmethod
    def _to_wire(messages: list[dict]) -> list[dict]:
        out = []
        for m in messages:
            if m["role"] == "assistant" and m.get("tool_calls"):
                out.append({
                    "role": "assistant", "content": m.get("content") or None,
                    "tool_calls": [{"id": tc.id, "type": "function",
                                    "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)}}
                                   for tc in m["tool_calls"]],
                })
            elif m["role"] == "tool":
                out.append({"role": "tool", "tool_call_id": m["tool_call_id"], "content": m["content"]})
            else:
                out.append({"role": m["role"], "content": m["content"]})
        return out

    def _chat(self, messages, tools, schema, temperature, max_tokens) -> LLMResponse:
        kw: dict = {"model": self.model, "messages": self._to_wire(messages), "temperature": temperature,
                    "max_tokens": max_tokens}
        if tools:
            kw["tools"] = [{"type": "function", "function": {"name": t.name, "description": t.description,
                                                             "parameters": t.parameters}} for t in tools]
        if schema is not None:
            if self._schema_mode:
                kw["response_format"] = {"type": "json_schema", "json_schema": {
                    "name": schema.__name__, "schema": schema.model_json_schema(), "strict": False}}
            else:  # many OpenAI-compatible servers only support JSON mode
                kw["response_format"] = {"type": "json_object"}
                kw["messages"] = [*kw["messages"], {"role": "system", "content":
                                  "Respond with JSON matching this schema:\n" + json.dumps(schema.model_json_schema())}]
        r = self._c.chat.completions.create(**kw)
        msg = r.choices[0].message
        calls = [ToolCall(id=tc.id, name=tc.function.name, arguments=json.loads(tc.function.arguments or "{}"))
                 for tc in (msg.tool_calls or [])]
        u = r.usage
        return LLMResponse(text=msg.content or "", tool_calls=calls, model=r.model,
                           usage=Usage(input_tokens=getattr(u, "prompt_tokens", 0) or 0,
                                       output_tokens=getattr(u, "completion_tokens", 0) or 0), raw=r)


class AnthropicProvider(LLMProvider):
    name = "anthropic"

    def __init__(self, client, model: str):
        self._c, self.model = client, model

    @staticmethod
    def _to_wire(messages: list[dict]) -> tuple[str, list[dict]]:
        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        out: list[dict] = []
        for m in messages:
            if m["role"] == "system":
                continue
            if m["role"] == "assistant" and m.get("tool_calls"):
                blocks = ([{"type": "text", "text": m["content"]}] if m.get("content") else [])
                blocks += [{"type": "tool_use", "id": tc.id, "name": tc.name, "input": tc.arguments} for tc in m["tool_calls"]]
                out.append({"role": "assistant", "content": blocks})
            elif m["role"] == "tool":
                block = {"type": "tool_result", "tool_use_id": m["tool_call_id"], "content": m["content"]}
                if out and out[-1]["role"] == "user" and isinstance(out[-1]["content"], list):
                    out[-1]["content"].append(block)  # parallel tool results go in one user turn
                else:
                    out.append({"role": "user", "content": [block]})
            else:
                out.append({"role": m["role"], "content": m["content"]})
        return system, out

    def _chat(self, messages, tools, schema: type[BaseModel] | None, temperature, max_tokens) -> LLMResponse:
        system, wire = self._to_wire(messages)
        kw: dict = {"model": self.model, "messages": wire, "max_tokens": max_tokens, "temperature": temperature}
        if system:
            kw["system"] = system
        wire_tools = [{"name": t.name, "description": t.description, "input_schema": t.parameters} for t in tools or []]
        if schema is not None:  # structured output = forced single tool call
            wire_tools.append({"name": "respond", "description": "Return the final structured answer.",
                               "input_schema": schema.model_json_schema()})
            kw["tool_choice"] = {"type": "tool", "name": "respond"}
        if wire_tools:
            kw["tools"] = wire_tools
        r = self._c.messages.create(**kw)
        text, calls = "", []
        for b in r.content:
            if b.type == "text":
                text += b.text
            elif b.type == "tool_use":
                if b.name == "respond" and schema is not None:
                    text = json.dumps(b.input)
                else:
                    calls.append(ToolCall(id=b.id, name=b.name, arguments=dict(b.input)))
        return LLMResponse(text=text, tool_calls=calls, model=r.model,
                           usage=Usage(input_tokens=r.usage.input_tokens, output_tokens=r.usage.output_tokens), raw=r)


def build_llm(s: Settings) -> LLMProvider:
    p = s.llm_provider
    if p == "offline":
        from finsight.llm.offline import OfflineProvider

        return OfflineProvider()
    if p == "openai":
        from openai import OpenAI

        return OpenAIChatProvider(OpenAI(api_key=s.openai_api_key, timeout=s.llm_timeout_s, max_retries=3), s.llm_model)
    if p == "openai_compatible":
        from openai import OpenAI

        c = OpenAI(api_key=s.openai_api_key or "not-needed", base_url=s.openai_base_url, timeout=s.llm_timeout_s, max_retries=3)
        return OpenAIChatProvider(c, s.llm_model, name="openai_compatible", json_schema_mode=False)
    if p == "azure_openai":
        from openai import AzureOpenAI

        c = AzureOpenAI(api_key=s.azure_openai_api_key, azure_endpoint=s.azure_openai_endpoint,
                        api_version=s.azure_openai_api_version, timeout=s.llm_timeout_s, max_retries=3)
        return OpenAIChatProvider(c, s.azure_openai_deployment or s.llm_model, name="azure_openai")
    if p == "anthropic":
        from anthropic import Anthropic

        return AnthropicProvider(Anthropic(api_key=s.anthropic_api_key, timeout=s.llm_timeout_s, max_retries=3), s.llm_model)
    raise ValueError(f"unknown LLM provider {p}")


def new_call_id() -> str:
    return "call_" + uuid.uuid4().hex[:12]
