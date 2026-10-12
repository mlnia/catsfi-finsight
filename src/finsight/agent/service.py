"""Agent facade used by the API: run a turn, surface pending approvals, resume after review."""

from __future__ import annotations

import logging
import threading
import time
import uuid

from langgraph.types import Command
from pydantic import BaseModel, Field

from finsight.config import Settings
from finsight.guardrails.pii import mask_pii
from finsight.observability.metrics import PENDING_APPROVALS
from finsight.observability.tracing import current_trace_id, log_event, span

log = logging.getLogger(__name__)


class AgentResult(BaseModel):
    thread_id: str
    status: str  # completed | awaiting_approval | blocked
    answer: str | None = None
    pending_approval: list[dict] = Field(default_factory=list)
    tool_trace: list[dict] = Field(default_factory=list)
    guardrails: dict = Field(default_factory=dict)
    usage: dict = Field(default_factory=dict)
    latency_ms: float = 0.0
    trace_id: str | None = None


class AgentService:
    def __init__(self, graph, settings: Settings):
        self.graph, self.s = graph, settings
        self._pending: dict[str, dict] = {}
        self._lock = threading.Lock()

    def _cfg(self, thread_id: str) -> dict:
        return {"configurable": {"thread_id": thread_id}, "recursion_limit": 4 * self.s.agent_max_steps + 10}

    def run(self, question: str, thread_id: str | None = None) -> AgentResult:
        thread_id = thread_id or uuid.uuid4().hex
        if self.pending_for(thread_id):
            raise ValueError("thread is waiting for human approval; resume it first")
        masked, pii = mask_pii(question.strip()) if self.s.mask_pii else (question.strip(), [])
        t0 = time.perf_counter()
        with span("agent.run", {"thread_id": thread_id}):
            state = self.graph.invoke({"messages": [{"role": "user", "content": masked}], "guard": {"pii_masked": pii},
                                       "usage": {}, "trace": []}, self._cfg(thread_id))
            return self._result(thread_id, state, t0)

    def resume(self, thread_id: str, approved: bool, reviewer: str = "unknown", reason: str | None = None) -> AgentResult:
        if not self.pending_for(thread_id):
            raise KeyError("no pending approval for this thread")
        t0 = time.perf_counter()
        with span("agent.resume", {"thread_id": thread_id, "approved": approved, "reviewer": reviewer}):
            log_event(log, "approval decision", thread_id=thread_id, approved=approved, reviewer=reviewer, reason=reason)
            state = self.graph.invoke(Command(resume={"approved": approved, "reviewer": reviewer, "reason": reason}),
                                      self._cfg(thread_id))
            return self._result(thread_id, state, t0)

    def pending_for(self, thread_id: str) -> dict | None:
        with self._lock:
            return self._pending.get(thread_id)

    def list_pending(self) -> list[dict]:
        with self._lock:
            return [{"thread_id": k, **v} for k, v in self._pending.items()]

    def _result(self, thread_id: str, state: dict, t0: float) -> AgentResult:
        interrupts = state.get("__interrupt__") or []
        base = dict(thread_id=thread_id, tool_trace=state.get("trace", []), usage=state.get("usage", {}),
                    latency_ms=round((time.perf_counter() - t0) * 1000, 1), trace_id=current_trace_id(),
                    guardrails=state.get("guard", {}))
        with self._lock:
            if interrupts:
                payload = interrupts[0].value
                self._pending[thread_id] = {"tool_calls": payload.get("tool_calls", []), "since": time.time()}
            else:
                self._pending.pop(thread_id, None)
            PENDING_APPROVALS.set(len(self._pending))
        if interrupts:
            return AgentResult(status="awaiting_approval", pending_approval=self._pending[thread_id]["tool_calls"], **base)
        final = state.get("final") or {}
        guard = final.get("guardrails", base["guardrails"])
        base["guardrails"] = guard
        return AgentResult(status="blocked" if guard.get("action") == "blocked" else "completed",
                           answer=final.get("answer"), **base)
