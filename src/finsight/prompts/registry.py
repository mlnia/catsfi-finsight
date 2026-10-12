"""Versioned prompt templates.

Templates live in `templates/<name>.yaml` with one entry per version. Every rendered prompt carries
`name@version` plus a content hash, which is attached to traces, API responses and evaluation
reports - so a quality regression can always be tied to the exact prompt text that caused it.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from string import Template

import yaml

TEMPLATE_DIR = Path(__file__).parent / "templates"


@dataclass(frozen=True)
class PromptTemplate:
    name: str
    version: str
    description: str
    system: str
    user: str

    @property
    def ref(self) -> str:
        return f"{self.name}@{self.version}"

    @property
    def sha(self) -> str:
        return hashlib.sha256((self.system + "\x00" + self.user).encode()).hexdigest()[:12]

    def render(self, **vars) -> list[dict]:
        msgs = [{"role": "system", "content": Template(self.system).safe_substitute(**vars).strip()}]
        if self.user:
            msgs.append({"role": "user", "content": Template(self.user).safe_substitute(**vars).strip()})
        return msgs


@lru_cache
def _load(name: str) -> dict[str, PromptTemplate]:
    raw = yaml.safe_load((TEMPLATE_DIR / f"{name}.yaml").read_text(encoding="utf-8"))
    return {
        str(v): PromptTemplate(name=name, version=str(v), description=d.get("description", ""),
                               system=d["system"], user=d.get("user", ""))
        for v, d in raw["versions"].items()
    }


def get_prompt(ref: str) -> PromptTemplate:
    """`rag_answer@v2` -> that version; `rag_answer` -> latest version."""
    name, _, version = ref.partition("@")
    versions = _load(name)
    if not version:
        version = sorted(versions, key=lambda v: int(v.lstrip("v")))[-1]
    if version not in versions:
        raise KeyError(f"prompt {name}@{version} not found; available: {sorted(versions)}")
    return versions[version]


def list_prompts() -> list[dict]:
    out = []
    for f in sorted(TEMPLATE_DIR.glob("*.yaml")):
        for p in _load(f.stem).values():
            out.append({"ref": p.ref, "sha": p.sha, "description": p.description})
    return out
