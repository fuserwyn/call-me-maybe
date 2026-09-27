"""Step-by-step trace of constrained generation."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class GenerationStep(BaseModel):
    """One accepted token during decoding."""

    model_config = ConfigDict(extra="forbid")

    token_id: int
    text: str
    logit: float
    backtracked: bool = False


class PromptTrace(BaseModel):
    """Full trace for a single user prompt."""

    model_config = ConfigDict(extra="forbid")

    prompt: str
    model_name: str
    steps: list[GenerationStep] = Field(default_factory=list)
    output: str = ""
    backtracks: int = 0


class GenerationTrace(BaseModel):
    """Trace file written when visualization is enabled."""

    model_config = ConfigDict(extra="forbid")

    prompts: list[PromptTrace] = Field(default_factory=list)

    def save(self, path: Path) -> None:
        """Write the trace as JSON.

        Args:
            path: Destination file. Parent directories are created.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(self.model_dump(), handle, indent=2, ensure_ascii=False)
            handle.write("\n")


def format_step(step: GenerationStep, so_far: str) -> str:
    """Render one generation step for the terminal.

    Args:
        step: Accepted token.
        so_far: JSON text after this token.

    Returns:
        Single-line description.
    """
    flag = " backtrack" if step.backtracked else ""
    snippet = so_far.replace("\n", "\\n")
    if len(snippet) > 72:
        snippet = "…" + snippet[-72:]
    return (
        f"  id={step.token_id:<6} logit={step.logit:7.2f}{flag}  "
        f"{snippet}"
    )
