"""Backtracking budget when a greedy token leads to a dead end."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class BacktrackBudget(BaseModel):
    """How many times generation may abandon a chosen token."""

    model_config = ConfigDict(extra="forbid")

    max_alternatives: int = Field(default=3, ge=1)
    max_backtracks: int = Field(default=24, ge=0)
    used: int = 0

    def spend(self) -> bool:
        """Consume one backtrack.

        Returns:
            True if another backtrack is still allowed.
        """
        if self.used >= self.max_backtracks:
            return False
        self.used += 1
        return True


class SearchFrame(BaseModel):
    """One prefix plus the legal next tokens still left to try."""

    model_config = ConfigDict(extra="forbid")

    token_ids: list[int]
    text: str
    options: list[int]
    logits: dict[int, float] = Field(default_factory=dict)
    cursor: int = 0

    def take(self) -> int | None:
        """Pop the next alternative id.

        Returns:
            Token id, or None when this frame is exhausted.
        """
        if self.cursor >= len(self.options):
            return None
        token_id = self.options[self.cursor]
        self.cursor += 1
        return token_id
