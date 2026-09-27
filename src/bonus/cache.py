"""Caching and batched candidate checks for constrained decoding."""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, ConfigDict

from src.bonus.nested import NestedCallChecker


class CandidateBatch(BaseModel):
    """A slice of token ids checked together before a full vocab scan."""

    model_config = ConfigDict(extra="forbid")

    token_ids: list[int]
    width: int


class ConstraintCache:
    """Remember whether a token text keeps a prefix legal."""

    def __init__(self) -> None:
        """Create an empty cache."""
        self._ok: dict[tuple[str, str], bool] = {}
        self.hits = 0
        self.misses = 0

    def allows(self, checker: NestedCallChecker, prefix: str, piece: str) -> bool:
        """Return True when ``prefix + piece`` is still a valid call prefix.

        Args:
            checker: Schema checker.
            prefix: JSON generated so far.
            piece: Candidate token text.

        Returns:
            Whether the candidate is legal.
        """
        key = (prefix, piece)
        cached = self._ok.get(key)
        if cached is not None:
            self.hits += 1
            return cached
        self.misses += 1
        allowed = checker.check(prefix + piece).ok
        self._ok[key] = allowed
        return allowed


def top_batch(scores: np.ndarray, width: int) -> CandidateBatch:
    """Take the ``width`` highest-logit ids as one batch.

    Args:
        scores: Logit vector.
        width: How many ids to keep.

    Returns:
        Ids sorted from best logit to worst inside the batch.
    """
    width = min(width, int(scores.shape[0]))
    if width <= 0:
        return CandidateBatch(token_ids=[], width=0)
    picked = np.argpartition(-scores, width - 1)[:width]
    ordered = picked[np.argsort(-scores[picked])]
    ids = [int(token_id) for token_id in ordered]
    return CandidateBatch(token_ids=ids, width=width)


def legal_ids(
    scores: np.ndarray,
    prefix: str,
    checker: NestedCallChecker,
    id_to_text: dict[int, str],
    cache: ConstraintCache,
    *,
    limit: int,
    batch_width: int = 32,
) -> list[tuple[int, float]]:
    """Return up to ``limit`` legal ``(token_id, logit)`` pairs, best first.

    The first batch is the top logits. Only if that batch is not enough
    does the search walk the rest of the vocabulary.

    Args:
        scores: Next-token logits.
        prefix: Text generated so far.
        checker: Nested-aware prefix checker.
        id_to_text: Token id to decoded text.
        cache: Prefix/token legality cache.
        limit: Maximum alternatives to return (used by backtracking).
        batch_width: Size of the first candidate batch.

    Returns:
        Legal token ids paired with their logits.
    """
    found: list[tuple[int, float]] = []
    seen: set[int] = set()

    def consider(token_id: int) -> None:
        if token_id in seen or len(found) >= limit:
            return
        seen.add(token_id)
        piece = id_to_text.get(token_id)
        if piece is None:
            return
        if cache.allows(checker, prefix, piece):
            found.append((token_id, float(scores[token_id])))

    for token_id in top_batch(scores, batch_width).token_ids:
        consider(token_id)
        if len(found) >= limit:
            return found

    for token_id in np.argsort(-scores):
        consider(int(token_id))
        if len(found) >= limit:
            break
    return found
