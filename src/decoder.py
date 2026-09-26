"""Constrained decoding for schema-valid JSON function calls."""

from __future__ import annotations

import json
import sys
from typing import Any

import numpy as np
from llm_sdk import Small_LLM_Model  # type: ignore[attr-defined]

from src.constraints import FunctionCallPrefixChecker
from src.models import FunctionCallResult, FunctionDefinition
from src.prompt_builder import build_prompt
from src.vocab_utils import build_id_to_text, load_vocab

_MAX_NEW_TOKENS = 192


class ConstrainedDecoder:
    """Generate function calls using constrained decoding over model logits."""

    def __init__(
        self,
        model: Small_LLM_Model,
        functions: list[FunctionDefinition],
    ) -> None:
        """Initialize the decoder.

        Args:
            model: Wrapped LLM from llm_sdk (public API only).
            functions: Available function definitions for schema constraints.
        """
        self._model = model
        self._functions = functions
        self._checker = FunctionCallPrefixChecker(functions)
        vocab = load_vocab(model.get_path_to_vocab_file())
        self._id_to_text = build_id_to_text(vocab)

    def generate(self, prompt: str) -> FunctionCallResult:
        """Map a natural-language prompt to a structured function call.

        Args:
            prompt: User request in natural language.

        Returns:
            Structured function call with name and typed parameters.
        """
        full_prompt = build_prompt(prompt, self._functions)
        prompt_ids = self._model.encode(full_prompt).tolist()[0]
        generated = ""
        generated_ids: list[int] = []

        for _ in range(_MAX_NEW_TOKENS):
            input_ids = prompt_ids + generated_ids
            logits = self._model.get_logits_from_input_ids(input_ids)
            next_id = self._select_token(logits, generated)
            if next_id is None:
                print(
                    f"Warning: no valid tokens for prompt {prompt!r} "
                    f"at {generated!r}",
                    file=sys.stderr,
                )
                break

            piece = self._id_to_text.get(next_id, "")
            generated_ids.append(next_id)
            generated += piece

            result = self._checker.check(generated)
            if result.complete and result.parsed is not None:
                return self._to_result(prompt, result.parsed)

        # Fallback: try to parse whatever we have
        try:
            parsed = json.loads(generated)
            return self._to_result(prompt, parsed)
        except (json.JSONDecodeError, TypeError, KeyError, ValueError):
            print(
                f"Error: failed to produce valid JSON for {prompt!r}. "
                f"Got: {generated!r}",
                file=sys.stderr,
            )
            return FunctionCallResult(prompt=prompt, name="", parameters={})

    def _select_token(self, logits: list[float], generated: str) -> int | None:
        """Mask invalid tokens and return the argmax among allowed ones.

        Walks candidate ids in descending logit order so the first valid
        token is already the constrained argmax (fast when the model
        already prefers a legal continuation).

        Args:
            logits: Raw next-token scores from the model.
            generated: JSON text generated so far.

        Returns:
            Selected token id, or None if nothing is allowed.
        """
        scores = np.asarray(logits, dtype=np.float64)
        # Highest scores first — first legal hit == constrained argmax.
        for token_id in np.argsort(-scores):
            tid = int(token_id)
            text = self._id_to_text.get(tid)
            if text is None:
                continue
            if self._checker.check(generated + text).ok:
                return tid
        return None

    @staticmethod
    def _to_result(prompt: str, parsed: dict[str, Any]) -> FunctionCallResult:
        """Convert a parsed JSON object into FunctionCallResult.

        Args:
            prompt: Original user prompt.
            parsed: Parsed JSON dictionary.

        Returns:
            Validated FunctionCallResult (best-effort on types).
        """
        name = str(parsed.get("name", ""))
        parameters = parsed.get("parameters", {})
        if not isinstance(parameters, dict):
            parameters = {}
        return FunctionCallResult(prompt=prompt, name=name, parameters=parameters)
