"""Constrained decoding that uses the recoded tokenizer.

Encoding of the prompt and decoding of each token go through
:class:`BonusTokenizer`. The model is only asked for logits via
``get_logits_from_input_ids``.
"""

from __future__ import annotations

import sys
from typing import Any

import numpy as np
from llm_sdk import Small_LLM_Model  # type: ignore[attr-defined]

from src.bonus.cache import ConstraintCache, legal_ids
from src.bonus.nested import NestedCallChecker, repair_json
from src.bonus.recovery import BacktrackBudget, SearchFrame
from src.bonus.tokenizer import BonusTokenizer
from src.bonus.visualize import GenerationStep, PromptTrace, format_step
from src.models import FunctionCallResult, FunctionDefinition
from src.prompt_builder import build_prompt

_MAX_NEW_TOKENS = 192


class BonusDecoder:
    """Function calling with a hand-rolled tokenizer and backtracking."""

    def __init__(
        self,
        model: Small_LLM_Model,
        tokenizer: BonusTokenizer,
        functions: list[FunctionDefinition],
        *,
        model_name: str,
        visualize: bool = False,
    ) -> None:
        """Store the model, tokenizer, and schema.

        Args:
            model: SDK model. Only logits and file paths are used.
            tokenizer: Recoded BPE tokenizer.
            functions: Tools the call may invoke, including nested schemas.
            model_name: Hugging Face id, recorded in the trace.
            visualize: Print each accepted token.
        """
        self._model = model
        self._tokenizer = tokenizer
        self._functions = functions
        self._model_name = model_name
        self._visualize = visualize
        self._checker = NestedCallChecker(functions)
        self._id_to_text = tokenizer.id_to_text_map()
        self._cache = ConstraintCache()

    def generate(self, prompt: str) -> tuple[FunctionCallResult, PromptTrace]:
        """Generate one function call and a visualization trace.

        Args:
            prompt: Natural-language user request.

        Returns:
            The structured call and the token trace.
        """
        trace = PromptTrace(prompt=prompt, model_name=self._model_name)
        prompt_ids = self._tokenizer.encode(build_prompt(prompt, self._functions))
        budget = BacktrackBudget()
        root_options = self._options(prompt_ids, "", budget.max_alternatives)
        stack: list[SearchFrame] = [
            SearchFrame(
                token_ids=[],
                text="",
                options=[token_id for token_id, _logit in root_options],
                logits={token_id: logit for token_id, logit in root_options},
            )
        ]
        steps: list[GenerationStep] = []

        while stack:
            frame = stack[-1]
            if len(frame.token_ids) >= _MAX_NEW_TOKENS:
                break
            choice = frame.take()
            if choice is None:
                if len(stack) == 1 or not budget.spend():
                    break
                stack.pop()
                if steps:
                    steps.pop()
                continue

            new_ids = frame.token_ids + [choice]
            new_text = self._tokenizer.decode(new_ids)
            checked = self._checker.check(new_text)
            if not checked.ok:
                continue
            step = GenerationStep(
                token_id=choice,
                text=self._id_to_text.get(choice, ""),
                logit=frame.logits.get(choice, 0.0),
                backtracked=frame.cursor > 1,
            )
            if checked.complete and checked.parsed is not None:
                steps.append(step)
                if self._visualize:
                    print(format_step(step, new_text))
                trace.steps = steps
                trace.output = new_text
                trace.backtracks = budget.used
                return self._to_result(prompt, checked.parsed), trace

            options = self._options(
                prompt_ids + new_ids,
                new_text,
                budget.max_alternatives,
            )
            if not options:
                if not budget.spend():
                    break
                continue
            if self._visualize:
                print(format_step(step, new_text))
            steps.append(step)
            stack.append(
                SearchFrame(
                    token_ids=new_ids,
                    text=new_text,
                    options=[token_id for token_id, _logit in options],
                    logits={token_id: logit for token_id, logit in options},
                )
            )

        trace.steps = steps
        trace.backtracks = budget.used
        partial = stack[-1].text if stack else ""
        repaired = repair_json(partial)
        recovered = self._checker.check(repaired)
        if recovered.complete and recovered.parsed is not None:
            print(f"Recovered truncated JSON for {prompt!r}", file=sys.stderr)
            trace.output = repaired
            return self._to_result(prompt, recovered.parsed), trace

        print(f"Error: bonus decoder failed for {prompt!r}", file=sys.stderr)
        return FunctionCallResult(prompt=prompt, name="", parameters={}), trace

    def _options(
        self,
        input_ids: list[int],
        prefix: str,
        limit: int,
    ) -> list[tuple[int, float]]:
        """Legal next tokens, best logit first. One logits call."""
        scores = np.asarray(
            self._model.get_logits_from_input_ids(input_ids),
            dtype=np.float64,
        )
        return legal_ids(
            scores,
            prefix,
            self._checker,
            self._id_to_text,
            self._cache,
            limit=limit,
        )

    @staticmethod
    def _to_result(prompt: str, parsed: dict[str, Any]) -> FunctionCallResult:
        """Wrap a parsed object as a function-call result."""
        name = str(parsed.get("name", ""))
        parameters = parsed.get("parameters", {})
        if not isinstance(parameters, dict):
            parameters = {}
        return FunctionCallResult(prompt=prompt, name=name, parameters=parameters)
