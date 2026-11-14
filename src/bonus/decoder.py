"""Constrained decoding that uses the recoded tokenizer.

Encoding of the prompt and decoding of each token go through
:class:`BonusTokenizer`. The model is only asked for logits via
``get_logits_from_input_ids``.
"""

from __future__ import annotations

import sys
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, PrivateAttr

from src.bonus.cache import ConstraintCache, legal_ids
from src.bonus.nested import NestedCallChecker, repair_json
from src.bonus.recovery import BacktrackBudget, SearchFrame
from src.bonus.tokenizer import BonusTokenizer
from src.bonus.visualize import GenerationStep, PromptTrace, format_step
from src.models import FunctionCallResult, FunctionDefinition
from src.prompt_builder import build_prompt

_MAX_NEW_TOKENS = 192


class BonusDecoder(BaseModel):
    """Function calling with a hand-rolled tokenizer and backtracking."""

    model_config = ConfigDict(extra="forbid")

    llm: Any
    tokenizer: BonusTokenizer
    functions: list[FunctionDefinition]
    model_name: str
    visualize: bool = False
    _checker: NestedCallChecker = PrivateAttr()
    _id_to_text: dict[int, str] = PrivateAttr(default_factory=dict)
    _cache: ConstraintCache = PrivateAttr(default_factory=ConstraintCache)

    def __init__(
        self,
        model: Any,
        tokenizer: BonusTokenizer,
        functions: list[FunctionDefinition],
        **data: Any,
    ) -> None:
        """Store the model, tokenizer, and schema.

        Args:
            model: SDK model. Only logits and file paths are used.
            tokenizer: Recoded BPE tokenizer.
            functions: Tools the call may invoke, including nested schemas.
            **data: ``model_name`` (recorded in the trace) and
                ``visualize`` (print each accepted token).
        """
        super().__init__(
            llm=model, tokenizer=tokenizer, functions=functions, **data
        )

    def model_post_init(self, context: Any, /) -> None:
        """Build the checker and the id → text table.

        Args:
            context: Pydantic validation context (unused).
        """
        self._checker = NestedCallChecker(self.functions)
        self._id_to_text = self.tokenizer.id_to_text_map()

    def generate(self, prompt: str) -> tuple[FunctionCallResult, PromptTrace]:
        """Generate one function call and a visualization trace.

        Args:
            prompt: Natural-language user request.

        Returns:
            The structured call and the token trace.
        """
        trace = PromptTrace(prompt=prompt, model_name=self.model_name)
        prompt_ids = self.tokenizer.encode(build_prompt(prompt, self.functions))
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
            new_text = self.tokenizer.decode(new_ids)
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
                if self.visualize:
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
            if self.visualize:
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
            self.llm.get_logits_from_input_ids(input_ids),
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
