"""Subject V.5 / V.6 with the real Qwen/Qwen3-0.6B model.

Slow (a few minutes). Set ``SKIP_SLOW=1`` to skip.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import time
import unittest
from pathlib import Path
from typing import Any, Callable

from subject_fakes import (
    SAMPLE_FUNCTIONS,
    SAMPLE_PROMPTS,
    make_functions,
    sample_functions,
    schema_problems,
    write_json,
)

from src.models import FunctionDefinition
from src.pipeline import run_pipeline

SLOW = unittest.skipIf(os.environ.get("SKIP_SLOW") == "1", "SKIP_SLOW=1")

Check = Callable[[dict[str, Any]], bool]


def substitution(source: str, result: str) -> Check:
    """Check a regex call by applying it, not by comparing the pattern."""

    def check(call: dict[str, Any]) -> bool:
        params = call["parameters"]
        try:
            replaced = re.sub(
                params["regex"], params["replacement"], params["source_string"]
            )
        except (re.error, KeyError, TypeError):
            return False
        return bool(params["source_string"] == source and replaced == result)

    return check


def call_is(function: str, /, **expected: Any) -> Check:
    """Check the function name and each expected argument."""

    def check(call: dict[str, Any]) -> bool:
        return bool(call["name"] == function and all(
            call["parameters"].get(key) == value for key, value in expected.items()
        ))

    return check


SAMPLE_EXPECTED: dict[str, Check] = {
    "What is the sum of 2 and 3?": call_is("fn_add_numbers", a=2, b=3),
    "What is the sum of 265 and 345?": call_is("fn_add_numbers", a=265, b=345),
    "Greet shrek": call_is("fn_greet", name="shrek"),
    "Greet john": call_is("fn_greet", name="john"),
    "Reverse the string 'hello'": call_is("fn_reverse_string", s="hello"),
    "Reverse the string 'world'": call_is("fn_reverse_string", s="world"),
    "What is the square root of 16?": call_is("fn_get_square_root", a=16),
    "Calculate the square root of 144": call_is("fn_get_square_root", a=144),
    "Replace all numbers in \"Hello 34 I'm 233 years old\" with NUMBERS":
        substitution("Hello 34 I'm 233 years old",
                     "Hello NUMBERS I'm NUMBERS years old"),
    "Replace all vowels in 'Programming is fun' with asterisks":
        substitution("Programming is fun", "Pr*gr*mm*ng *s f*n"),
    "Substitute the word 'cat' with 'dog' in "
    "'The cat sat on the mat with another cat'":
        substitution("The cat sat on the mat with another cat",
                     "The dog sat on the mat with another dog"),
}

EDGE_FUNCTIONS: list[dict[str, Any]] = [
    {
        "name": "fn_set_light",
        "description": "Switch a lamp on or off and set its brightness.",
        "parameters": {"on": {"type": "boolean"}, "brightness": {"type": "number"}},
        "returns": {"type": "string"},
    },
    {
        "name": "fn_is_even",
        "description": "Tell whether an integer is even.",
        "parameters": {"n": {"type": "integer"}},
        "returns": {"type": "boolean"},
    },
]

EDGE_EXPECTED: dict[str, Check | None] = {
    "Greet José": call_is("fn_greet", name="José"),
    "What is the sum of 123456789012 and 987654321098?":
        call_is("fn_add_numbers", a=123456789012, b=987654321098),
    "Turn the light off and set brightness to 0":
        call_is("fn_set_light", on=False, brightness=0),
    "Is 42 an even number?": call_is("fn_is_even", n=42),
    "Reverse the string ''": None,
    "Reverse the string 'He said \"hi\"'": None,
    "Do the thing with 7": None,
}
"""``None`` = only the output rules are checked (ambiguous / odd prompts)."""


def run_real(
    functions_path: Path, prompts: list[str], tmp: Path
) -> tuple[list[dict[str, Any]], float]:
    """Run the real pipeline and return (output objects, seconds)."""
    prompts_path = write_json(tmp / "prompts.json", [{"prompt": p} for p in prompts])
    output = tmp / "function_calling_results.json"
    start = time.perf_counter()
    run_pipeline(functions_path, prompts_path, output)
    elapsed = time.perf_counter() - start
    with output.open(encoding="utf-8") as handle:
        data: list[dict[str, Any]] = json.load(handle)
    return data, elapsed


@SLOW
class RealModelSampleTest(unittest.TestCase):
    """V.5 Performance and Reliability on the shipped test prompts."""

    calls: list[dict[str, Any]]
    elapsed: float

    @classmethod
    def setUpClass(cls) -> None:
        """Run the shipped prompts through the real model once."""
        scratch = tempfile.TemporaryDirectory()
        cls.addClassCleanup(scratch.cleanup)
        with SAMPLE_PROMPTS.open(encoding="utf-8") as handle:
            prompts = [item["prompt"] for item in json.load(handle)]
        cls.calls, cls.elapsed = run_real(
            SAMPLE_FUNCTIONS, prompts, Path(scratch.name)
        )

    def test_under_five_minutes(self) -> None:
        """All test prompts processed in under 5 minutes (model load included)."""
        self.assertLess(self.elapsed, 300)

    def test_every_output_is_schema_valid(self) -> None:
        """100% valid JSON, schema-compliant."""
        functions = sample_functions()
        problems = {
            call["prompt"]: schema_problems(call, functions) for call in self.calls
        }
        self.assertEqual({k: v for k, v in problems.items() if v}, {})

    def test_accuracy_at_least_90_percent(self) -> None:
        """90%+ correct function selection and argument extraction."""
        wrong = [
            call for call in self.calls
            if not SAMPLE_EXPECTED[call["prompt"]](call)
        ]
        accuracy = 1 - len(wrong) / len(self.calls)
        self.assertGreaterEqual(accuracy, 0.9, json.dumps(wrong, indent=1))


@SLOW
class RealModelEdgeCaseTest(unittest.TestCase):
    """V.6 edge cases: special characters, large numbers, booleans,
    multi-parameter and integer functions, empty strings, ambiguity."""

    calls: dict[str, dict[str, Any]]
    functions: list[FunctionDefinition]

    @classmethod
    def setUpClass(cls) -> None:
        """Run the edge prompts with the sample functions plus two more."""
        scratch = tempfile.TemporaryDirectory()
        cls.addClassCleanup(scratch.cleanup)
        tmp = Path(scratch.name)
        with SAMPLE_FUNCTIONS.open(encoding="utf-8") as handle:
            raw = json.load(handle) + EDGE_FUNCTIONS
        functions_path = write_json(tmp / "functions.json", raw)
        cls.functions = make_functions(raw)
        data, _ = run_real(functions_path, list(EDGE_EXPECTED), tmp)
        cls.calls = {call["prompt"]: call for call in data}

    def test_every_output_is_schema_valid(self) -> None:
        """Even odd prompts produce schema-valid calls."""
        problems = {
            prompt: schema_problems(call, self.functions)
            for prompt, call in self.calls.items()
        }
        self.assertEqual({k: v for k, v in problems.items() if v}, {})

    def test_special_characters(self) -> None:
        """Non-ASCII names come back intact."""
        prompt = "Greet José"
        check = EDGE_EXPECTED[prompt]
        assert check is not None
        self.assertTrue(check(self.calls[prompt]), self.calls[prompt])

    def test_large_numbers(self) -> None:
        """Twelve-digit operands are copied exactly."""
        prompt = "What is the sum of 123456789012 and 987654321098?"
        check = EDGE_EXPECTED[prompt]
        assert check is not None
        self.assertTrue(check(self.calls[prompt]), self.calls[prompt])

    def test_boolean_and_number_parameters(self) -> None:
        """Function with several parameters of different types."""
        prompt = "Turn the light off and set brightness to 0"
        check = EDGE_EXPECTED[prompt]
        assert check is not None
        self.assertTrue(check(self.calls[prompt]), self.calls[prompt])

    def test_integer_parameter(self) -> None:
        """``integer`` parameters come back as JSON integers."""
        prompt = "Is 42 an even number?"
        check = EDGE_EXPECTED[prompt]
        assert check is not None
        self.assertTrue(check(self.calls[prompt]), self.calls[prompt])


if __name__ == "__main__":
    unittest.main()
