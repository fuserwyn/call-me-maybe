"""Subject V.4 warning: "Do not hardcode solutions based on the provided
examples" — the input files change during peer review."""

from __future__ import annotations

import json
import unittest

from subject_fakes import ROOT, SAMPLE_FUNCTIONS, make_functions

from src.prompt_builder import build_prompt

OTHER_FUNCTIONS = make_functions([
    {
        "name": "fn_convert_currency",
        "description": "Convert an amount between two currencies.",
        "parameters": {
            "amount": {"type": "number"},
            "from_currency": {"type": "string"},
            "to_currency": {"type": "string"},
        },
        "returns": {"type": "number"},
    },
])

SAMPLE_ANSWER_FRAGMENTS = [
    "[aeiouAEIOU]",
    "NUMBERS",
    "shrek",
    "Programming is fun",
    "The cat sat on the mat",
    "Hello 34",
]
"""Pieces of the expected answers to the shipped data/input prompts."""


class NoHardcodingTest(unittest.TestCase):
    """The solution must not depend on the shipped example files."""

    def test_prompt_only_mentions_the_given_functions(self) -> None:
        """With a different function set, the prompt never names the sample
        functions (they may not exist during the review)."""
        with SAMPLE_FUNCTIONS.open(encoding="utf-8") as handle:
            sample_names = [item["name"] for item in json.load(handle)]
        prompt = build_prompt("Convert 10 EUR to USD", OTHER_FUNCTIONS)
        leaked = [name for name in sample_names if name in prompt]
        self.assertEqual(leaked, [], "sample function names in the prompt")

    def test_source_has_no_sample_answers(self) -> None:
        """No answer to a shipped test prompt is written into src/."""
        found = [
            f"{path.relative_to(ROOT)}: {fragment}"
            for path in sorted((ROOT / "src").rglob("*.py"))
            for fragment in SAMPLE_ANSWER_FRAGMENTS
            if fragment in path.read_text(encoding="utf-8")
        ]
        self.assertEqual(found, [])

    def test_no_function_name_dispatch_in_source(self) -> None:
        """No sample function name appears in src/ outside comments/docs,
        i.e. nothing branches on a specific function."""
        with SAMPLE_FUNCTIONS.open(encoding="utf-8") as handle:
            sample_names = [item["name"] for item in json.load(handle)]
        found = [
            f"{path.relative_to(ROOT)}: {name}"
            for path in sorted((ROOT / "src").rglob("*.py"))
            if path.name != "prompt_builder.py"
            for name in sample_names
            if name in path.read_text(encoding="utf-8")
        ]
        self.assertEqual(found, [])


if __name__ == "__main__":
    unittest.main()
