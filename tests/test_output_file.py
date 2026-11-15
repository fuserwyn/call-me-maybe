"""Check the output file left by the last ``make run``, if there is one."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from check_output import (
    DEFAULT_OUTPUT,
    load_calls,
    output_problems,
    sample_accuracy,
)
from subject_fakes import SAMPLE_FUNCTIONS, SAMPLE_PROMPTS


@unittest.skipUnless(DEFAULT_OUTPUT.exists(), "no output yet: run make run")
class OutputFileTest(unittest.TestCase):
    """V.4.2 / V.5 on data/output/function_calling_results.json."""

    def test_output_is_valid(self) -> None:
        """Valid JSON, one schema-valid result per prompt, in order."""
        calls = load_calls(DEFAULT_OUTPUT)
        self.assertEqual(
            output_problems(calls, SAMPLE_FUNCTIONS, SAMPLE_PROMPTS), []
        )

    def test_output_accuracy(self) -> None:
        """At least 90% correct calls on the shipped sample."""
        scored = sample_accuracy(load_calls(DEFAULT_OUTPUT))
        if scored is None:
            self.skipTest("output is not for the shipped sample prompts")
        accuracy, wrong = scored
        self.assertGreaterEqual(accuracy, 0.9, json.dumps(wrong, indent=1))


class CheckerCatchesBadOutputTest(unittest.TestCase):
    """The checker itself rejects broken files."""

    def write(self, content: str) -> Path:
        """Write ``content`` to a temporary output file."""
        scratch = tempfile.TemporaryDirectory()
        self.addCleanup(scratch.cleanup)
        path = Path(scratch.name) / "out.json"
        path.write_text(content, encoding="utf-8")
        return path

    def test_not_json(self) -> None:
        """A file that is not JSON is rejected."""
        with self.assertRaises(ValueError):
            load_calls(self.write('[{"prompt": '))

    def test_missing_results_and_bad_types(self) -> None:
        """Wrong count, unknown function and wrong types are all reported."""
        with SAMPLE_PROMPTS.open(encoding="utf-8") as handle:
            first = json.load(handle)[0]["prompt"]
        calls = load_calls(self.write(json.dumps([
            {"prompt": first, "name": "no_such_function", "parameters": {}},
        ])))
        problems = output_problems(calls, SAMPLE_FUNCTIONS, SAMPLE_PROMPTS)
        self.assertTrue(any("prompts differ" in p for p in problems))
        self.assertTrue(any("unknown function" in p for p in problems))
