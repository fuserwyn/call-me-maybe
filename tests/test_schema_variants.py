"""Function sets that differ from the sample: enum, missing optional keys."""

from __future__ import annotations

import unittest

from src.constraints import FunctionCallPrefixChecker
from src.models import FunctionDefinition
from src.prompt_builder import build_prompt


def definitions() -> list[FunctionDefinition]:
    """A small set using enum and leaving out description/returns."""
    raw = [
        {
            "name": "set_firmware",
            "description": "Install a firmware version.",
            "parameters": {
                "firmware": {"type": "string", "enum": ["v1.0", "v1.0-rc"]},
            },
        },
        {
            "name": "set_level",
            "parameters": {"level": {"type": "integer", "enum": [1, 10]}},
        },
        {"name": "ping"},
    ]
    return [FunctionDefinition.model_validate(item) for item in raw]


class SchemaVariantsTest(unittest.TestCase):
    """The loader and the checker accept other valid function sets."""

    def setUp(self) -> None:
        """Build a checker for the variant definitions."""
        self.checker = FunctionCallPrefixChecker(definitions())

    def test_optional_keys_may_be_missing(self) -> None:
        """description, parameters and returns are optional."""
        ping = definitions()[2]
        self.assertEqual(ping.description, "")
        self.assertEqual(ping.parameters, {})
        self.assertIsNone(ping.returns)

    def test_enum_accepts_only_listed_values(self) -> None:
        """A value outside the enum is rejected as soon as it diverges."""
        head = '{"name":"set_firmware","parameters":{"firmware":'
        self.assertTrue(self.checker.check(head + '"v1.0"}}').complete)
        self.assertTrue(self.checker.check(head + '"v1.0-rc"}}').complete)
        self.assertTrue(self.checker.check(head + '"v1.0-').ok)
        self.assertFalse(self.checker.check(head + '"v2').ok)

    def test_numeric_enum_prefers_longest_value(self) -> None:
        """10 is not read as 1 followed by an invalid 0."""
        head = '{"name":"set_level","parameters":{"level":'
        self.assertTrue(self.checker.check(head + "10}}").complete)
        self.assertTrue(self.checker.check(head + "1}}").complete)
        self.assertFalse(self.checker.check(head + "5").ok)

    def test_function_without_parameters(self) -> None:
        """A call with an empty parameters object is complete."""
        self.assertTrue(
            self.checker.check('{"name":"ping","parameters":{}}').complete
        )

    def test_prompt_lists_allowed_values(self) -> None:
        """The prompt tells the model which values are allowed."""
        prompt = build_prompt("update", definitions())
        self.assertIn('one of "v1.0", "v1.0-rc"', prompt)


if __name__ == "__main__":
    unittest.main()
