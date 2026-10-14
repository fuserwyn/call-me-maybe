"""Subject V.1 / V.3 / V.4 / V.6: constrained decoding with a scripted model.

``ScriptedModel`` stands in for Qwen so every rule can be checked quickly
and deterministically. ``test_subject_v_real_model.py`` runs the real model.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

from subject_fakes import (
    EXTRA_IDS,
    SAMPLE_FUNCTIONS,
    ScriptedModel,
    make_functions,
    sample_functions,
    schema_problems,
    write_json,
)

from src.decoder import ConstrainedDecoder
from src.models import FunctionDefinition
from src.pipeline import run_pipeline

EXTRA_FUNCTIONS: list[dict[str, Any]] = [
    {
        "name": "fn_set_light",
        "description": "Switch a lamp and set its brightness.",
        "parameters": {"on": {"type": "boolean"}, "brightness": {"type": "number"}},
        "returns": {"type": "string"},
    },
    {
        "name": "fn_is_even",
        "description": "Tell whether an integer is even.",
        "parameters": {"n": {"type": "integer"}},
        "returns": {"type": "boolean"},
    },
    {
        "name": "fn_ping",
        "description": "Check that the service answers.",
        "parameters": {},
        "returns": {"type": "string"},
    },
]

SAMPLE_ANSWERS = {
    "What is the sum of 2 and 3?":
        '{"name":"fn_add_numbers","parameters":{"a":2.0,"b":3.0}}',
    "What is the sum of 265 and 345?":
        '{"name":"fn_add_numbers","parameters":{"a":265,"b":345}}',
    "Greet shrek": '{"name":"fn_greet","parameters":{"name":"shrek"}}',
    "Greet john": '{"name":"fn_greet","parameters":{"name":"john"}}',
    "Reverse the string 'hello'":
        '{"name":"fn_reverse_string","parameters":{"s":"hello"}}',
    "Reverse the string 'world'":
        '{"name":"fn_reverse_string","parameters":{"s":"world"}}',
    "What is the square root of 16?":
        '{"name":"fn_get_square_root","parameters":{"a":16}}',
    "Calculate the square root of 144":
        '{"name":"fn_get_square_root","parameters":{"a":144}}',
    "Replace all numbers in \"Hello 34 I'm 233 years old\" with NUMBERS":
        '{"name":"fn_substitute_string_with_regex","parameters":{'
        '"source_string":"Hello 34 I\'m 233 years old","regex":"[0-9]+",'
        '"replacement":"NUMBERS"}}',
    "Replace all vowels in 'Programming is fun' with asterisks":
        '{"name":"fn_substitute_string_with_regex","parameters":{'
        '"source_string":"Programming is fun","regex":"[aeiouAEIOU]",'
        '"replacement":"*"}}',
    "Substitute the word 'cat' with 'dog' in "
    "'The cat sat on the mat with another cat'":
        '{"name":"fn_substitute_string_with_regex","parameters":{'
        '"source_string":"The cat sat on the mat with another cat",'
        '"regex":"cat","replacement":"dog"}}',
}


class DecodingCase(unittest.TestCase):
    """Shared helpers: run the mandatory decoder over a scripted model."""

    def setUp(self) -> None:
        """Create a scratch directory and the extended function set."""
        scratch = tempfile.TemporaryDirectory()
        self.addCleanup(scratch.cleanup)
        self.tmp = Path(scratch.name)
        self.functions: list[FunctionDefinition] = (
            sample_functions() + make_functions(EXTRA_FUNCTIONS)
        )

    def decode(
        self,
        answers: dict[str, str],
        distractors: dict[str, list[str]] | None = None,
    ) -> dict[str, dict[str, Any]]:
        """Generate one call per scripted prompt.

        Args:
            answers: User prompt -> answer the fake model prefers.
            distractors: Generated suffix -> wrong tokens boosted after it.

        Returns:
            User prompt -> output object (as it would be written to disk).
        """
        self.model = ScriptedModel(self.tmp, answers, distractors=distractors)
        decoder = ConstrainedDecoder(self.model, self.functions)
        return {
            prompt: decoder.generate(prompt).model_dump() for prompt in answers
        }

    def decode_one(self, answer: str, **kwargs: Any) -> dict[str, Any]:
        """Generate the call for a single scripted answer."""
        return self.decode({"Handle this request": answer}, **kwargs)[
            "Handle this request"
        ]

    def assert_valid(self, call: dict[str, Any]) -> None:
        """The call follows the output rules of V.4.2."""
        self.assertEqual(schema_problems(call, self.functions), [], call)


class ConstrainedDecodingTest(DecodingCase):
    """V.1 / V.3 / V.3.3: decoding through the SDK with token masking."""

    def test_uses_the_public_sdk_methods(self) -> None:
        """V.3.1: logits come from get_logits_from_input_ids and token text
        from get_path_to_vocab_file."""
        self.decode_one('{"name":"fn_greet","parameters":{"name":"ann"}}')
        self.assertGreater(self.model.calls.get("get_logits_from_input_ids", 0), 0)
        self.assertGreater(self.model.calls.get("get_path_to_vocab_file", 0), 0)

    def test_function_is_chosen_by_the_model(self) -> None:
        """IV.3.1: the same prompt maps to whatever function the logits pick."""
        greet = self.decode_one('{"name":"fn_greet","parameters":{"name":"x"}}')
        reverse = self.decode_one(
            '{"name":"fn_reverse_string","parameters":{"s":"x"}}'
        )
        self.assertEqual(greet["name"], "fn_greet")
        self.assertEqual(reverse["name"], "fn_reverse_string")

    def test_no_keyword_heuristics(self) -> None:
        """IV.3.1: words in the prompt do not override the model's choice."""
        calls = self.decode(
            {"What is the sum of 2 and 3?":
             '{"name":"fn_greet","parameters":{"name":"two"}}'}
        )
        self.assertEqual(calls["What is the sum of 2 and 3?"]["name"], "fn_greet")

    def test_prose_before_the_json_is_masked(self) -> None:
        """V.3.3: a chatty first token can never be selected."""
        call = self.decode_one(
            '{"name":"fn_greet","parameters":{"name":"ann"}}',
            distractors={"": ["Sure", "```json", "The answer is"]},
        )
        self.assertEqual(call["parameters"], {"name": "ann"})

    def test_unknown_function_names_are_masked(self) -> None:
        """V.3.3: the name can only spell a function from the definitions."""
        call = self.decode_one(
            '{"name":"fn_greet","parameters":{"name":"ann"}}',
            distractors={'"name":"': ["fn_unknown", "greet"]},
        )
        self.assertEqual(call["name"], "fn_greet")

    def test_wrong_keys_are_masked(self) -> None:
        """V.3.3: only the schema's keys can be produced."""
        call = self.decode_one(
            '{"name":"fn_greet","parameters":{"name":"ann"}}',
            distractors={"{": ['"function"', '"arguments"', '"who"']},
        )
        self.assert_valid(call)
        self.assertEqual(call["parameters"], {"name": "ann"})

    def test_string_cannot_replace_a_number(self) -> None:
        """V.3.3: a number field only accepts number tokens."""
        call = self.decode_one(
            '{"name":"fn_add_numbers","parameters":{"a":2,"b":3}}',
            distractors={'"a":': ['"2"', "two"], '"b":': ['"3"', "null"]},
        )
        self.assertEqual(call["parameters"], {"a": 2, "b": 3})

    def test_ids_without_vocab_text_are_skipped(self) -> None:
        """V.3: logits are longer than the vocab; those ids are never picked
        (the fake scores them above everything else)."""
        call = self.decode_one('{"name":"fn_greet","parameters":{"name":"ann"}}')
        self.assertEqual(self.model.size, len(self.model.vocab) + EXTRA_IDS)
        self.assertEqual(call["parameters"], {"name": "ann"})


class ArgumentTypesTest(DecodingCase):
    """V.3.3 / V.4.2 / V.6: typed arguments and the listed edge cases."""

    def test_numbers_int_float_negative_large(self) -> None:
        """number accepts ints, floats, negatives and very large values."""
        calls = self.decode({
            "small": '{"name":"fn_add_numbers","parameters":{"a":2,"b":3.5}}',
            "negative": '{"name":"fn_add_numbers","parameters":{"a":-5,"b":-0.25}}',
            "large": '{"name":"fn_add_numbers","parameters":'
                     '{"a":123456789012345678901234567890,"b":1e3}}',
        })
        self.assertEqual(calls["small"]["parameters"], {"a": 2, "b": 3.5})
        self.assertEqual(calls["negative"]["parameters"], {"a": -5, "b": -0.25})
        large = calls["large"]["parameters"]
        self.assertEqual(large["a"], 123456789012345678901234567890)
        for call in calls.values():
            self.assert_valid(call)

    def test_boolean_and_multiple_parameters(self) -> None:
        """boolean values, two parameters of different types."""
        call = self.decode_one(
            '{"name":"fn_set_light","parameters":{"on":false,"brightness":0}}'
        )
        self.assertIs(call["parameters"]["on"], False)
        self.assertEqual(call["parameters"]["brightness"], 0)
        self.assert_valid(call)

    def test_three_parameters_in_order(self) -> None:
        """Every required argument is present (V.4.2)."""
        call = self.decode_one(
            '{"name":"fn_substitute_string_with_regex","parameters":'
            '{"source_string":"a b","regex":" ","replacement":"_"}}'
        )
        self.assertEqual(
            call["parameters"],
            {"source_string": "a b", "regex": " ", "replacement": "_"},
        )

    def test_function_without_parameters(self) -> None:
        """A function with no parameters yields ``{}``."""
        call = self.decode_one('{"name":"fn_ping","parameters":{}}')
        self.assertEqual(call["parameters"], {})
        self.assert_valid(call)

    def test_empty_string(self) -> None:
        """V.6 empty strings."""
        call = self.decode_one('{"name":"fn_reverse_string","parameters":{"s":""}}')
        self.assertEqual(call["parameters"], {"s": ""})

    def test_escaped_quotes_and_backslashes(self) -> None:
        """V.6 special characters that need JSON escapes."""
        call = self.decode_one(
            '{"name":"fn_reverse_string","parameters":'
            '{"s":"He said \\"hi\\" \\\\ bye"}}'
        )
        self.assertEqual(call["parameters"], {"s": 'He said "hi" \\ bye'})

    def test_non_ascii_text(self) -> None:
        """V.6 special characters: accents, Cyrillic, emoji survive intact."""
        calls = self.decode({
            "accent": '{"name":"fn_greet","parameters":{"name":"José"}}',
            "cyrillic": '{"name":"fn_greet","parameters":{"name":"Дмитрий"}}',
            "emoji": '{"name":"fn_reverse_string","parameters":{"s":"hi 👋"}}',
        })
        self.assertEqual(calls["accent"]["parameters"], {"name": "José"})
        self.assertEqual(calls["cyrillic"]["parameters"], {"name": "Дмитрий"})
        self.assertEqual(calls["emoji"]["parameters"], {"s": "hi 👋"})

    def test_invalid_json_escape_cannot_break_the_output(self) -> None:
        """A model writing a regex as ``\\d+`` (not valid JSON) must still
        end up with a schema-valid call (V.3.3, V.5 100% valid)."""
        call = self.decode_one(
            '{"name":"fn_substitute_string_with_regex","parameters":'
            '{"source_string":"a 1","regex":"\\d+","replacement":"N"}}'
        )
        self.assert_valid(call)

    def test_raw_control_character_cannot_break_the_output(self) -> None:
        """A raw tab inside a string is invalid JSON; output must stay valid."""
        call = self.decode_one(
            '{"name":"fn_reverse_string","parameters":{"s":"a\tb"}}'
        )
        self.assert_valid(call)

    def test_leading_zeros_cannot_break_the_output(self) -> None:
        """``007`` is not a JSON number; output must stay valid."""
        call = self.decode_one(
            '{"name":"fn_get_square_root","parameters":{"a":007}}'
        )
        self.assert_valid(call)

    def test_integer_type(self) -> None:
        """V.4.2 "number, string, boolean, etc.": integer stays an int."""
        call = self.decode_one('{"name":"fn_is_even","parameters":{"n":42}}')
        self.assert_valid(call)
        self.assertEqual(call["parameters"], {"n": 42})


class OutputFileTest(unittest.TestCase):
    """V.4 Output File Format, through ``run_pipeline``."""

    prompts: list[str]
    factory: mock.MagicMock
    text: str

    @classmethod
    def setUpClass(cls) -> None:
        """Run the whole pipeline once on the sample prompts with a fake model.

        The output goes to a directory that does not exist yet.
        """
        scratch = tempfile.TemporaryDirectory()
        cls.addClassCleanup(scratch.cleanup)
        tmp = Path(scratch.name)
        cls.prompts = list(SAMPLE_ANSWERS)
        prompts_path = write_json(
            tmp / "prompts.json", [{"prompt": text} for text in cls.prompts]
        )
        output = tmp / "nested" / "dir" / "function_calling_results.json"
        cls.factory = mock.MagicMock(
            return_value=ScriptedModel(tmp, SAMPLE_ANSWERS)
        )
        with mock.patch("src.pipeline.Small_LLM_Model", cls.factory):
            run_pipeline(SAMPLE_FUNCTIONS, prompts_path, output)
        cls.text = output.read_text(encoding="utf-8")

    def test_file_is_strict_json_array(self) -> None:
        """Valid JSON: no comments, no trailing commas, no NaN (V.4.2)."""

        def reject(constant: str) -> None:
            raise ValueError(f"non-standard JSON constant {constant}")

        data = json.loads(self.text, parse_constant=reject)
        self.assertIsInstance(data, list)
        self.assertEqual(len(data), len(self.prompts))

    def test_one_object_per_prompt_in_order(self) -> None:
        """Each prompt gets one object, original text kept, same order."""
        data = json.loads(self.text)
        self.assertEqual([item["prompt"] for item in data], self.prompts)

    def test_objects_follow_the_schema_exactly(self) -> None:
        """Exactly prompt/name/parameters, keys and types from the definitions."""
        functions = sample_functions()
        for item in json.loads(self.text):
            self.assertEqual(schema_problems(item, functions), [], item)

    def test_values_match_the_model_answers(self) -> None:
        """What the model generated is what ends up in the file."""
        for item in json.loads(self.text):
            expected = json.loads(SAMPLE_ANSWERS[item["prompt"]])
            self.assertEqual(item["name"], expected["name"])
            self.assertEqual(item["parameters"], expected["parameters"])

    def test_default_model_is_qwen3_0_6b(self) -> None:
        """IV.3.1: the pipeline loads the SDK default (Qwen/Qwen3-0.6B)."""
        self.factory.assert_called_once()
        args, kwargs = self.factory.call_args
        requested = kwargs.get("model_name", args[0] if args else "Qwen/Qwen3-0.6B")
        self.assertEqual(requested, "Qwen/Qwen3-0.6B")


if __name__ == "__main__":
    unittest.main()
