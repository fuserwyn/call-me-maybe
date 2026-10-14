"""Subject VII (bonus), with the scripted model.

Recovery, caching and nested checks also live in ``test_bonus_nested.py``;
parity with the real Qwen tokenizer in ``test_bonus_tokenizer.py``.
"""

from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

from subject_fakes import (
    ScriptedModel,
    make_functions,
    schema_problems,
    write_json,
    write_tokenizer_files,
)

from src.bonus.__main__ import build_parser
from src.bonus.decoder import BonusDecoder
from src.bonus.runner import BonusConfig, run_bonus
from src.bonus.tokenizer import BonusTokenizer, TokenizerFiles

NESTED: list[dict[str, Any]] = [
    {
        "name": "fn_create_user",
        "description": "Create a user profile.",
        "parameters": {
            "user": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "number"},
                },
            },
            "tags": {"type": "array", "items": {"type": "string"}},
        },
        "returns": {"type": "string"},
    },
    {
        "name": "fn_greet",
        "description": "Greet someone.",
        "parameters": {"name": {"type": "string"}},
        "returns": {"type": "string"},
    },
]
USER = (
    '{"name":"fn_create_user","parameters":'
    '{"user":{"name":"Zoé","age":31},"tags":["admin","dev"]}}'
)
GREET = '{"name":"fn_greet","parameters":{"name":"José"}}'


class BonusCase(unittest.TestCase):
    """Run the bonus pipeline over a scripted model."""

    def setUp(self) -> None:
        """Scratch directory, inputs and a fake SDK model."""
        scratch = tempfile.TemporaryDirectory()
        self.addCleanup(scratch.cleanup)
        self.tmp = Path(scratch.name)
        self.functions_path = write_json(self.tmp / "f.json", NESTED)
        self.prompts_path = write_json(
            self.tmp / "p.json", [{"prompt": "Add Zoé"}, {"prompt": "Greet José"}]
        )
        self.model = ScriptedModel(
            self.tmp, {"Add Zoé": USER, "Greet José": GREET}
        )
        self.factory = mock.MagicMock(return_value=self.model)

    def run_bonus(self, model_name: str, visualize: bool) -> list[dict[str, Any]]:
        """Run ``run_bonus`` and return the written calls."""
        config = BonusConfig(
            model_name=model_name,
            functions_path=self.functions_path,
            input_path=self.prompts_path,
            output_path=self.tmp / "out.json",
            trace_path=self.tmp / "trace.json",
            visualize=visualize,
        )
        with mock.patch("src.bonus.runner.Small_LLM_Model", self.factory):
            run_bonus(config)
        with (self.tmp / "out.json").open(encoding="utf-8") as handle:
            data: list[dict[str, Any]] = json.load(handle)
        return data


class BonusTest(BonusCase):
    """VII Bonus Part."""

    def test_other_models_can_be_selected(self) -> None:
        """Support for LLMs beyond Qwen3-0.6B: --model reaches the SDK."""
        self.run_bonus("HuggingFaceTB/SmolLM2-360M-Instruct", visualize=False)
        _, kwargs = self.factory.call_args
        self.assertEqual(kwargs.get("model_name"),
                         "HuggingFaceTB/SmolLM2-360M-Instruct")

    def test_recoded_tokenizer_replaces_sdk_encode_decode(self) -> None:
        """The bonus path only asks the SDK for logits and file paths."""
        self.run_bonus("Qwen/Qwen3-0.6B", visualize=False)
        self.assertEqual(self.model.calls.get("encode", 0), 0)
        self.assertEqual(self.model.calls.get("decode", 0), 0)
        self.assertGreater(self.model.calls.get("get_path_to_vocab_file", 0), 0)
        self.assertGreater(self.model.calls.get("get_path_to_merges_file", 0), 0)

    def test_nested_arguments_end_to_end(self) -> None:
        """Complex nested arguments (object + array) decode schema-valid."""
        calls = self.run_bonus("Qwen/Qwen3-0.6B", visualize=False)
        user = calls[0]
        self.assertEqual(
            user["parameters"],
            {"user": {"name": "Zoé", "age": 31}, "tags": ["admin", "dev"]},
        )
        self.assertEqual(schema_problems(user, make_functions(NESTED)), [])

    def test_unicode_survives_the_bonus_decoder(self) -> None:
        """Encoding/decoding integrates with constrained decoding."""
        calls = self.run_bonus("Qwen/Qwen3-0.6B", visualize=False)
        self.assertEqual(calls[1]["parameters"], {"name": "José"})

    def test_visualization(self) -> None:
        """--visualize prints each step and writes a JSON trace."""
        printed = io.StringIO()
        with contextlib.redirect_stdout(printed):
            self.run_bonus("Qwen/Qwen3-0.6B", visualize=True)
        self.assertIn("id=", printed.getvalue())
        with (self.tmp / "trace.json").open(encoding="utf-8") as handle:
            trace = json.load(handle)
        self.assertEqual(len(trace["prompts"]), 2)
        self.assertGreater(len(trace["prompts"][0]["steps"]), 0)
        self.assertEqual(json.loads(trace["prompts"][1]["output"]),
                         json.loads(GREET))

    def test_recovery_from_a_dead_end(self) -> None:
        """Advanced error recovery: a model stuck on an invalid escape
        still yields a valid call."""
        model = ScriptedModel(
            self.tmp, {"Greet": '{"name":"fn_greet","parameters":{"name":"a\\q"}}'}
        )
        tokenizer = BonusTokenizer.from_files(
            TokenizerFiles(
                vocab_path=model.get_path_to_vocab_file(),
                merges_path=model.get_path_to_merges_file(),
            )
        )
        decoder = BonusDecoder(
            model, tokenizer, make_functions(NESTED), model_name="fake"
        )
        call, _ = decoder.generate("Greet")
        self.assertEqual(
            schema_problems(call.model_dump(), make_functions(NESTED)), []
        )


class BonusCliTest(unittest.TestCase):
    """VII: the bonus entry point ``python -m src.bonus``."""

    def test_defaults_and_model_flag(self) -> None:
        """Qwen by default, any model id through --model, trace path set."""
        defaults = build_parser().parse_args([])
        self.assertEqual(defaults.model, "Qwen/Qwen3-0.6B")
        self.assertFalse(defaults.visualize)
        custom = build_parser().parse_args(["--model", "org/other", "--visualize"])
        self.assertEqual(custom.model, "org/other")
        self.assertTrue(custom.visualize)
        self.assertEqual(custom.trace, Path("data/output/generation_trace.json"))


class PublicTokenizerTest(unittest.TestCase):
    """VII: public tokenizer ``encode`` and ``decode``."""

    def test_round_trip_without_the_sdk(self) -> None:
        """encode -> decode is lossless, including multi-byte characters."""
        with tempfile.TemporaryDirectory() as scratch:
            directory = Path(scratch)
            vocab = write_tokenizer_files(directory, [])
            tokenizer = BonusTokenizer.from_files(
                TokenizerFiles(
                    vocab_path=str(vocab),
                    merges_path=str(directory / "merges.txt"),
                )
            )
        text = 'Zoé says "hi" 👋 — 12.5\n\ttab'
        ids = tokenizer.encode(text)
        self.assertEqual(tokenizer.decode(ids), text)
        self.assertEqual(len(ids), len(text.encode("utf-8")))


if __name__ == "__main__":
    unittest.main()
