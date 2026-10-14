"""Subject IV.3.1 / IV.3.2 / V.2 / V.5: CLI usage and graceful error handling."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from subject_fakes import (
    ROOT,
    SAMPLE_FUNCTIONS,
    SAMPLE_PROMPTS,
    ScriptedModel,
    write_json,
)

from src.__main__ import build_parser, main
from src.io_handlers import load_function_definitions, load_prompts
from src.pipeline import run_pipeline

GREET = '{"name":"fn_greet","parameters":{"name":"ann"}}'


def run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    """Run ``python -m src`` from the repo root."""
    return subprocess.run(
        [sys.executable, "-m", "src", *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )


class UsageTest(unittest.TestCase):
    """IV.3.2 Usage."""

    def test_default_paths(self) -> None:
        """Without flags, inputs come from data/input/ and output goes to
        data/output/function_calling_results.json (IV.3.2, V.4)."""
        args = build_parser().parse_args([])
        self.assertEqual(
            args.functions_definition, Path("data/input/functions_definition.json")
        )
        self.assertEqual(args.input, Path("data/input/function_calling_tests.json"))
        self.assertEqual(
            args.output, Path("data/output/function_calling_results.json")
        )

    def test_custom_paths(self) -> None:
        """--functions_definition, --input and --output override the defaults."""
        args = build_parser().parse_args(
            ["--functions_definition", "f.json", "--input", "i.json",
             "--output", "o.json"]
        )
        self.assertEqual(
            (args.functions_definition, args.input, args.output),
            (Path("f.json"), Path("i.json"), Path("o.json")),
        )

    def test_runs_as_python_m_src(self) -> None:
        """``python -m src --help`` works and lists the three flags."""
        result = run_cli("--help")
        self.assertEqual(result.returncode, 0, result.stderr)
        for flag in ("--functions_definition", "--input", "--output"):
            self.assertIn(flag, result.stdout)


class CleanFailureTest(unittest.TestCase):
    """IV.1 / IV.3.1 / V.2 / V.5: bad input gives a clear error, never a crash."""

    def setUp(self) -> None:
        """Create a scratch directory."""
        scratch = tempfile.TemporaryDirectory()
        self.addCleanup(scratch.cleanup)
        self.tmp = Path(scratch.name)

    def assert_clean_failure(self, *args: str) -> None:
        """The CLI exits with status 1 and an error message, no traceback."""
        result = run_cli(*args, "--output", str(self.tmp / "out.json"))
        output = result.stdout + result.stderr
        self.assertNotIn("Traceback", output, output)
        self.assertEqual(result.returncode, 1, output)
        self.assertIn("rror", output)

    def file_with(self, name: str, content: bytes) -> str:
        """Write raw bytes to a scratch file and return its path."""
        path = self.tmp / name
        path.write_bytes(content)
        return str(path)

    def test_missing_functions_file(self) -> None:
        """Missing functions_definition.json."""
        self.assert_clean_failure("--functions_definition", str(self.tmp / "no.json"))

    def test_missing_input_file(self) -> None:
        """Missing function_calling_tests.json."""
        self.assert_clean_failure("--input", str(self.tmp / "no.json"))

    def test_invalid_json_in_input(self) -> None:
        """Truncated JSON in the prompts file."""
        self.assert_clean_failure("--input", self.file_with("i.json", b'[{"prompt": '))

    def test_invalid_json_in_functions(self) -> None:
        """Trailing comma in the functions file."""
        bad = self.file_with("f.json", b'[{"name": "fn_a",}]')
        self.assert_clean_failure("--functions_definition", bad)

    def test_empty_file(self) -> None:
        """Zero-byte prompts file."""
        self.assert_clean_failure("--input", self.file_with("i.json", b""))

    def test_non_utf8_file(self) -> None:
        """Binary / non-UTF-8 prompts file."""
        self.assert_clean_failure("--input", self.file_with("i.json", b"\xff\xfe[]"))

    def test_input_path_is_a_directory(self) -> None:
        """--input pointing at a directory."""
        self.assert_clean_failure("--input", str(self.tmp))

    def test_input_is_an_object_not_a_list(self) -> None:
        """Prompts file holding an object instead of an array."""
        self.assert_clean_failure(
            "--input", self.file_with("i.json", b'{"prompt": "Greet ann"}')
        )

    def test_prompt_item_without_prompt_key(self) -> None:
        """Prompt object missing its ``prompt`` key."""
        self.assert_clean_failure(
            "--input", self.file_with("i.json", b'[{"question": "hi"}]')
        )

    def test_prompt_is_not_a_string(self) -> None:
        """``prompt`` holding a number."""
        self.assert_clean_failure(
            "--input", self.file_with("i.json", b'[{"prompt": 42}]')
        )

    def test_function_without_name(self) -> None:
        """Function definition missing ``name``."""
        bad = self.file_with(
            "f.json",
            b'[{"description": "d", "parameters": {}, "returns": {"type": "x"}}]',
        )
        self.assert_clean_failure("--functions_definition", bad)

    def test_no_functions_at_all(self) -> None:
        """Empty function list."""
        self.assert_clean_failure(
            "--functions_definition", self.file_with("f.json", b"[]")
        )


class InProcessErrorTest(unittest.TestCase):
    """Failures that happen after the inputs are read."""

    def setUp(self) -> None:
        """Create a scratch directory and a one-prompt input file."""
        scratch = tempfile.TemporaryDirectory()
        self.addCleanup(scratch.cleanup)
        self.tmp = Path(scratch.name)
        self.prompts = write_json(self.tmp / "p.json", [{"prompt": "Greet ann"}])

    def fake_factory(self) -> mock.MagicMock:
        """A patched ``Small_LLM_Model`` that builds a ScriptedModel."""
        model = ScriptedModel(self.tmp, {"Greet ann": GREET})
        return mock.MagicMock(return_value=model)

    def test_model_load_failure_is_reported_cleanly(self) -> None:
        """No network / no disk while loading Qwen -> message + exit 1."""
        argv = ["src", "--input", str(self.prompts),
                "--output", str(self.tmp / "o.json")]
        broken = mock.MagicMock(side_effect=OSError("cannot download model"))
        with mock.patch.object(sys, "argv", argv), \
                mock.patch("src.pipeline.Small_LLM_Model", broken):
            try:
                main()
            except SystemExit as exc:
                self.assertEqual(exc.code, 1)
            except Exception as exc:
                self.fail(f"program crashed with {exc!r}")
            else:
                self.fail("a model load failure must exit with status 1")

    def test_unwritable_output_is_reported_cleanly(self) -> None:
        """--output pointing at a directory -> SystemExit(1), no traceback."""
        with mock.patch("src.pipeline.Small_LLM_Model", self.fake_factory()), \
                self.assertRaises(SystemExit) as caught:
            run_pipeline(SAMPLE_FUNCTIONS, self.prompts, self.tmp)
        self.assertEqual(caught.exception.code, 1)

    def test_empty_prompt_list_writes_an_empty_array(self) -> None:
        """``[]`` as input is valid and produces ``[]``."""
        empty = write_json(self.tmp / "empty.json", [])
        out = self.tmp / "o.json"
        with mock.patch("src.pipeline.Small_LLM_Model", self.fake_factory()):
            run_pipeline(SAMPLE_FUNCTIONS, empty, out)
        with out.open(encoding="utf-8") as handle:
            self.assertEqual(json.load(handle), [])

    def test_extra_schema_keys_are_accepted(self) -> None:
        """V.2: other function sets may carry extra JSON-schema keys
        (per-parameter ``description``, top-level ``required``)."""
        functions = write_json(self.tmp / "f.json", [{
            "name": "fn_greet",
            "description": "Greet someone.",
            "parameters": {
                "name": {"type": "string", "description": "Who to greet"},
            },
            "required": ["name"],
            "returns": {"type": "string"},
        }])
        out = self.tmp / "o.json"
        with mock.patch("src.pipeline.Small_LLM_Model", self.fake_factory()):
            try:
                run_pipeline(functions, self.prompts, out)
            except SystemExit:
                self.fail("definitions with extra schema keys were rejected")
        with out.open(encoding="utf-8") as handle:
            self.assertEqual(json.load(handle)[0]["parameters"], {"name": "ann"})

    def test_sample_inputs_are_accepted(self) -> None:
        """V.2: the shipped data/input files load and validate."""
        functions = load_function_definitions(SAMPLE_FUNCTIONS)
        prompts = load_prompts(SAMPLE_PROMPTS)
        self.assertGreater(len(functions), 0)
        self.assertGreater(len(prompts), 0)
        for fn in functions:
            self.assertTrue(fn.name and fn.description and fn.returns.type)


if __name__ == "__main__":
    unittest.main()
