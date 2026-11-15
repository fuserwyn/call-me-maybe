"""Check the output file written by ``make run`` (subject V.4.2 and V.5).

Checks, without running the model:

- the file is valid JSON: a list of objects;
- one result per input prompt, in the same order;
- every result has exactly ``prompt``, ``name`` and ``parameters``, the name
  is a defined function and every argument has its declared type;
- when the input is the shipped sample, at least 90% of the calls are right.

Usage::

    python tests/check_output.py [output] [functions_definition] [input]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from subject_fakes import (  # noqa: E402
    ROOT,
    SAMPLE_FUNCTIONS,
    SAMPLE_PROMPTS,
    schema_problems,
)

from src.io_handlers import load_function_definitions  # noqa: E402

DEFAULT_OUTPUT = ROOT / "data" / "output" / "function_calling_results.json"


def load_calls(output: Path) -> list[Any]:
    """Read the output file.

    Args:
        output: Path to the results file.

    Returns:
        The decoded JSON list.

    Raises:
        ValueError: When the file is missing, not JSON or not a list.
    """
    try:
        data = json.loads(output.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"{output}: cannot read JSON: {exc}") from exc
    if not isinstance(data, list):
        raise ValueError(f"{output}: top level must be a list")
    return data


def output_problems(
    calls: list[Any],
    functions_path: Path,
    prompts_path: Path,
) -> list[str]:
    """List every way the results break the output rules.

    Args:
        calls: Decoded output file.
        functions_path: Function definitions the calls must follow.
        prompts_path: Input prompts the results must answer.

    Returns:
        Human-readable problems; empty when the file is valid.
    """
    functions = load_function_definitions(functions_path)
    with prompts_path.open(encoding="utf-8") as handle:
        prompts = [item["prompt"] for item in json.load(handle)]

    problems: list[str] = []
    answered = [call.get("prompt") if isinstance(call, dict) else None
                for call in calls]
    if answered != prompts:
        problems.append(
            f"prompts differ from the input: {len(answered)} results for "
            f"{len(prompts)} prompts, or a different order"
        )
    for index, call in enumerate(calls):
        if not isinstance(call, dict):
            problems.append(f"#{index}: not an object: {call!r}")
            continue
        for problem in schema_problems(call, functions):
            problems.append(f"#{index} {call.get('prompt')!r}: {problem}")
    return problems


def sample_accuracy(calls: list[Any]) -> tuple[float, list[Any]] | None:
    """Score the calls against the expected answers for the shipped sample.

    Args:
        calls: Decoded output file.

    Returns:
        ``(accuracy, wrong calls)``, or None when the prompts are not the
        shipped sample.
    """
    from test_subject_v_real_model import SAMPLE_EXPECTED

    if not calls or not all(
        isinstance(call, dict) and call.get("prompt") in SAMPLE_EXPECTED
        for call in calls
    ):
        return None
    wrong = [call for call in calls if not SAMPLE_EXPECTED[call["prompt"]](call)]
    return 1 - len(wrong) / len(calls), wrong


def main(argv: list[str]) -> int:
    """Check the output file and print a report.

    Args:
        argv: Optional paths: output, function definitions, input prompts.

    Returns:
        Exit code: 0 when every check passes, 1 otherwise.
    """
    output = Path(argv[0]) if argv else DEFAULT_OUTPUT
    functions_path = Path(argv[1]) if len(argv) > 1 else SAMPLE_FUNCTIONS
    prompts_path = Path(argv[2]) if len(argv) > 2 else SAMPLE_PROMPTS

    try:
        calls = load_calls(output)
    except ValueError as exc:
        print(f"FAIL {exc}")
        return 1

    problems = output_problems(calls, functions_path, prompts_path)
    for problem in problems:
        print(f"FAIL {problem}")
    if not problems:
        print(f"OK   valid JSON, {len(calls)} results, all schema-valid")

    scored = sample_accuracy(calls)
    if scored is not None:
        accuracy, wrong = scored
        for call in wrong:
            print(f"WRONG {json.dumps(call, ensure_ascii=False)}")
        verdict = "OK  " if accuracy >= 0.9 else "FAIL"
        print(f"{verdict} accuracy {accuracy:.0%} (at least 90% required)")
        if accuracy < 0.9:
            problems.append("accuracy")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
