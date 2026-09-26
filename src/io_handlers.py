"""Load and save project JSON files with graceful error handling."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from src.models import FunctionCallResult, FunctionDefinition, PromptItem


def load_json_file(path: Path) -> Any:
    """Read a JSON file and return the parsed value.

    Args:
        path: Path to the JSON file.

    Returns:
        Parsed JSON content.

    Raises:
        SystemExit: On missing file or invalid JSON (prints a clear message).
    """
    if not path.exists():
        print(f"Error: input file not found: {path}", file=sys.stderr)
        raise SystemExit(1)
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except json.JSONDecodeError as exc:
        print(f"Error: invalid JSON in {path}: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    except OSError as exc:
        print(f"Error: cannot read {path}: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


def load_function_definitions(path: Path) -> list[FunctionDefinition]:
    """Load and validate function definitions.

    Args:
        path: Path to functions_definition.json.

    Returns:
        Validated list of FunctionDefinition models.
    """
    raw = load_json_file(path)
    try:
        return [FunctionDefinition.model_validate(item) for item in raw]
    except (ValidationError, TypeError) as exc:
        print(f"Error: invalid function definitions in {path}: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


def load_prompts(path: Path) -> list[PromptItem]:
    """Load and validate natural-language prompts.

    Args:
        path: Path to function_calling_tests.json.

    Returns:
        Validated list of PromptItem models.
    """
    raw = load_json_file(path)
    try:
        return [PromptItem.model_validate(item) for item in raw]
    except (ValidationError, TypeError) as exc:
        print(f"Error: invalid prompts in {path}: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


def save_results(path: Path, results: list[FunctionCallResult]) -> None:
    """Write function-calling results to a JSON file.

    Args:
        path: Destination path (parent directories are created if needed).
        results: List of FunctionCallResult models to serialize.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = [item.model_dump() for item in results]
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
    except OSError as exc:
        print(f"Error: cannot write {path}: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
