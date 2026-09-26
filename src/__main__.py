"""CLI entry point: ``uv run python -m src``."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.pipeline import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser.

    Returns:
        Configured ArgumentParser instance.
    """
    parser = argparse.ArgumentParser(
        description="Translate natural language into structured function calls.",
    )
    parser.add_argument(
        "--functions_definition",
        type=Path,
        default=Path("data/input/functions_definition.json"),
        help="Path to functions_definition.json",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/input/function_calling_tests.json"),
        help="Path to function_calling_tests.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/output/function_calling_results.json"),
        help="Path to write function_calling_results.json",
    )
    parser.add_argument(
        "--skip-model",
        action="store_true",
        help="Skip LLM load (IO smoke test only; not for submission).",
    )
    return parser


def main() -> None:
    """Parse CLI args and run the function-calling pipeline."""
    args = build_parser().parse_args()
    run_pipeline(
        args.functions_definition,
        args.input,
        args.output,
        load_model=not args.skip_model,
    )


if __name__ == "__main__":
    main()
