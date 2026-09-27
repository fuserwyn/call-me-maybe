"""CLI: ``uv run python -m src.bonus``."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.bonus.runner import BonusConfig, run_bonus


def build_parser() -> argparse.ArgumentParser:
    """Build the bonus command-line parser.

    Returns:
        Configured ArgumentParser.
    """
    parser = argparse.ArgumentParser(
        description="Bonus function calling: custom tokenizer, trace, any model.",
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-0.6B",
        help="Hugging Face model id (default: Qwen/Qwen3-0.6B).",
    )
    parser.add_argument(
        "--functions_definition",
        type=Path,
        default=Path("data/input/functions_definition.json"),
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/input/function_calling_tests.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/output/bonus_function_calls.json"),
    )
    parser.add_argument(
        "--trace",
        type=Path,
        default=Path("data/output/generation_trace.json"),
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Print each accepted token while generating.",
    )
    return parser


def main() -> None:
    """Parse args and run the bonus pipeline."""
    args = build_parser().parse_args()
    run_bonus(
        BonusConfig(
            model_name=args.model,
            functions_path=args.functions_definition,
            input_path=args.input,
            output_path=args.output,
            trace_path=args.trace,
            visualize=args.visualize,
        )
    )


if __name__ == "__main__":
    main()
