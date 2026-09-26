"""End-to-end pipeline: load inputs, decode, write results."""

from __future__ import annotations

from pathlib import Path

from llm_sdk import Small_LLM_Model  # type: ignore[attr-defined]

from src.decoder import ConstrainedDecoder
from src.io_handlers import load_function_definitions, load_prompts, save_results
from src.models import FunctionCallResult


def run_pipeline(
    functions_path: Path,
    input_path: Path,
    output_path: Path,
    *,
    load_model: bool = True,
) -> list[FunctionCallResult]:
    """Run the full function-calling pipeline.

    Args:
        functions_path: Path to functions_definition.json.
        input_path: Path to function_calling_tests.json.
        output_path: Destination for function_calling_results.json.
        load_model: If False, skip model load (used for IO-only smoke tests).

    Returns:
        List of generated FunctionCallResult objects.
    """
    functions = load_function_definitions(functions_path)
    prompts = load_prompts(input_path)

    if not functions:
        print("Error: no function definitions available.")
        raise SystemExit(1)

    results: list[FunctionCallResult] = []

    if load_model:
        print("Loading LLM (Qwen/Qwen3-0.6B)...")
        model = Small_LLM_Model()
        decoder = ConstrainedDecoder(model, functions)
        for item in prompts:
            print(f"Processing: {item.prompt!r}")
            results.append(decoder.generate(item.prompt))
    else:
        # IO skeleton path — placeholders until constrained decoding lands
        for item in prompts:
            first = functions[0]
            params = {
                name: 0.0 if spec.type == "number" else ""
                for name, spec in first.parameters.items()
            }
            results.append(
                FunctionCallResult(
                    prompt=item.prompt,
                    name=first.name,
                    parameters=params,
                )
            )

    save_results(output_path, results)
    print(f"Wrote {len(results)} results to {output_path}")
    return results
