"""End-to-end pipeline: load inputs, decode, write results."""

from __future__ import annotations

import sys
from pathlib import Path

from llm_sdk import Small_LLM_Model  # type: ignore[attr-defined]

from src.decoder import ConstrainedDecoder
from src.io_handlers import load_function_definitions, load_prompts, save_results
from src.models import FunctionCallResult, FunctionDefinition


def placeholder_value(json_type: str) -> object:
    """Return a value of the given JSON type for the IO smoke test.

    Args:
        json_type: Schema type name.

    Returns:
        A neutral value of that type.
    """
    defaults: dict[str, object] = {
        "number": 0.0, "integer": 0, "boolean": False,
        "array": [], "object": {}, "null": None,
    }
    return defaults.get(json_type, "")


def load_decoder(functions: list[FunctionDefinition]) -> ConstrainedDecoder:
    """Load Qwen and wrap it in the constrained decoder.

    Args:
        functions: Validated function definitions.

    Returns:
        Ready decoder.

    Raises:
        SystemExit: When the model cannot be loaded (no network, no disk...).
    """
    print("Loading LLM (Qwen/Qwen3-0.6B)...")
    try:
        model = Small_LLM_Model()
        return ConstrainedDecoder(model, functions)
    except SystemExit:
        raise
    except Exception as exc:  # the SDK can fail in many library-specific ways
        print(f"Error: cannot load the language model: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


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
        print("Error: no function definitions available.", file=sys.stderr)
        raise SystemExit(1)

    results: list[FunctionCallResult] = []

    if load_model:
        decoder = load_decoder(functions)
        for item in prompts:
            print(f"Processing: {item.prompt!r}")
            try:
                results.append(decoder.generate(item.prompt))
            except Exception as exc:  # model runtime errors (OOM, device...)
                print(
                    f"Error: generation failed for {item.prompt!r}: {exc}",
                    file=sys.stderr,
                )
                raise SystemExit(1) from exc
    else:
        first = functions[0]
        for item in prompts:
            params = {
                name: placeholder_value(spec.type)
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
