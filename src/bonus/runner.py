"""Run the bonus pipeline: custom tokenizer, trace, any HF causal LM."""

from __future__ import annotations

from pathlib import Path

from llm_sdk import Small_LLM_Model  # type: ignore[attr-defined]
from pydantic import BaseModel, ConfigDict, Field

from src.bonus.decoder import BonusDecoder
from src.bonus.tokenizer import BonusTokenizer, TokenizerFiles
from src.bonus.visualize import GenerationTrace
from src.io_handlers import load_function_definitions, load_prompts, save_results
from src.models import FunctionCallResult


class BonusConfig(BaseModel):
    """CLI options for the bonus entry point."""

    model_config = ConfigDict(extra="forbid")

    model_name: str = "Qwen/Qwen3-0.6B"
    functions_path: Path = Field(
        default=Path("data/input/functions_definition.json"),
    )
    input_path: Path = Field(
        default=Path("data/input/function_calling_tests.json"),
    )
    output_path: Path = Field(
        default=Path("data/output/bonus_function_calls.json"),
    )
    trace_path: Path = Field(
        default=Path("data/output/generation_trace.json"),
    )
    visualize: bool = False


def run_bonus(config: BonusConfig) -> list[FunctionCallResult]:
    """Load a model by name and decode every prompt with the bonus stack.

    Args:
        config: Model id, file paths, and whether to print the trace.

    Returns:
        One function call per input prompt.
    """
    functions = load_function_definitions(config.functions_path)
    prompts = load_prompts(config.input_path)
    if not functions:
        print("Error: no function definitions available.")
        raise SystemExit(1)

    print(f"Loading LLM ({config.model_name})...")
    model = Small_LLM_Model(model_name=config.model_name)
    tokenizer = BonusTokenizer.from_files(
        TokenizerFiles(
            vocab_path=model.get_path_to_vocab_file(),
            merges_path=model.get_path_to_merges_file(),
            model_name=config.model_name,
        )
    )
    decoder = BonusDecoder(
        model,
        tokenizer,
        functions,
        model_name=config.model_name,
        visualize=config.visualize,
    )
    results: list[FunctionCallResult] = []
    trace = GenerationTrace()
    for item in prompts:
        print(f"Processing: {item.prompt!r}")
        result, prompt_trace = decoder.generate(item.prompt)
        results.append(result)
        trace.prompts.append(prompt_trace)

    save_results(config.output_path, results)
    trace.save(config.trace_path)
    print(f"Wrote {len(results)} results to {config.output_path}")
    print(f"Wrote generation trace to {config.trace_path}")
    return results
