*This project has been created as part of the 42 curriculum by fuserwyn.*

# call me maybe

## Description

This project implements **LLM function calling** with **constrained decoding**.

Given a natural-language request such as *"What is the sum of 2 and 3?"*, the program does **not** answer `5`. It asks a small local model (`Qwen/Qwen3-0.6B`) to emit a structured JSON **function call**:

```json
{"name": "fn_add_numbers", "parameters": {"a": 2, "b": 3}}
```

Small models often break JSON when left unconstrained. Here, every generated token is filtered so the output always stays a valid prefix of a schema-compliant call (function name from the catalog, parameter keys in definition order, typed values).

The stack is Python 3.10+, `uv`, `pydantic`, `numpy`, and the provided `llm_sdk` wrapper (no direct use of `transformers` / `torch` / `outlines` in application code).

## Instructions

### Requirements

- Python 3.10+
- [`uv`](https://github.com/astral-sh/uv)
- Network on first run (downloads the Qwen model and tokenizer vocab into the Hugging Face cache)

### Install

```bash
make install
# equivalent: uv sync --extra dev
```

### Run

Default paths (`data/input/…` → `data/output/function_calling_results.json`):

```bash
make run
# equivalent: uv run python -m src
```

Custom paths:

```bash
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calling_results.json
```

### Lint

```bash
make lint
```

### Clean

```bash
make clean
```

### Debug

```bash
make debug
```

## Algorithm explanation

Generation is a token-by-token loop on top of `llm_sdk.Small_LLM_Model`:

1. Build a text prompt listing available functions and the user request (`prompt_builder.py`).
2. `encode` the prompt to token ids.
3. At each step call `get_logits_from_input_ids` to get next-token scores.
4. Walk candidate token ids in **descending logit order**.
5. Map each id to text via `vocab.json` (`get_path_to_vocab_file`, with Qwen markers `Ġ`→space and `Ċ`→newline).
6. Accept the **first** token such that `generated + token_text` is still a valid **prefix** of some legal call JSON (`constraints.py`). That token is the constrained argmax.
7. Append it and repeat until the checker reports a complete, `json.loads`-able object.
8. Return `{prompt, name, parameters}`.

The prefix checker enforces:

- object shape `{"name": "...", "parameters": {...}}` (optional whitespace);
- `name` must be one of the defined function names (trie / prefix match while the string is open);
- after the name is known, parameter keys must follow the definition order;
- values must match declared types (`number`, `string`, `boolean`), including incomplete prefixes (e.g. open strings, partial numbers).

Invalid tokens never win, so the final string is always parseable JSON matching the schema when generation finishes successfully.

## Design decisions

- **Prefix grammar over hard-coded answers** — the model still chooses function names and argument values via logits; we only restrict the legal token set.
- **Public `llm_sdk` API only** — no private attributes; no direct Hugging Face calls in `src/`.
- **Pydantic models** for definitions, prompts, and results (`models.py`).
- **Fast constrained argmax** — scan tokens from highest logit downward and take the first legal one instead of masking the full vocabulary every time (same result, much faster when the model already prefers valid JSON).
- **Prompt with a few examples** — improves regex / substitution argument quality on the tiny 0.6B model without replacing constrained decoding.

## Performance analysis

On the bundled sample (`11` prompts):

- **Validity:** 100% parseable, schema-shaped JSON in practice.
- **Accuracy:** near-perfect function selection and argument extraction on the provided tests (sums, greetings, reverse, square root, regex substitutions).
- **Speed:** full run typically around **2–3 minutes** on an Apple Silicon Mac after the model is cached (under the 5-minute guideline). First run is slower because weights download.

Reliability comes mainly from structural masking: even when the raw model would emit prose or broken braces, those tokens are rejected.

## Challenges faced

- **Merged BPE tokens** — Qwen often emits pieces like `{"`, `":"`, `{\n`, not single characters. Constraints must validate **decoded text prefixes**, not assume one character per step.
- **Vocab markers** — space/newline appear as `Ġ` / `Ċ` in `vocab.json`; mapping them incorrectly breaks both masking and decoding.
- **Logits longer than vocab** — some ids have no vocab entry; they are skipped.
- **Regex prompts** — structure can be forced, but string *contents* still depend on the model; few-shot hints in the prompt helped without violating “choose the function with the LLM”.
- **Speed** — naive “test all ~150k tokens every step” was too slow; ordered logit scan fixed that.

## Testing strategy

- `make lint` — `flake8` + `mypy` with the flags required by the subject.
- Manual full pipeline: `uv run python -m src`, then inspect `data/output/function_calling_results.json`.
- Spot-checks on individual prompts through `ConstrainedDecoder.generate`.
- IO / missing-file paths return clear errors and exit non-zero instead of crashing.
- Note: reviewers may swap input JSON; nothing is hard-coded to the sample answers.

## Example usage

```bash
make install
uv run python -m src
```

Example output entry:

```json
{
  "prompt": "Greet shrek",
  "name": "fn_greet",
  "parameters": {"name": "shrek"}
}
```

IO-only smoke test (no model load; not for evaluation):

```bash
uv run python -m src --skip-model
```

## Bonus

Optional features live in `src/bonus` and stay off the mandatory path. Run them with:

```bash
make bonus
```

That target runs the bonus test suite, then:

```bash
uv run python -m src.bonus --visualize
```

Another model (still a causal LM the SDK can load):

```bash
uv run python -m src.bonus --model Qwen/Qwen3-0.6B --visualize
```

What the bonus module adds:

- **Other models** — `--model` is passed to `Small_LLM_Model`.
- **Recoded tokenizer** — `BonusTokenizer.encode` / `decode` use `vocab.json` and `merges.txt` from `get_path_to_vocab_file` and `get_path_to_merges_file`. The bonus decoder never calls the SDK `encode` or `decode`; it only calls `get_logits_from_input_ids`.
- **Nested arguments** — object and array parameter schemas (`NestedCallChecker`).
- **Recovery** — if a token leads to a dead end, generation tries the next legal alternative, then closes truncated braces.
- **Cache and batching** — legality of `(prefix, token)` is cached; each step checks a top-logit batch before scanning the rest of the vocabulary.
- **Visualization** — `--visualize` prints every accepted token. A full trace is written to `data/output/generation_trace.json`.
- **Tests** — `tests/test_bonus_nested.py` and `tests/test_bonus_tokenizer.py`.

## Resources

- [Hugging Face — Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B)
- [Transformers docs (reference only; app code uses `llm_sdk`)](https://huggingface.co/docs/transformers)
- Guidance / constrained decoding overview articles on structured generation and grammar-constrained decoding (e.g. discussions around logit masking and JSON grammars)
- 42 subject PDF: *call me maybe — Introduction to function calling in LLMs*

### How AI was used

AI assistance (Cursor) was used to:

- scaffold the project layout (`pyproject.toml`, `Makefile`, CLI, pydantic models);
- implement and iterate on constrained decoding / prefix checking;
- draft and refine this README;
- help debug tokenizer edge cases (merged tokens, `Ġ`/`Ċ`) and performance of token selection.

All algorithm choices and final code were reviewed, run locally, and are understood by the author (`fuserwyn`). AI output was not submitted blindly.
