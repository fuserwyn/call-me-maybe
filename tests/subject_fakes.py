"""Test doubles shared by the subject-compliance suite.

``ScriptedModel`` exposes the public API of ``llm_sdk.Small_LLM_Model`` on top
of a tiny byte-level vocabulary, so constrained decoding can be tested
without loading Qwen. Its logits favour one scripted answer per prompt.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from src.io_handlers import load_function_definitions
from src.models import FunctionDefinition

ROOT = Path(__file__).resolve().parent.parent
SAMPLE_FUNCTIONS = ROOT / "data" / "input" / "functions_definition.json"
SAMPLE_PROMPTS = ROOT / "data" / "input" / "function_calling_tests.json"

EXTRA_IDS = 4
"""Logit slots with no vocab entry (Qwen's logits are longer than its vocab)."""

MERGED_TOKENS = (
    '{"',
    '":',
    '":"',
    '","',
    '"}',
    "}}",
    '"name"',
    '"parameters"',
    "fn_",
)
"""Multi-character tokens, like real BPE merges, mixed into the fake vocab."""

FALLBACK_SCORES = {'"': 1.0, "}": 0.5, ",": 0.4, "0": 0.3}
"""Scores that let generation finish once it leaves the scripted answer."""


def byte_level_alphabet() -> dict[int, str]:
    """Return the GPT-2 / Qwen byte-to-unicode table.

    Written independently of ``src`` so the tests do not trust the code under
    test for their ground truth.

    Returns:
        Mapping from byte value to the character vocab.json uses for it.
    """
    printable = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    codes = printable[:]
    shift = 0
    for byte in range(256):
        if byte not in printable:
            printable.append(byte)
            codes.append(256 + shift)
            shift += 1
    return {byte: chr(code) for byte, code in zip(printable, codes)}


BYTE_TO_CHAR = byte_level_alphabet()
CHAR_TO_BYTE = {char: byte for byte, char in BYTE_TO_CHAR.items()}


def to_vocab_token(text: str) -> str:
    """Spell ``text`` the way vocab.json stores it.

    Args:
        text: Plain text.

    Returns:
        Byte-level unicode spelling (``" "`` becomes ``"Ġ"`` and so on).
    """
    return "".join(BYTE_TO_CHAR[byte] for byte in text.encode("utf-8"))


def write_tokenizer_files(directory: Path, extra_tokens: list[str]) -> Path:
    """Write a vocab.json (all 256 bytes plus extras) and an empty merges.txt.

    Args:
        directory: Where to write the files.
        extra_tokens: Multi-character tokens to add after the 256 bytes.

    Returns:
        Path to the vocab.json file.
    """
    vocab: dict[str, int] = {}
    for byte in range(256):
        vocab[BYTE_TO_CHAR[byte]] = len(vocab)
    for text in extra_tokens:
        token = to_vocab_token(text)
        if token not in vocab:
            vocab[token] = len(vocab)
    vocab_path = directory / "vocab.json"
    with vocab_path.open("w", encoding="utf-8") as handle:
        json.dump(vocab, handle, ensure_ascii=False)
    with (directory / "merges.txt").open("w", encoding="utf-8") as handle:
        handle.write("#version: 0.2\n")
    return vocab_path


class ScriptedModel:
    """Fake ``Small_LLM_Model`` whose logits prefer a scripted answer.

    The prompt of each generation is detected from the first logits call.
    The scripted answer is the one whose user prompt appears last in it.
    Tokens continuing that answer score highest (longest token first, as
    with BPE). ``distractors`` boosts wrong tokens above everything right
    after a given generated suffix (``""`` means "before the first token"),
    which proves the decoder masks them out. Ids past the vocab get the
    highest score of all and must be skipped.
    """

    def __init__(
        self,
        directory: Path,
        answers: dict[str, str],
        *,
        distractors: dict[str, list[str]] | None = None,
        model_name: str = "Qwen/Qwen3-0.6B",
    ) -> None:
        """Build the fake vocabulary and remember the scripted answers.

        Args:
            directory: Scratch directory for vocab.json and merges.txt.
            answers: User prompt -> JSON text the fake model "wants" to emit.
            distractors: Generated suffix -> tokens to boost after it.
            model_name: Recorded only, mirrors the SDK constructor.
        """
        self.model_name = model_name
        self.distractors = distractors or {}
        extra = list(MERGED_TOKENS)
        for tokens in self.distractors.values():
            extra.extend(tokens)
        self.vocab_path = write_tokenizer_files(directory, extra)
        self.merges_path = directory / "merges.txt"
        with self.vocab_path.open(encoding="utf-8") as handle:
            self.vocab: dict[str, int] = json.load(handle)
        self.id_to_bytes = {
            token_id: bytes(CHAR_TO_BYTE[char] for char in token)
            for token, token_id in self.vocab.items()
        }
        self.size = len(self.vocab) + EXTRA_IDS
        self.answers = [
            (prompt.encode("utf-8"), answer.encode("utf-8"))
            for prompt, answer in answers.items()
        ]
        self.calls: dict[str, int] = {}
        self._prompts: list[tuple[bytes, bytes]] = []

    def encode(self, text: str) -> np.ndarray:
        """Encode text one byte per token (2-D, like the SDK tensor)."""
        self._count("encode")
        ids = [self.vocab[BYTE_TO_CHAR[byte]] for byte in text.encode("utf-8")]
        return np.array([ids], dtype=np.int64)

    def decode(self, ids: list[int]) -> str:
        """Decode ids back to text."""
        self._count("decode")
        data = b"".join(self.id_to_bytes.get(token_id, b"") for token_id in ids)
        return data.decode("utf-8", errors="replace")

    def get_path_to_vocab_file(self) -> str:
        """Return the fake vocab.json path."""
        self._count("get_path_to_vocab_file")
        return str(self.vocab_path)

    def get_path_to_merges_file(self) -> str:
        """Return the fake merges.txt path."""
        self._count("get_path_to_merges_file")
        return str(self.merges_path)

    def get_logits_from_input_ids(self, input_ids: list[int]) -> list[float]:
        """Score every id for the next position.

        Args:
            input_ids: Prompt ids followed by the ids generated so far.

        Returns:
            One float per id, including the ``EXTRA_IDS`` without vocab text.
        """
        self._count("get_logits_from_input_ids")
        data = b"".join(self.id_to_bytes.get(token_id, b"") for token_id in input_ids)
        prompt, answer = self._locate(data)
        done = data[len(prompt):]
        scores = np.full(self.size, -5.0)
        for token_id, piece in self.id_to_bytes.items():
            if piece.isspace():
                scores[token_id] = -10.0
        for text, score in FALLBACK_SCORES.items():
            scores[self.vocab[to_vocab_token(text)]] = score
        if answer.startswith(done):
            rest = answer[len(done):]
            for token_id, piece in self.id_to_bytes.items():
                if piece and rest.startswith(piece):
                    scores[token_id] = 10.0 + len(piece)
        for suffix, tokens in self.distractors.items():
            trigger = suffix.encode("utf-8")
            if (not trigger and not done) or (trigger and done.endswith(trigger)):
                for text in tokens:
                    scores[self.vocab[to_vocab_token(text)]] = 40.0
        scores[len(self.vocab):] = 50.0
        return [float(score) for score in scores]

    def _locate(self, data: bytes) -> tuple[bytes, bytes]:
        """Split ``data`` into (prompt, scripted answer) for this generation."""
        for prompt, answer in self._prompts:
            if data.startswith(prompt):
                return prompt, answer
        positions = [(data.rfind(user), answer) for user, answer in self.answers]
        position, answer = max(positions, key=lambda item: item[0])
        if position < 0:
            raise AssertionError("ScriptedModel got a prompt it has no answer for")
        self._prompts.insert(0, (data, answer))
        return data, answer

    def _count(self, method: str) -> None:
        """Record that ``method`` was called."""
        self.calls[method] = self.calls.get(method, 0) + 1


def sample_functions() -> list[FunctionDefinition]:
    """Load the function definitions shipped in data/input."""
    return load_function_definitions(SAMPLE_FUNCTIONS)


def make_functions(raw: list[dict[str, Any]]) -> list[FunctionDefinition]:
    """Validate raw function definition dicts.

    Args:
        raw: Definitions in the functions_definition.json format.

    Returns:
        Validated models.
    """
    return [FunctionDefinition.model_validate(item) for item in raw]


def write_json(path: Path, data: Any) -> Path:
    """Write ``data`` as JSON and return the path."""
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False)
    return path


def type_matches(value: Any, json_type: str) -> bool:
    """Check a decoded JSON value against a schema type name.

    Args:
        value: Decoded value.
        json_type: ``number``, ``integer``, ``string``, ``boolean``, ...

    Returns:
        Whether the value has that JSON type.
    """
    if json_type == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if json_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if json_type == "string":
        return isinstance(value, str)
    if json_type == "boolean":
        return isinstance(value, bool)
    if json_type == "array":
        return isinstance(value, list)
    if json_type == "object":
        return isinstance(value, dict)
    return False


def schema_problems(
    call: dict[str, Any],
    functions: list[FunctionDefinition],
) -> list[str]:
    """List every way ``call`` breaks the subject's output rules (V.4.2).

    Args:
        call: One object from the output file.
        functions: Definitions the call must follow.

    Returns:
        Human-readable problems; empty when the call is valid.
    """
    problems: list[str] = []
    if set(call) != {"prompt", "name", "parameters"}:
        problems.append(f"keys must be prompt/name/parameters, got {sorted(call)}")
    by_name = {fn.name: fn for fn in functions}
    fn = by_name.get(call.get("name", ""))
    if fn is None:
        problems.append(f"unknown function name {call.get('name')!r}")
        return problems
    params = call.get("parameters")
    if not isinstance(params, dict):
        problems.append(f"parameters must be an object, got {params!r}")
        return problems
    if set(params) != set(fn.parameters):
        problems.append(
            f"{fn.name}: parameter keys {sorted(params)} != {sorted(fn.parameters)}"
        )
    for key, spec in fn.parameters.items():
        if key in params and not type_matches(params[key], spec.type):
            problems.append(f"{fn.name}.{key}={params[key]!r} is not {spec.type}")
    return problems
