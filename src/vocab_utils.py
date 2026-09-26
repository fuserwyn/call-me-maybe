"""Vocabulary helpers for mapping token ids to printable text."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def load_vocab(path: str | Path) -> dict[str, int]:
    """Load a Hugging Face vocab.json file.

    Args:
        path: Path returned by ``get_path_to_vocab_file()``.

    Returns:
        Mapping from raw token string to token id.
    """
    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"Error: cannot load vocab from {path}: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    if not isinstance(data, dict):
        print(f"Error: unexpected vocab format in {path}", file=sys.stderr)
        raise SystemExit(1)
    return data


def token_to_text(raw_token: str) -> str:
    """Convert a raw vocab token into the text it contributes when decoded.

    Qwen uses special markers: ``Ġ`` (U+0120) for space and ``Ċ`` for newline.

    Args:
        raw_token: Token string as stored in vocab.json.

    Returns:
        Human-readable text fragment.
    """
    return (
        raw_token.replace("\u0120", " ")
        .replace("Ċ", "\n")
        .replace("ĉ", "\t")
    )


def build_id_to_text(vocab: dict[str, int]) -> dict[int, str]:
    """Build token-id → decoded text mapping.

    Args:
        vocab: Raw vocab mapping token → id.

    Returns:
        Mapping id → printable text.
    """
    return {token_id: token_to_text(token) for token, token_id in vocab.items()}
