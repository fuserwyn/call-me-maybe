"""Byte-level BPE tokenizer rebuilt from vocab.json and merges.txt.

The bonus path never calls ``Small_LLM_Model.encode`` or ``decode``.
Token ids come from the public vocab file; merges come from
``get_path_to_merges_file``.
"""

from __future__ import annotations

import json
import re
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from src.vocab_utils import bytes_to_unicode

_LETTER = r"[^\W\d_]"
"""Unicode letter (``\\p{L}``) in the standard ``re`` module."""

_SPLIT = re.compile(
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)"
    rf"|(?:[^\w\r\n]|_)?{_LETTER}+"
    r"|\d"
    r"| ?(?:[^\s\w]|_)+[\r\n]*"
    r"|\s*[\r\n]+"
    r"|\s+(?!\S)"
    r"|\s+"
)
"""Qwen2 pre-tokenizer split, written for ``re`` instead of ``regex``.

``\\p{L}`` becomes ``[^\\W\\d_]`` and ``\\p{N}`` becomes ``\\d``. The two
only differ on rare numeric symbols such as ``²`` (counted as letters here).
"""


class TokenizerFiles(BaseModel):
    """Paths the SDK exposes for a local recode of the tokenizer."""

    model_config = ConfigDict(extra="forbid")

    vocab_path: str
    merges_path: str
    model_name: str = Field(default="Qwen/Qwen3-0.6B")


class BonusTokenizer(BaseModel):
    """Public encode/decode compatible with Qwen byte-level BPE."""

    model_config = ConfigDict(extra="forbid")

    vocab: dict[str, int]
    merges: list[tuple[str, str]]
    _id_to_token: dict[int, str] = PrivateAttr(default_factory=dict)
    _ranks: dict[tuple[str, str], int] = PrivateAttr(default_factory=dict)
    _byte_encoder: dict[int, str] = PrivateAttr(default_factory=dict)
    _byte_decoder: dict[str, int] = PrivateAttr(default_factory=dict)
    _bpe_cache: dict[str, tuple[str, ...]] = PrivateAttr(default_factory=dict)

    def __init__(
        self,
        vocab: dict[str, int],
        merges: list[tuple[str, str]],
        **data: Any,
    ) -> None:
        """Store vocab and merges.

        Args:
            vocab: Token string to id, as stored in vocab.json.
            merges: Merge pairs in priority order (earlier = higher priority).
            **data: Extra pydantic fields (none for now).
        """
        super().__init__(vocab=vocab, merges=merges, **data)

    def model_post_init(self, context: Any, /) -> None:
        """Build lookup tables.

        Args:
            context: Pydantic validation context (unused).
        """
        self._id_to_token = {
            token_id: token for token, token_id in self.vocab.items()
        }
        self._ranks = {pair: rank for rank, pair in enumerate(self.merges)}
        self._byte_encoder = bytes_to_unicode()
        self._byte_decoder = {
            char: byte for byte, char in self._byte_encoder.items()
        }

    @classmethod
    def from_files(cls, files: TokenizerFiles) -> "BonusTokenizer":
        """Load vocab and merges from disk.

        Args:
            files: Paths returned by the SDK.

        Returns:
            Ready tokenizer.
        """
        vocab = _load_vocab(files.vocab_path)
        merges = _load_merges(files.merges_path)
        return cls(vocab, merges)

    def encode(self, text: str) -> list[int]:
        """Encode text to token ids without the model tokenizer.

        Args:
            text: Raw string.

        Returns:
            List of vocabulary ids.
        """
        normalized = unicodedata.normalize("NFC", text)
        ids: list[int] = []
        for piece in _SPLIT.findall(normalized):
            as_bytes = "".join(
                self._byte_encoder[byte] for byte in piece.encode("utf-8")
            )
            for token in self._bpe(as_bytes):
                token_id = self.vocab.get(token)
                if token_id is None:
                    for char in token:
                        ids.append(self.vocab[char])
                else:
                    ids.append(token_id)
        return ids

    def decode(self, token_ids: list[int]) -> str:
        """Decode token ids back to text.

        Args:
            token_ids: Ids produced by :meth:`encode` or by generation.

        Returns:
            Decoded string.
        """
        raw = "".join(self._id_to_token.get(token_id, "") for token_id in token_ids)
        data = bytearray()
        for char in raw:
            byte = self._byte_decoder.get(char)
            if byte is None:
                data.extend(char.encode("utf-8"))
            else:
                data.append(byte)
        return data.decode("utf-8", errors="replace")

    def token_text(self, token_id: int) -> str | None:
        """Decode a single id to the text it contributes.

        Args:
            token_id: Vocabulary id.

        Returns:
            Text fragment, or None if the id is unknown.
        """
        if token_id not in self._id_to_token:
            return None
        return self.decode([token_id])

    def id_to_text_map(self) -> dict[int, str]:
        """Precompute id → text for every vocab entry.

        Returns:
            Mapping used by constrained decoding.
        """
        return {
            token_id: self.decode([token_id])
            for token_id in self._id_to_token
        }

    def _bpe(self, token: str) -> tuple[str, ...]:
        """Apply merge rules to one pre-token.

        Args:
            token: Byte-level unicode string.

        Returns:
            Tuple of merged token strings.
        """
        cached = self._bpe_cache.get(token)
        if cached is not None:
            return cached
        word: tuple[str, ...] = tuple(token)
        if len(word) < 2:
            self._bpe_cache[token] = word
            return word
        while True:
            pairs = _pairs(word)
            bigram = min(pairs, key=lambda pair: self._ranks.get(pair, 10**12))
            if bigram not in self._ranks:
                break
            word = _merge(word, bigram)
            if len(word) < 2:
                break
        self._bpe_cache[token] = word
        return word


def _pairs(word: tuple[str, ...]) -> set[tuple[str, str]]:
    """Return adjacent symbol pairs inside a BPE word."""
    return {(word[i], word[i + 1]) for i in range(len(word) - 1)}


def _merge(word: tuple[str, ...], pair: tuple[str, str]) -> tuple[str, ...]:
    """Merge every non-overlapping occurrence of ``pair`` in ``word``."""
    first, second = pair
    merged: list[str] = []
    index = 0
    while index < len(word):
        if (
            index < len(word) - 1
            and word[index] == first
            and word[index + 1] == second
        ):
            merged.append(first + second)
            index += 2
        else:
            merged.append(word[index])
            index += 1
    return tuple(merged)


def _load_vocab(path: str) -> dict[str, int]:
    """Read vocab.json."""
    with Path(path).open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Unexpected vocab format: {path}")
    return {str(token): int(token_id) for token, token_id in data.items()}


def _load_merges(path: str) -> list[tuple[str, str]]:
    """Read merges.txt, skipping the version header."""
    pairs: list[tuple[str, str]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            left, right = stripped.split(" ", 1)
            pairs.append((left, right))
    return pairs


@lru_cache(maxsize=1)
def gpt2_space_char() -> str:
    """Return the unicode character Qwen uses for a leading space (Ġ)."""
    return bytes_to_unicode()[ord(" ")]
