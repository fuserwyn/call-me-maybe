"""Byte-level BPE tokenizer rebuilt from vocab.json and merges.txt.

The bonus path never calls ``Small_LLM_Model.encode`` or ``decode``.
Token ids come from the public vocab file; merges come from
``get_path_to_merges_file``.
"""

from __future__ import annotations

import unicodedata
from functools import lru_cache
from pathlib import Path

import regex as re
from pydantic import BaseModel, ConfigDict, Field

_SPLIT = re.compile(
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)"
    r"|[^\r\n\p{L}\p{N}]?\p{L}+"
    r"|\p{N}"
    r"| ?[^\s\p{L}\p{N}]+[\r\n]*"
    r"|\s*[\r\n]+"
    r"|\s+(?!\S)"
    r"|\s+"
)


class TokenizerFiles(BaseModel):
    """Paths the SDK exposes for a local recode of the tokenizer."""

    model_config = ConfigDict(extra="forbid")

    vocab_path: str
    merges_path: str
    model_name: str = Field(default="Qwen/Qwen3-0.6B")


def bytes_to_unicode() -> dict[int, str]:
    """Map each byte to a unicode character, GPT-2 / Qwen byte-level style.

    Returns:
        Mapping from byte value 0..255 to a single character.
    """
    ordered = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    chars = ordered[:]
    extra = 0
    for byte in range(256):
        if byte not in ordered:
            ordered.append(byte)
            chars.append(256 + extra)
            extra += 1
    return {byte: chr(code) for byte, code in zip(ordered, chars)}


class BonusTokenizer:
    """Public encode/decode compatible with Qwen byte-level BPE."""

    def __init__(self, vocab: dict[str, int], merges: list[tuple[str, str]]) -> None:
        """Build lookup tables.

        Args:
            vocab: Token string to id, as stored in vocab.json.
            merges: Merge pairs in priority order (earlier = higher priority).
        """
        self._vocab = vocab
        self._id_to_token = {token_id: token for token, token_id in vocab.items()}
        self._ranks = {pair: rank for rank, pair in enumerate(merges)}
        self._byte_encoder = bytes_to_unicode()
        self._byte_decoder = {
            char: byte for byte, char in self._byte_encoder.items()
        }
        self._bpe_cache: dict[str, tuple[str, ...]] = {}

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
                token_id = self._vocab.get(token)
                if token_id is None:
                    for char in token:
                        ids.append(self._vocab[char])
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
    import json

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
