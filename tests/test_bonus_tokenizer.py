"""The recoded tokenizer must match Qwen encode/decode on real prompts."""

from __future__ import annotations

import unittest

from llm_sdk import Small_LLM_Model  # type: ignore[attr-defined]

from src.bonus.tokenizer import BonusTokenizer, TokenizerFiles


class TokenizerParityTest(unittest.TestCase):
    """Bonus encode/decode agrees with the SDK on the strings we generate."""

    model: Small_LLM_Model
    tokenizer: BonusTokenizer

    @classmethod
    def setUpClass(cls) -> None:
        cls.model = Small_LLM_Model()
        cls.tokenizer = BonusTokenizer.from_files(
            TokenizerFiles(
                vocab_path=cls.model.get_path_to_vocab_file(),
                merges_path=cls.model.get_path_to_merges_file(),
            )
        )

    def test_encode_matches_sdk(self) -> None:
        samples = [
            "{",
            '{"name":"fn_add_numbers","parameters":{"a":2,"b":3}}',
            "What is the sum of 2 and 3?",
            "Greet shrek",
            "fn_add_numbers",
            "Hello 34 I'm 233 years old",
        ]
        for text in samples:
            expected = self.model.encode(text).tolist()[0]
            got = self.tokenizer.encode(text)
            self.assertEqual(got, expected, text)

    def test_decode_roundtrip(self) -> None:
        text = '{"name":"fn_greet","parameters":{"name":"shrek"}}'
        ids = self.tokenizer.encode(text)
        self.assertEqual(self.tokenizer.decode(ids), text)
        self.assertEqual(self.model.decode(ids), text)
