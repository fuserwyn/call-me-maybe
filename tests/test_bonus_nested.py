"""Tests for nested arguments, recovery, and the constraint cache."""

from __future__ import annotations

import unittest

from src.bonus.cache import ConstraintCache
from src.bonus.nested import NestedCallChecker, closes_partial
from src.bonus.recovery import BacktrackBudget, SearchFrame
from src.models import FunctionDefinition, ParameterSpec, ReturnSpec


def _profile() -> list[FunctionDefinition]:
    return [
        FunctionDefinition(
            name="fn_profile",
            description="Store a user.",
            parameters={
                "user": ParameterSpec(
                    type="object",
                    properties={
                        "name": ParameterSpec(type="string"),
                        "age": ParameterSpec(type="number"),
                    },
                ),
                "tags": ParameterSpec(
                    type="array",
                    items=ParameterSpec(type="string"),
                ),
            },
            returns=ReturnSpec(type="string"),
        )
    ]


class NestedCheckerTest(unittest.TestCase):
    """Nested object and array prefixes stay schema-valid."""

    def setUp(self) -> None:
        self.checker = NestedCallChecker(_profile())

    def test_complete_nested_call(self) -> None:
        text = (
            '{"name":"fn_profile","parameters":'
            '{"user":{"name":"sam","age":3},"tags":["a","b"]}}'
        )
        result = self.checker.check(text)
        self.assertTrue(result.ok)
        self.assertTrue(result.complete)
        self.assertIsNotNone(result.parsed)
        assert result.parsed is not None
        self.assertEqual(result.parsed["parameters"]["user"]["age"], 3)

    def test_partial_object_is_allowed(self) -> None:
        text = '{"name":"fn_profile","parameters":{"user":{"name":"sa'
        self.assertTrue(self.checker.check(text).ok)
        self.assertFalse(self.checker.check(text).complete)

    def test_wrong_nested_key_is_rejected(self) -> None:
        text = '{"name":"fn_profile","parameters":{"user":{"nope":'
        self.assertFalse(self.checker.check(text).ok)

    def test_cache_remembers_a_prefix(self) -> None:
        cache = ConstraintCache()
        prefix = '{"name":"fn_profile"'
        self.assertTrue(cache.allows(self.checker, prefix, ","))
        self.assertTrue(cache.allows(self.checker, prefix, ","))
        self.assertEqual(cache.hits, 1)
        self.assertEqual(cache.misses, 1)

    def test_backtrack_budget_and_frame(self) -> None:
        budget = BacktrackBudget(max_backtracks=1)
        self.assertTrue(budget.spend())
        self.assertFalse(budget.spend())
        frame = SearchFrame(
            token_ids=[],
            text="",
            options=[7, 8],
            logits={7: 1.0, 8: 0.5},
        )
        self.assertEqual(frame.take(), 7)
        self.assertEqual(frame.take(), 8)
        self.assertIsNone(frame.take())


class RecoveryTest(unittest.TestCase):
    """Closing braces can repair a truncated object."""

    def test_closes_open_braces(self) -> None:
        raw = '{"name":"fn_profile","parameters":{"user":{"name":"sam","age":1}'
        result = closes_partial(raw)
        self.assertTrue(result.complete)
        self.assertIsNotNone(result.parsed)
