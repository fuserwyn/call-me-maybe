"""Prefix checks for nested object and array parameters."""

from __future__ import annotations

from src.constraints import FunctionCallPrefixChecker, ParseResult
from src.models import ParameterSpec

_WS = frozenset(" \t\n\r")


class NestedCallChecker(FunctionCallPrefixChecker):
    """Same call grammar as the mandatory checker, plus object/array values."""

    def _match_value_prefix(
        self,
        text: str,
        start: int,
        spec: ParameterSpec,
    ) -> int | None:
        """Match a value, including nested objects and arrays.

        Args:
            text: Generated text.
            start: Index of the value.
            spec: Parameter schema.

        Returns:
            End index, -1 for a valid partial, or None if invalid.
        """
        if spec.type == "object" and spec.properties:
            return self._match_object(text, start, spec)
        if spec.type == "array" and spec.items is not None:
            return self._match_array(text, start, spec)
        return super()._match_value_prefix(text, start, spec)

    def _match_object(
        self,
        text: str,
        start: int,
        spec: ParameterSpec,
    ) -> int | None:
        """Match a JSON object whose keys follow ``spec.properties`` order."""
        props = list((spec.properties or {}).items())
        index = start
        if index >= len(text):
            return -1
        if text[index] != "{":
            return None
        index += 1
        for position, (name, child) in enumerate(props):
            index = _skip_ws(text, index)
            if index >= len(text):
                return -1
            if position > 0:
                comma = _need(text, index, ",")
                if comma is None:
                    return None
                if comma == -1:
                    return -1
                index = comma
            key = _need(text, index, f'"{name}"')
            if key is None:
                return None
            if key == -1:
                return -1
            index = key
            colon = _need(text, index, ":")
            if colon is None:
                return None
            if colon == -1:
                return -1
            index = colon
            index = _skip_ws(text, index)
            if index >= len(text):
                return -1
            value_end = self._match_value_prefix(text, index, child)
            if value_end is None:
                return None
            if value_end == -1:
                return -1
            index = value_end
        closing = _need(text, index, "}")
        if closing is None:
            return None
        if closing == -1:
            return -1
        return closing

    def _match_array(
        self,
        text: str,
        start: int,
        spec: ParameterSpec,
    ) -> int | None:
        """Match a JSON array of ``spec.items``."""
        item = spec.items
        if item is None:
            return None
        index = start
        if index >= len(text):
            return -1
        if text[index] != "[":
            return None
        index += 1
        index = _skip_ws(text, index)
        if index >= len(text):
            return -1
        if text[index] == "]":
            return index + 1
        while True:
            value_end = self._match_value_prefix(text, index, item)
            if value_end is None:
                return None
            if value_end == -1:
                return -1
            index = value_end
            index = _skip_ws(text, index)
            if index >= len(text):
                return -1
            if text[index] == "]":
                return index + 1
            if text[index] != ",":
                return None
            index += 1
            index = _skip_ws(text, index)
            if index >= len(text):
                return -1


def repair_json(text: str) -> str:
    """Close an open string and any dangling brackets.

    Args:
        text: Incomplete generation.

    Returns:
        Text with a closing quote and enough ``]`` / ``}`` to balance.
    """
    repaired = text.rstrip()
    if repaired.count('"') % 2 == 1:
        repaired += '"'
    open_square = repaired.count("[") - repaired.count("]")
    open_curly = repaired.count("{") - repaired.count("}")
    if open_square > 0:
        repaired += "]" * open_square
    if open_curly > 0:
        repaired += "}" * open_curly
    return repaired


def closes_partial(text: str) -> ParseResult:
    """Try to finish a truncated JSON object by closing strings and braces.

    Args:
        text: Incomplete generation.

    Returns:
        Parse result of the repaired string when it becomes valid JSON.
    """
    repaired = repair_json(text)
    try:
        import json

        parsed = json.loads(repaired)
    except json.JSONDecodeError:
        return ParseResult(ok=False)
    if not isinstance(parsed, dict):
        return ParseResult(ok=False)
    return ParseResult(ok=True, complete=True, parsed=parsed)


def _skip_ws(text: str, index: int) -> int:
    """Advance past spaces, tabs, and newlines."""
    while index < len(text) and text[index] in _WS:
        index += 1
    return index


def _need(text: str, index: int, literal: str) -> int | None:
    """Match ``literal`` or a proper prefix of it at the end of ``text``."""
    index = _skip_ws(text, index)
    remaining = text[index:]
    if remaining.startswith(literal):
        return index + len(literal)
    if literal.startswith(remaining) and index + len(remaining) == len(text):
        return -1
    return None
