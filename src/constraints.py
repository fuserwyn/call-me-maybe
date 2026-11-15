"""Prefix checker: is generated text a valid prefix of a schema-valid call?

The checker is *prefix-closed*: it only says "ok" for text that can still be
completed into a call ``json.loads`` accepts. Otherwise generation could walk
into a dead end (``01``, ``"\\q``, a raw tab in a string) and only discover it
at the closing brace.

Every ``_match_*`` helper returns the index right after the value, ``-1``
when the text ends inside a value that is still valid so far, or ``None``
when the text can no longer be valid.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, ConfigDict, PrivateAttr

from src.models import FunctionDefinition, ParameterSpec

_WS = frozenset(" \t\n\r")
_SIMPLE_ESCAPES = frozenset('"\\/bfnrt')
_HEX = frozenset("0123456789abcdefABCDEF")
_PARTIAL = -1


class ParseResult(BaseModel):
    """Result of checking a partial JSON function-call string."""

    model_config = ConfigDict(extra="forbid")

    ok: bool
    complete: bool = False
    parsed: dict[str, Any] | None = None


def skip_ws(text: str, pos: int) -> int:
    """Advance past JSON whitespace.

    Args:
        text: Generated text.
        pos: Start index.

    Returns:
        Index of the first non-whitespace character (or ``len(text)``).
    """
    while pos < len(text) and text[pos] in _WS:
        pos += 1
    return pos


def need(text: str, pos: int, literal: str) -> int | None:
    """Match ``literal`` (after optional whitespace) at ``pos``.

    Args:
        text: Generated text.
        pos: Start index.
        literal: Exact text that must come next.

    Returns:
        Index after the literal, ``-1`` if the text ends inside it, or None.
    """
    pos = skip_ws(text, pos)
    remaining = text[pos:]
    if remaining.startswith(literal):
        return pos + len(literal)
    if literal.startswith(remaining):
        return _PARTIAL
    return None


def match_digits(text: str, pos: int) -> int:
    """Return the index after a run of ASCII digits starting at ``pos``."""
    while pos < len(text) and "0" <= text[pos] <= "9":
        pos += 1
    return pos


def match_number(text: str, start: int, *, integer: bool = False) -> int | None:
    """Match a JSON number: ``-? (0 | [1-9][0-9]*) (.[0-9]+)? ([eE][+-]?[0-9]+)?``.

    Args:
        text: Generated text.
        start: Index of the first character of the number.
        integer: Forbid the fraction and exponent parts.

    Returns:
        Index after the number, ``-1`` for a valid partial, or None.
    """
    n = len(text)
    i = start
    if i < n and text[i] == "-":
        i += 1
    if i >= n:
        return _PARTIAL
    if text[i] == "0":
        i += 1
    elif "1" <= text[i] <= "9":
        i = match_digits(text, i + 1)
    else:
        return None
    if integer:
        return i
    if i < n and text[i] == ".":
        i += 1
        if i >= n:
            return _PARTIAL
        end = match_digits(text, i)
        if end == i:
            return None
        i = end
    if i < n and text[i] in "eE":
        i += 1
        if i < n and text[i] in "+-":
            i += 1
        if i >= n:
            return _PARTIAL
        end = match_digits(text, i)
        if end == i:
            return None
        i = end
    return i


def match_literal(text: str, start: int, choices: tuple[str, ...]) -> int | None:
    """Match one of several keywords (``true``, ``false``, ``null``).

    Args:
        text: Generated text.
        start: Index of the keyword.
        choices: Allowed keywords.

    Returns:
        Index after the keyword, ``-1`` for a valid partial, or None.
    """
    rest = text[start:]
    # Longest first, so "10" is not read as "1" followed by junk.
    for word in sorted(choices, key=len, reverse=True):
        if rest.startswith(word):
            return start + len(word)
    if any(word.startswith(rest) for word in choices):
        return _PARTIAL
    return None


def match_string(text: str, start: int) -> int | None:
    """Match a JSON string with valid escapes and no raw control characters.

    Args:
        text: Generated text.
        start: Index of the opening quote.

    Returns:
        Index after the closing quote, ``-1`` if still open, or None.
    """
    n = len(text)
    if start >= n:
        return _PARTIAL
    if text[start] != '"':
        return None
    i = start + 1
    while i < n:
        ch = text[i]
        if ch == '"':
            return i + 1
        if ord(ch) < 0x20:
            return None
        if ch != "\\":
            i += 1
            continue
        if i + 1 >= n:
            return _PARTIAL
        esc = text[i + 1]
        if esc in _SIMPLE_ESCAPES:
            i += 2
        elif esc == "u":
            digits = text[i + 2:i + 6]
            if any(char not in _HEX for char in digits):
                return None
            if len(digits) < 4:
                return _PARTIAL
            i += 6
        else:
            return None
    return _PARTIAL


def match_any_value(text: str, start: int) -> int | None:
    """Match any JSON value (used for types without a precise schema).

    Args:
        text: Generated text.
        start: Index of the value.

    Returns:
        Index after the value, ``-1`` for a valid partial, or None.
    """
    if start >= len(text):
        return _PARTIAL
    first = text[start]
    if first == '"':
        return match_string(text, start)
    if first == "-" or "0" <= first <= "9":
        return match_number(text, start)
    if first == "[":
        return match_sequence(text, start, "[", "]", keyed=False)
    if first == "{":
        return match_sequence(text, start, "{", "}", keyed=True)
    return match_literal(text, start, ("true", "false", "null"))


def match_sequence(
    text: str,
    start: int,
    opening: str,
    closing: str,
    *,
    keyed: bool,
) -> int | None:
    """Match a free-form JSON array or object.

    Args:
        text: Generated text.
        start: Index of the opening bracket.
        opening: ``[`` or ``{``.
        closing: ``]`` or ``}``.
        keyed: True for objects (``"key": value`` members).

    Returns:
        Index after the closing bracket, ``-1`` for a valid partial, or None.
    """
    n = len(text)
    if start >= n:
        return _PARTIAL
    if text[start] != opening:
        return None
    i = skip_ws(text, start + 1)
    if i >= n:
        return _PARTIAL
    if text[i] == closing:
        return i + 1
    while True:
        if keyed:
            key_end = match_string(text, i)
            if key_end is None or key_end == _PARTIAL:
                return key_end
            colon = need(text, key_end, ":")
            if colon is None or colon == _PARTIAL:
                return colon
            i = skip_ws(text, colon)
        value_end = match_any_value(text, i)
        if value_end is None or value_end == _PARTIAL:
            return value_end
        i = skip_ws(text, value_end)
        if i >= n:
            return _PARTIAL
        if text[i] == closing:
            return i + 1
        if text[i] != ",":
            return None
        i = skip_ws(text, i + 1)
        if i >= n:
            return _PARTIAL


class FunctionCallPrefixChecker(BaseModel):
    """Validate that a string is a prefix of some legal function-call JSON.

    Expected shape (whitespace allowed between tokens)::

        {"name": "<fname>", "parameters": {"p1": <typed>, ...}}

    Parameter keys must appear in definition order for the chosen function.
    """

    model_config = ConfigDict(extra="forbid")

    functions: list[FunctionDefinition]
    _by_name: dict[str, FunctionDefinition] = PrivateAttr(default_factory=dict)

    def __init__(self, functions: list[FunctionDefinition], **data: Any) -> None:
        """Initialize with available function schemas.

        Args:
            functions: Function definitions used for name/type constraints.
            **data: Extra pydantic fields (none for now).
        """
        super().__init__(functions=functions, **data)

    def model_post_init(self, context: Any, /) -> None:
        """Index the functions by name once the model is validated.

        Args:
            context: Pydantic validation context (unused).
        """
        self._by_name = {fn.name: fn for fn in self.functions}

    def check(self, text: str) -> ParseResult:
        """Check whether ``text`` is a valid (possibly partial) call.

        Args:
            text: Generated text so far.

        Returns:
            ParseResult with ok/complete flags and parsed object if done.
        """
        end = self._match_call(text)
        # Invalid, can never become a call: '{"nme"', '{"name":"unknown"',
        # '..."a":"2"' (string for a number) -> reject this token.
        if end is None:
            return ParseResult(ok=False)
        # Valid so far but unfinished: '{"na', '..."a":2.', '...{"a":2.0}'
        # (last } missing) -> accept, keep generating.
        if end == _PARTIAL:
            return ParseResult(ok=True)
        # Both } are closed, but something other than whitespace follows:
        # '...}}x', '...}},' -> reject. '...}} ' or '...}}\n' is fine.
        if skip_ws(text, end) < len(text):
            return ParseResult(ok=False)
        # Closed call, nothing after it: parse it into a dict. A ValueError
        # here would mean a gap in the grammar above (safety net).
        try:
            parsed = json.loads(text)
        except ValueError:
            return ParseResult(ok=False)
        # Complete call: generate() stops and returns it.
        return ParseResult(ok=True, complete=True, parsed=parsed)

    def _match_call(self, text: str) -> int | None:
        """Match the whole ``{"name": ..., "parameters": {...}}`` object.

        Args:
            text: Generated text so far.

        Returns:
            Index after the call, ``-1`` for a valid partial, or None.
        """
        i = 0
        for literal in ("{", '"name"', ":"):
            got = need(text, i, literal)
            if got is None or got == _PARTIAL:
                return got
            i = got

        fn = None
        i = skip_ws(text, i)
        if i >= len(text):
            return _PARTIAL
        if text[i] != '"':
            return None
        close = text.find('"', i + 1)
        typed = text[i + 1:] if close < 0 else text[i + 1:close]
        if close < 0:
            names = self._by_name
            return _PARTIAL if any(n.startswith(typed) for n in names) else None
        fn = self._by_name.get(typed)
        if fn is None:
            return None
        i = close + 1

        for literal in (",", '"parameters"', ":", "{"):
            got = need(text, i, literal)
            if got is None or got == _PARTIAL:
                return got
            i = got

        for index, (pname, spec) in enumerate(fn.parameters.items()):
            key = json.dumps(pname, ensure_ascii=False)
            for literal in ((",",) if index else ()) + (key, ":"):
                got = need(text, i, literal)
                if got is None or got == _PARTIAL:
                    return got
                i = got
            i = skip_ws(text, i)
            value_end = self._match_value_prefix(text, i, spec)
            if value_end is None or value_end == _PARTIAL:
                return value_end
            i = value_end

        for literal in ("}", "}"):
            got = need(text, i, literal)
            if got is None or got == _PARTIAL:
                return got
            i = got
        return i

    def _match_value_prefix(
        self,
        text: str,
        start: int,
        spec: ParameterSpec,
    ) -> int | None:
        """Match a JSON value of the given type, allowing incomplete prefixes.

        Args:
            text: Full generated text.
            start: Index where the value starts.
            spec: Parameter schema.

        Returns:
            Index after the value, -1 if text ends inside a valid partial value,
            or None if invalid.
        """
        if start >= len(text):
            return _PARTIAL
        if spec.enum is not None:
            allowed = tuple(json.dumps(v, ensure_ascii=False) for v in spec.enum)
            return match_literal(text, start, allowed)
        ptype = spec.type.lower()
        if ptype in ("number", "float"):
            return match_number(text, start)
        if ptype in ("integer", "int"):
            return match_number(text, start, integer=True)
        if ptype in ("boolean", "bool"):
            return match_literal(text, start, ("true", "false"))
        if ptype in ("string", "str"):
            return match_string(text, start)
        if ptype == "null":
            return match_literal(text, start, ("null",))
        if ptype == "array":
            return match_sequence(text, start, "[", "]", keyed=False)
        if ptype == "object":
            return match_sequence(text, start, "{", "}", keyed=True)
        return match_any_value(text, start)
