"""Prefix checker: is generated text a valid prefix of a schema-valid call?"""

from __future__ import annotations

import json
from dataclasses import dataclass

from src.models import FunctionDefinition, ParameterSpec

_WS = frozenset(" \t\n\r")


@dataclass
class ParseResult:
    """Result of checking a partial JSON function-call string."""

    ok: bool
    complete: bool = False
    parsed: dict | None = None


class FunctionCallPrefixChecker:
    """Validate that a string is a prefix of some legal function-call JSON.

    Expected shape (whitespace allowed between tokens)::

        {"name": "<fname>", "parameters": {"p1": <typed>, ...}}

    Parameter keys must appear in definition order for the chosen function.
    """

    def __init__(self, functions: list[FunctionDefinition]) -> None:
        """Initialize with available function schemas.

        Args:
            functions: Function definitions used for name/type constraints.
        """
        self._functions = {fn.name: fn for fn in functions}
        self._names = list(self._functions.keys())

    def check(self, text: str) -> ParseResult:
        """Check whether ``text`` is a valid (possibly partial) call.

        Args:
            text: Generated text so far.

        Returns:
            ParseResult with ok/complete flags and parsed object if done.
        """
        i = 0
        n = len(text)

        def skip_ws(pos: int) -> int:
            while pos < n and text[pos] in _WS:
                pos += 1
            return pos

        def need(pos: int, literal: str) -> int | None:
            """Match literal or a proper prefix of it at end of text."""
            pos = skip_ws(pos)
            remaining = text[pos:]
            if remaining.startswith(literal):
                return pos + len(literal)
            if literal.startswith(remaining) and pos + len(remaining) == n:
                # consumed rest of text as prefix of literal — OK partial
                return -1  # sentinel: valid partial end
            return None

        def end_partial() -> ParseResult:
            return ParseResult(ok=True, complete=False)

        # {
        i = skip_ws(i)
        if i >= n:
            return end_partial()
        got = need(i, "{")
        if got is None:
            return ParseResult(ok=False)
        if got == -1:
            return end_partial()
        i = got

        # "name"
        got = need(i, '"name"')
        if got is None:
            return ParseResult(ok=False)
        if got == -1:
            return end_partial()
        i = got

        # :
        got = need(i, ":")
        if got is None:
            return ParseResult(ok=False)
        if got == -1:
            return end_partial()
        i = got

        # " <function name> "
        i = skip_ws(i)
        if i >= n:
            return end_partial()
        if text[i] != '"':
            return ParseResult(ok=False)
        i += 1
        name_start = i
        while i < n and text[i] != '"':
            if text[i] in "\n\r":
                return ParseResult(ok=False)
            i += 1
        name_so_far = text[name_start:i]
        if i >= n:
            # unclosed name string — must be prefix of some function name
            if any(name.startswith(name_so_far) for name in self._names):
                return end_partial()
            return ParseResult(ok=False)
        # closed quote
        i += 1
        fname = name_so_far
        if fname not in self._functions:
            return ParseResult(ok=False)
        fn = self._functions[fname]
        param_names = list(fn.parameters.keys())

        # ,
        got = need(i, ",")
        if got is None:
            return ParseResult(ok=False)
        if got == -1:
            return end_partial()
        i = got

        # "parameters"
        got = need(i, '"parameters"')
        if got is None:
            return ParseResult(ok=False)
        if got == -1:
            return end_partial()
        i = got

        # :
        got = need(i, ":")
        if got is None:
            return ParseResult(ok=False)
        if got == -1:
            return end_partial()
        i = got

        # {
        got = need(i, "{")
        if got is None:
            return ParseResult(ok=False)
        if got == -1:
            return end_partial()
        i = got

        # parameters body
        for idx, pname in enumerate(param_names):
            i = skip_ws(i)
            if i >= n:
                return end_partial()

            # optional comma before params after the first
            if idx > 0:
                got = need(i, ",")
                if got is None:
                    return ParseResult(ok=False)
                if got == -1:
                    return end_partial()
                i = got

            # "pname"
            got = need(i, f'"{pname}"')
            if got is None:
                return ParseResult(ok=False)
            if got == -1:
                return end_partial()
            i = got

            # :
            got = need(i, ":")
            if got is None:
                return ParseResult(ok=False)
            if got == -1:
                return end_partial()
            i = got

            i = skip_ws(i)
            if i >= n:
                return end_partial()

            value_end = self._match_value_prefix(text, i, fn.parameters[pname])
            if value_end is None:
                return ParseResult(ok=False)
            if value_end == -1:
                return end_partial()
            i = value_end

        # }
        got = need(i, "}")
        if got is None:
            return ParseResult(ok=False)
        if got == -1:
            return end_partial()
        i = got

        # }
        got = need(i, "}")
        if got is None:
            return ParseResult(ok=False)
        if got == -1:
            return end_partial()
        i = got

        i = skip_ws(i)
        if i < n:
            # trailing junk
            return ParseResult(ok=False)

        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return ParseResult(ok=False)
        return ParseResult(ok=True, complete=True, parsed=parsed)

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
            spec: Parameter schema (type, and optionally nested fields).

        Returns:
            Index after the value, -1 if text ends inside a valid partial value,
            or None if invalid.
        """
        n = len(text)
        if start >= n:
            return -1

        ptype = spec.type
        if ptype == "number":
            return self._match_number(text, start)
        if ptype == "boolean":
            return self._match_bool(text, start)
        if ptype == "string":
            return self._match_string(text, start)
        # fallback: treat as string
        return self._match_string(text, start)

    @staticmethod
    def _match_number(text: str, start: int) -> int | None:
        """Match a JSON number or a valid incomplete number prefix."""
        n = len(text)
        i = start
        if text[i] == "-":
            i += 1
            if i >= n:
                return -1
        if i >= n:
            return -1
        if not text[i].isdigit():
            return None
        i += 1
        while i < n and text[i].isdigit():
            i += 1
        if i < n and text[i] == ".":
            i += 1
            if i >= n:
                return -1
            if not text[i].isdigit():
                # allow trailing dot only as partial ("2.")
                return None
            while i < n and text[i].isdigit():
                i += 1
        # exponent not required for this project
        if i >= n:
            # complete-looking number at EOF is still partial structurally
            # (caller continues with comma/brace). Treat as finished value.
            return i
        # peek: number ends before non-number char
        return i

    @staticmethod
    def _match_bool(text: str, start: int) -> int | None:
        """Match true/false or a prefix thereof."""
        rest = text[start:]
        for lit in ("true", "false"):
            if rest.startswith(lit):
                return start + len(lit)
            if lit.startswith(rest):
                return -1
        return None

    @staticmethod
    def _match_string(text: str, start: int) -> int | None:
        """Match a JSON string or an unclosed valid string prefix."""
        n = len(text)
        if text[start] != '"':
            return None
        i = start + 1
        while i < n:
            ch = text[i]
            if ch == "\\":
                if i + 1 >= n:
                    return -1
                i += 2
                continue
            if ch == '"':
                return i + 1
            if ch == "\n" or ch == "\r":
                return None
            i += 1
        return -1  # unclosed string — OK partial
