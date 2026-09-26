"""Build the natural-language prompt sent to the LLM."""

from __future__ import annotations

from src.models import FunctionDefinition


def format_functions(functions: list[FunctionDefinition]) -> str:
    """Render function definitions for the model prompt.

    Args:
        functions: Available tools.

    Returns:
        Multi-line description of each function.
    """
    lines: list[str] = []
    for fn in functions:
        params = ", ".join(
            f"{name}: {spec.type}" for name, spec in fn.parameters.items()
        )
        lines.append(f"- {fn.name}({params}): {fn.description}")
    return "\n".join(lines)


def build_prompt(user_prompt: str, functions: list[FunctionDefinition]) -> str:
    """Create the full prompt that asks the model for a JSON function call.

    Args:
        user_prompt: Natural-language user request.
        functions: Available function definitions.

    Returns:
        Prompt string to encode and feed to the model.
    """
    catalog = format_functions(functions)
    return (
        "You are a function-calling system. "
        "Convert the user request into exactly one JSON function call.\n"
        "Rules:\n"
        "- Output ONLY a single JSON object, no markdown, no explanation.\n"
        '- Format: {"name":"<function>","parameters":{...}}\n'
        "- Use double quotes. Match parameter names and types exactly.\n"
        "- Numbers must be JSON numbers. Strings must be JSON strings.\n"
        "- For regex substitution: source_string is the text, regex is the "
        "pattern to find, replacement is what to put instead.\n"
        "- Prefer general patterns (e.g. [0-9]+ for digits, "
        "[aeiouAEIOU] for vowels) when the user describes a class of matches.\n"
        "- When replacing a whole word, use that word as the regex "
        "(e.g. cat), not a character class like [cat].\n"
        "Examples:\n"
        '- User: What is the sum of 2 and 3?\n'
        '  {"name":"fn_add_numbers","parameters":{"a":2.0,"b":3.0}}\n'
        '- User: Replace all numbers in "ab 12 cd" with NUM\n'
        '  {"name":"fn_substitute_string_with_regex",'
        '"parameters":{"source_string":"ab 12 cd","regex":"[0-9]+",'
        '"replacement":"NUM"}}\n'
        '- User: Substitute the word cat with dog in "a cat"\n'
        '  {"name":"fn_substitute_string_with_regex",'
        '"parameters":{"source_string":"a cat","regex":"cat",'
        '"replacement":"dog"}}\n'
        f"Available functions:\n{catalog}\n\n"
        f"User request: {user_prompt}\n\n"
        "JSON:"
    )
