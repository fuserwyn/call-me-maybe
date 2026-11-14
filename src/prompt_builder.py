"""Build the natural-language prompt sent to the LLM.

Everything function-specific in the prompt is generated from the loaded
definitions, so the prompt never mentions functions that are not in
``functions_definition.json``.
"""

from __future__ import annotations

import json

from src.models import FunctionDefinition, ParameterSpec


def describe_type(spec: ParameterSpec) -> str:
    """Render a parameter type, including nested object/array shapes.

    Args:
        spec: Parameter schema.

    Returns:
        Short human-readable type, e.g. ``array of string``.
    """
    if spec.type == "array" and spec.items is not None:
        return f"array of {describe_type(spec.items)}"
    if spec.type == "object" and spec.properties:
        inner = ", ".join(
            f"{name}: {describe_type(child)}"
            for name, child in spec.properties.items()
        )
        return f"object {{{inner}}}"
    return spec.type


def call_template(fn: FunctionDefinition) -> str:
    """Show the exact JSON shape a call to ``fn`` must have.

    Args:
        fn: Function definition.

    Returns:
        JSON with ``<type>`` placeholders for every argument.
    """
    params = ",".join(
        f"{json.dumps(name, ensure_ascii=False)}:<{describe_type(spec)}>"
        for name, spec in fn.parameters.items()
    )
    name = json.dumps(fn.name, ensure_ascii=False)
    return f'{{"name":{name},"parameters":{{{params}}}}}'


def format_functions(functions: list[FunctionDefinition]) -> str:
    """Render function definitions for the model prompt.

    Args:
        functions: Available tools.

    Returns:
        Multi-line description of each function with its call shape.
    """
    lines: list[str] = []
    for fn in functions:
        lines.append(f"- {fn.name}: {fn.description}")
        for name, spec in fn.parameters.items():
            hint = f" - {spec.description}" if spec.description else ""
            lines.append(f"    {name} ({describe_type(spec)}){hint}")
        lines.append(f"    call: {call_template(fn)}")
    return "\n".join(lines)


def build_prompt(user_prompt: str, functions: list[FunctionDefinition]) -> str:
    """Create the full prompt that asks the model for a JSON function call.

    Args:
        user_prompt: Natural-language user request.
        functions: Available function definitions.

    Returns:
        Prompt string to encode and feed to the model.
    """
    return (
        "You are a function-calling system. Choose the ONE function that "
        "fulfils the user request and fill in its arguments.\n"
        "Rules:\n"
        "- Output only the JSON call, nothing else.\n"
        "- Take argument values from the request. Copy quoted text exactly, "
        "without the surrounding quotes.\n"
        "- Numbers are plain JSON numbers, not strings.\n"
        "- When an argument is a regular expression, write a pattern that "
        "matches what the user describes, not the words of the description. "
        "For a kind of character, use a character class that lists every "
        "member of that kind (e.g. [a-zA-Z] for letters). Use + to match a "
        "run of characters.\n"
        "- When the user names a symbol in words, use the symbol itself.\n\n"
        f"Available functions:\n{format_functions(functions)}\n\n"
        f"User request: {user_prompt}\n"
        "JSON:"
    )
