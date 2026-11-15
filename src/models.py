"""Pydantic models for function definitions, prompts, and results."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ParameterSpec(BaseModel):
    """Schema for a single function parameter.

    ``enum`` restricts the value to a fixed list. A missing ``type`` means any
    JSON value. Other JSON-schema keywords (``minimum`` ...) are ignored so
    that other function sets still load.
    """

    model_config = ConfigDict(extra="ignore")

    type: str = Field("any", description="JSON type: number, string, boolean, etc.")
    description: str | None = None
    enum: list[Any] | None = Field(
        None, description="Allowed values; the generated value must be one of them."
    )


class ReturnSpec(BaseModel):
    """Schema for a function return value."""

    model_config = ConfigDict(extra="ignore")

    type: str


class FunctionDefinition(BaseModel):
    """One callable tool exposed to the LLM.

    Extra top-level keys such as ``required`` are accepted and ignored:
    every declared parameter is always generated. ``description``,
    ``parameters`` and ``returns`` may be missing.
    """

    model_config = ConfigDict(extra="ignore")

    name: str = Field(..., min_length=1)
    description: str = ""
    parameters: dict[str, ParameterSpec] = Field(default_factory=dict)
    returns: ReturnSpec | None = None


class PromptItem(BaseModel):
    """A natural-language request from the test suite."""

    model_config = ConfigDict(extra="ignore")

    prompt: str


class FunctionCallResult(BaseModel):
    """One structured function call produced by the decoder."""

    model_config = ConfigDict(extra="forbid")

    prompt: str
    name: str
    parameters: dict[str, Any]
