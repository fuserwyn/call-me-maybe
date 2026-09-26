"""Pydantic models for function definitions, prompts, and results."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ParameterSpec(BaseModel):
    """Schema for a single function parameter."""

    model_config = ConfigDict(extra="forbid")

    type: str = Field(..., description="JSON type: number, string, boolean, etc.")


class ReturnSpec(BaseModel):
    """Schema for a function return value."""

    model_config = ConfigDict(extra="forbid")

    type: str


class FunctionDefinition(BaseModel):
    """One callable tool exposed to the LLM."""

    model_config = ConfigDict(extra="forbid")

    name: str
    description: str
    parameters: dict[str, ParameterSpec]
    returns: ReturnSpec


class PromptItem(BaseModel):
    """A natural-language request from the test suite."""

    model_config = ConfigDict(extra="forbid")

    prompt: str


class FunctionCallResult(BaseModel):
    """One structured function call produced by the decoder."""

    model_config = ConfigDict(extra="forbid")

    prompt: str
    name: str
    parameters: dict[str, Any]
