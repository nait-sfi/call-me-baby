"""Pydantic models for input validation."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class FunctionReturn(BaseModel):
    """Describe a function return type."""

    type: Literal["string", "number", "boolean", "integer"]
    model_config = ConfigDict(extra="forbid")


class ParameterDetail(BaseModel):
    """Describe a function parameter."""

    type: Literal["string", "number", "boolean", "integer"]
    value: Any | None = None
    model_config = ConfigDict(extra="forbid")


class FunctionDef(BaseModel):
    """Validate a function definition."""

    name: str = Field(min_length=2)
    description: str
    parameters: dict[str, ParameterDetail]
    returns: FunctionReturn
    model_config = ConfigDict(extra="forbid")

    def to_dict(self) -> dict[str, list[tuple[str, str]]]:
        """Convert the function definition into tokenizable metadata."""
        return {
            self.name: [
                (param, param_type.type)
                for param, param_type in self.parameters.items()
            ]
        }

    def get_parameter_types(self) -> list[tuple[str, str]]:
        """Return the function parameters as name/type pairs."""
        return [
            (param, param_type.type)
            for param, param_type in self.parameters.items()
        ]


class Prompt(BaseModel):
    """Validate a single prompt."""

    prompt: str
    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def parse_prompt(self) -> "Prompt":
        """Reject empty prompts."""
        if not self.prompt.strip():
            raise ValueError("Error: empty prompt")
        return self
