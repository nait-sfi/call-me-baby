"""Pydantic models for input validation."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class FunctionReturn(BaseModel):
    """Describe a function return type.

    Attributes:
        type (Literal["string", "number", "boolean", "integer"]): Declared
            return type.
    """

    type: Literal["string", "number", "boolean", "integer"]
    model_config = ConfigDict(extra="forbid")


class ParameterDetail(BaseModel):
    """Describe a function parameter.

    Attributes:
        type (Literal["string", "number", "boolean", "integer"]): Declared
            parameter type.
        value (Any | None): Optional default/example parameter value.
    """

    type: Literal["string", "number", "boolean", "integer"]
    value: Any | None = None
    model_config = ConfigDict(extra="forbid")


class FunctionDef(BaseModel):
    """Validate a function definition.

    Attributes:
        name (str): Function name.
        description (str): Human-readable function description.
        parameters (dict[str, ParameterDetail]): Parameter definitions.
        returns (FunctionReturn): Return value schema.
    """

    name: str = Field(min_length=2)
    description: str
    parameters: dict[str, ParameterDetail]
    returns: FunctionReturn
    model_config = ConfigDict(extra="forbid")


class Prompt(BaseModel):
    """Validate a single prompt.

    Attributes:
        prompt (str): User prompt text.
    """

    prompt: str
    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def parse_prompt(self) -> "Prompt":
        """Reject empty prompts.

        Returns:
            Prompt: Validated prompt instance.
        """
        if not self.prompt.strip():
            raise ValueError("Error: empty prompt")
        return self
