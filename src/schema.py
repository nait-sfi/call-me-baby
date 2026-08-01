from pydantic import BaseModel, model_validator, Field, ConfigDict
from typing import Dict, Optional, Any, Literal


class FunctionReturn(BaseModel):
    type: Literal["string", "number", "boolean", "integer"]
    model_config = ConfigDict(extra="forbid")


class ParameterDetail(BaseModel):
    type: Literal["string", "number", "boolean", "integer"]
    value: Optional[Any] = None
    model_config = ConfigDict(extra="forbid")


class FunctionDef(BaseModel):
    name: str = Field(min_length=2)
    description: str
    parameters: Dict[str, ParameterDetail]
    returns: FunctionReturn
    model_config = ConfigDict(extra="forbid")

    def to_dict(self) -> Dict[str, list]:
        return {
            self.name: [
                (param, param_type.type)
                for param, param_type in self.parameters.items()
            ]
        }


class Prompt(BaseModel):
    prompt: str
    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def parse_prompt(self):
        if not self.prompt.strip():
            raise ValueError("Error: empty prompt")
        return self
