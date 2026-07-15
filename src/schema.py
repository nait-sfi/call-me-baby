from pydantic import BaseModel, model_validator, Field
from typing import Dict, Optional, Any


class FunctionReturn(BaseModel):
    type: str


class ParameterDetail(BaseModel):
    type: str
    value: Optional[Any] = None


class FunctionDef(BaseModel):
    name: str = Field(min_length=2)
    description: str
    parameters: Dict[str, ParameterDetail]
    returns: FunctionReturn
    
    def to_dict(self):
        return {
            self.name: [(param, param_type["type"]) for param, param_type in self.parameters.items()]
        }


class Prompt(BaseModel):
    prompt: str

    @model_validator(mode="after")
    def parse_prompt(self):
        if not self.prompt.strip():
            raise ValueError("Error: empty prompt")
        return self
