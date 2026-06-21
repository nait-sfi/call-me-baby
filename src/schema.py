from pydantic import BaseModel
from typing import Dict


class FunctionReturn(BaseModel):
    type: str


class ParameterDetail(BaseModel):
    type: str


class FunctionDef(BaseModel):
    name: str
    description: str
    parameters: Dict[str, ParameterDetail]
    returns: FunctionReturn
