from typing import List

from pydantic import BaseModel


class ParseRequest(BaseModel):
    text: str


class ParseResponse(BaseModel):
    status: str = "success"
    data: List[str]
