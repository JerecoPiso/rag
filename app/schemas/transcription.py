from pydantic import BaseModel
from typing import Literal

class TranscriptionResponse(BaseModel):
    text:     str
    language: str
    duration: float
