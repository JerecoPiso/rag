from pydantic import BaseModel
from typing import Literal

class TranscriptionResponse(BaseModel):
    text:     str
    language: str
    duration: float
    cost_usd: float | None = None
    audio:    str | None = None  # data URI (e.g. "data:audio/mpeg;base64,...") of the transcribed text spoken back
