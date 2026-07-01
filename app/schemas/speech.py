from pydantic import BaseModel, Field


class SpeechRequest(BaseModel):
    text:   str = Field(..., min_length=1, max_length=4096)
    voice:  str = "alloy"
    model:  str = "tts-1"
    format: str = "mp3"
