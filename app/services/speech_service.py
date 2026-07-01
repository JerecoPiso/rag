from fastapi import HTTPException
from openai import OpenAI
from app.core.config import settings

_openai_client: OpenAI | None = None


def _get_openai_client() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI(api_key=settings.OPENAI_API_KEY)
    return _openai_client


_VALID_VOICES = {"alloy", "ash", "coral", "echo", "fable", "onyx", "nova", "sage", "shimmer"}

_FORMAT_MEDIA_TYPES = {
    "mp3":  "audio/mpeg",
    "opus": "audio/opus",
    "aac":  "audio/aac",
    "flac": "audio/flac",
    "wav":  "audio/wav",
    "pcm":  "audio/pcm",
}


class SpeechService:
    def synthesize(
        self,
        text: str,
        voice: str = "alloy",
        model: str = "tts-1",
        audio_format: str = "mp3",
    ) -> tuple[bytes, str]:
        if voice not in _VALID_VOICES:
            raise HTTPException(status_code=400, detail=f"Unsupported voice: {voice}")
        if audio_format not in _FORMAT_MEDIA_TYPES:
            raise HTTPException(status_code=400, detail=f"Unsupported format: {audio_format}")

        client = _get_openai_client()
        response = client.audio.speech.create(
            model=model,
            voice=voice,
            input=text,
            response_format=audio_format,
        )
        return response.read(), _FORMAT_MEDIA_TYPES[audio_format]
