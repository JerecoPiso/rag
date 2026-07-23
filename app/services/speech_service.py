import io
import wave
from fastapi import HTTPException
from openai import OpenAI
from piper import PiperVoice
from app.core.config import settings

_openai_client: OpenAI | None = None
_piper_voice: PiperVoice | None = None


def _get_openai_client() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI(api_key=settings.OPENAI_API_KEY)
    return _openai_client


# PiperVoice.load() takes ~2s (loading the onnx model), so it's cached as a
# module-level singleton and reused across requests instead of reloaded per call.
def _get_piper_voice() -> PiperVoice:
    global _piper_voice
    if _piper_voice is None:
        _piper_voice = PiperVoice.load(settings.PIPER_VOICE_MODEL)
    return _piper_voice


_VALID_PROVIDERS = {"openai", "piper"}

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
        provider: str = "openai",
    ) -> tuple[bytes, str]:
        if provider not in _VALID_PROVIDERS:
            raise HTTPException(status_code=400, detail=f"Unsupported TTS provider: {provider}")

        if provider == "piper":
            return self._synthesize_piper(text)

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

    # Runs entirely locally via the onnx voice model — no network call, no API cost.
    def _synthesize_piper(self, text: str) -> tuple[bytes, str]:
        voice = _get_piper_voice()
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wav_file:
            voice.synthesize_wav(text, wav_file)
        return buf.getvalue(), _FORMAT_MEDIA_TYPES["wav"]
