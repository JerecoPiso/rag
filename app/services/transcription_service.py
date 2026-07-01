import os
import base64
import tempfile
from openai import OpenAI
from app.core.config import settings
from app.services.speech_service import SpeechService

_openai_client: OpenAI | None = None

def _get_openai_client() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI(api_key=settings.OPENAI_API_KEY)
    return _openai_client


def _transcribe_openai(audio_bytes: bytes, filename: str) -> dict:
    suffix = os.path.splitext(filename)[-1] or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name
    try:
        client = _get_openai_client()
        with open(tmp_path, "rb") as audio_file:
            response = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                language="en",
                response_format="verbose_json",
            )
        duration = round(getattr(response, "duration", 0.0), 2)
        return {
            "text":     response.text.lower(),
            "language": getattr(response, "language", None),
            "duration": duration,
            "cost_usd": round((duration / 60) * 0.006, 6),  # $0.006 per minute
        }
    finally:
        os.unlink(tmp_path)


def _transcribe_faster_whisper(audio_bytes: bytes, filename: str, model_size: str) -> dict:
    from faster_whisper import WhisperModel
    suffix = os.path.splitext(filename)[-1] or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name
    try:
        model = WhisperModel(model_size, device="cpu", compute_type="int8")
        segments, info = model.transcribe(tmp_path, beam_size=5, language="en")
        text = " ".join(seg.text.strip() for seg in segments)
        return {
            "text":     text.lower(),
            "language": info.language,
            "duration": round(info.duration, 2),
        }
    finally:
        os.unlink(tmp_path)


class TranscriptionService:
    def transcribe(
        self,
        audio_bytes: bytes,
        filename: str = "audio",
        provider: str = "openai",
        model_size: str = "base",
    ) -> dict:
        if provider == "faster-whisper":
            result = _transcribe_faster_whisper(audio_bytes, filename, model_size)
        else:
            result = _transcribe_openai(audio_bytes, filename)

        result["audio"] = None
        if result.get("text"):
            try:
                audio_bytes, media_type = SpeechService().synthesize(result["text"])
                encoded = base64.b64encode(audio_bytes).decode("ascii")
                result["audio"] = f"data:{media_type};base64,{encoded}"
            except Exception:
                pass  # Text transcript still stands even if speech synthesis fails.

        return result
