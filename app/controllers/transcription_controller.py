from fastapi import UploadFile, File, Query
from app.services.transcription_service import TranscriptionService


class TranscriptionController:
    @staticmethod
    async def transcribe(
        file: UploadFile = File(...),
        provider: str = Query("openai", description="Transcription provider: openai, faster-whisper"),
        model_size: str = Query("base", description="Model size for faster-whisper: tiny, base, small, medium, large"),
    ):
        audio_bytes = await file.read()
        svc = TranscriptionService()
        return svc.transcribe(audio_bytes, filename=file.filename or "audio", provider=provider, model_size=model_size)
