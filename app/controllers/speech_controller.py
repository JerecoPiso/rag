from fastapi import Response
from app.services.speech_service import SpeechService
from app.schemas.speech import SpeechRequest


class SpeechController:
    @staticmethod
    async def synthesize(body: SpeechRequest):
        svc = SpeechService()
        audio_bytes, media_type = svc.synthesize(
            text=body.text,
            voice=body.voice,
            model=body.model,
            audio_format=body.format,
        )
        return Response(content=audio_bytes, media_type=media_type)
