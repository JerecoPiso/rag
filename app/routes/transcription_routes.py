from fastapi import APIRouter
from app.controllers.transcription_controller import TranscriptionController
from app.schemas.transcription import TranscriptionResponse

router = APIRouter(prefix="/transcribe", tags=["Transcription"])

router.post("", response_model=TranscriptionResponse)(TranscriptionController.transcribe)
