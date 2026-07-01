from fastapi import APIRouter
from app.controllers.speech_controller import SpeechController

router = APIRouter(prefix="/speech", tags=["Speech"])

router.post("")(SpeechController.synthesize)
