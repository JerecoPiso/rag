import io
import re
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


# Strips markdown syntax down to plain spoken text — otherwise TTS engines read
# out literal symbols ("asterisk asterisk", "pound", "pipe") instead of the words.
def _strip_markdown(text: str) -> str:
    # Literal escape sequences (backslash + letter, as opposed to an actual newline
    # char) show up when text passed through a JSON round trip without decoding —
    # normalize them to real whitespace before anything else, or the markdown
    # regexes below (which anchor on real \n) won't see line boundaries, and the
    # TTS engine ends up reading "backslash n" out loud.
    text = text.replace("\\r\\n", "\n").replace("\\n", "\n").replace("\\t", " ").replace("\\r", "\n")
    text = re.sub(r"```[a-zA-Z0-9]*\n?(.*?)```", r"\1", text, flags=re.DOTALL)   # fenced code blocks
    text = re.sub(r"`([^`]+)`", r"\1", text)                                     # inline code
    text = re.sub(r"!\[([^\]]*)\]\([^)]*\)", r"\1", text)                        # images
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)                         # links
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.MULTILINE)                   # headers
    text = re.sub(r"(\*\*\*|___)(.+?)\1", r"\2", text)                           # bold+italic
    text = re.sub(r"(\*\*|__)(.+?)\1", r"\2", text)                              # bold
    text = re.sub(r"(?<!\w)(\*|_)(.+?)\1(?!\w)", r"\2", text)                    # italic
    text = re.sub(r"~~(.+?)~~", r"\1", text)                                     # strikethrough
    text = re.sub(r"^>\s?", "", text, flags=re.MULTILINE)                        # blockquotes
    text = re.sub(r"^[ \t]*[-*+]\s+", "", text, flags=re.MULTILINE)              # bullet list markers
    text = re.sub(r"^\s*([-*_])\1{2,}\s*$", "", text, flags=re.MULTILINE)        # horizontal rules
    text = re.sub(r"^\s*\|?[-:| ]+\|[-:| ]*\s*$", "", text, flags=re.MULTILINE)  # table separator rows
    text = text.replace("|", " ")                                                # remaining table pipes
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


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

        text = _strip_markdown(text)

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
