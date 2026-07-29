from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    APP_NAME:                    str = "FastAPI App"
    DATABASE_URL:                str
    SECRET_KEY:                  str
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    OPENAI_API_KEY:              str = ""
    ANTHROPIC_API_KEY:           str = ""
    GOOGLE_API_KEY:              str = ""
    QDRANT_URL:                  str = "http://localhost:6333"
    QDRANT_API_KEY:              str = ""
    OLLAMA_URL:                  str = "http://localhost:11434"
    OLLAMA_API_KEY:              str = ""
    OLLAMA_EMBED_MODEL:          str = "nomic-embed-text"
    OLLAMA_LLM_MODEL:            str = "llama3.2"
    OLLAMA_NUM_CTX:              int = 8192
    OLLAMA_KEEP_ALIVE:           str = "30m"
    PIPER_VOICE_MODEL:           str = "app/assets/voices/en_US-lessac-medium.onnx"

    class Config:
        env_file = ".env"

settings = Settings()