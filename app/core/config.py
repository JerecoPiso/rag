from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    APP_NAME:                    str = "FastAPI App"
    DATABASE_URL:                str
    SECRET_KEY:                  str
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    class Config:
        env_file = ".env"

settings = Settings()