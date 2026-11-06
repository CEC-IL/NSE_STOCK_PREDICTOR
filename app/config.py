from pydantic_settings import BaseSettings
from typing import List

class Settings(BaseSettings):
    PROJECT_NAME: str = "NSE Stock Predictor API"
    API_V1_STR: str = "/api/v1"
    DATABASE_URL: str = "sqlite:///.nse_stocks.db"
    CORS_ORIGINS: List[str] = ["http://localhost:3000"]

    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()
