from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Service configuration, read from environment variables or `.env`.
    """

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    ollama_host: str = "http://localhost:11434"

    model_name: str = "qwen3:4b-instruct-2507-q4_K_M"

    num_ctx: int = 8192

    temperature: float = 0.0

    max_doc_chars: int = 16000
    
    keep_alive: str = "30m"
    
    request_timeout_s: float = 600.0
    
    test_set_dir: Path = Path("data/test_set")


@lru_cache
def get_settings() -> Settings:
    """Return the cached Settings instance."""
    return Settings()
