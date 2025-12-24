"""
Application configuration using Pydantic Settings.

Automatically loads environment variables from .env files.
"""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # OpenAI
    openai_api_key: str | None = None

    # Database (required - must be set in .env or environment)
    database_url: str

    # Debug flags
    sql_debug: bool = False
    debug: bool = False

    model_config = SettingsConfigDict(
        # Load from .env file in the backend directory
        env_file=Path(__file__).parent / ".env",
        env_file_encoding="utf-8",
        # Also check parent directories for .env
        extra="ignore",
        # Case insensitive environment variable names
        case_sensitive=False,
    )


@lru_cache
def get_settings() -> Settings:
    """
    Get cached application settings.

    Returns:
        Settings: Application configuration loaded from environment.
    """
    return Settings()

