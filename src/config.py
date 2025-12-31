"""
Configuration management for the Hospital Treatment Duration Prediction application.

This module handles loading and validation of configuration from environment variables.
"""

import os
from typing import Literal, Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, field_validator


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    Uses .env file for local development.
    """

    # LLM API Keys
    openai_api_key: Optional[str] = Field(
        None,
        description="OpenAI API key for GPT models"
    )
    google_api_key: Optional[str] = Field(
        None,
        description="Google API key for Gemini models"
    )

    # Model Configuration
    model_path: str = Field(
        "best_dt_classifier_model.joblib",
        description="Path to the trained model file"
    )

    # LLM Provider Configuration
    primary_llm_provider: Literal["openai", "gemini"] = Field(
        "openai",
        description="Primary LLM provider to use"
    )
    fallback_llm_provider: Optional[Literal["openai", "gemini"]] = Field(
        "gemini",
        description="Fallback LLM provider if primary fails"
    )

    # OpenAI Configuration
    openai_model: str = Field(
        "gpt-4o",
        description="OpenAI model name"
    )
    openai_temperature: float = Field(
        0.0,
        ge=0.0,
        le=2.0,
        description="OpenAI sampling temperature"
    )

    # Gemini Configuration
    gemini_model: str = Field(
        "gemini-2.0-flash-exp",
        description="Gemini model name"
    )
    gemini_temperature: float = Field(
        0.0,
        ge=0.0,
        le=2.0,
        description="Gemini sampling temperature"
    )

    # Cache Configuration
    redis_url: Optional[str] = Field(
        None,
        description="Redis URL for caching (e.g., redis://localhost:6379/0)"
    )
    cache_ttl_hours: int = Field(
        24,
        ge=1,
        description="Cache TTL in hours"
    )
    enable_cache: bool = Field(
        True,
        description="Whether to enable caching"
    )

    # Database Configuration
    database_url: Optional[str] = Field(
        None,
        description="Database URL for storing predictions"
    )

    # API Configuration
    api_host: str = Field(
        "0.0.0.0",
        description="API host to bind to"
    )
    api_port: int = Field(
        8000,
        ge=1,
        le=65535,
        description="API port to bind to"
    )
    api_workers: int = Field(
        4,
        ge=1,
        description="Number of API workers"
    )

    # Logging Configuration
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        "INFO",
        description="Logging level"
    )
    log_format: Literal["json", "text"] = Field(
        "json",
        description="Log format (json for production, text for development)"
    )

    # Feature flags
    enable_validation: bool = Field(
        True,
        description="Whether to enable extraction validation"
    )
    enable_metrics: bool = Field(
        True,
        description="Whether to collect metrics"
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )

    @field_validator("primary_llm_provider")
    @classmethod
    def validate_primary_provider_key(cls, v: str, info) -> str:
        """Validate that API key exists for primary provider (warning only)"""
        import warnings
        # Note: info.data contains the values parsed so far
        if v == "openai" and not os.getenv("OPENAI_API_KEY"):
            warnings.warn(
                "OPENAI_API_KEY not found. LLM extraction will not work until API key is configured."
            )
        if v == "gemini" and not os.getenv("GOOGLE_API_KEY"):
            warnings.warn(
                "GOOGLE_API_KEY not found. LLM extraction will not work until API key is configured."
            )
        return v

    @field_validator("model_path")
    @classmethod
    def validate_model_path(cls, v: str) -> str:
        """Validate that model file exists (warning only)"""
        if not os.path.exists(v):
            import warnings
            warnings.warn(f"Model file not found at path: {v}. Model loading will fail until file is available.")
        return v

    @property
    def cache_ttl_seconds(self) -> int:
        """Get cache TTL in seconds"""
        return self.cache_ttl_hours * 3600


# Global settings instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """
    Get application settings singleton.

    Returns:
        Settings instance

    Raises:
        ValueError: If settings validation fails
    """
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reload_settings():
    """Reload settings from environment (useful for testing)"""
    global _settings
    _settings = Settings()
    return _settings
