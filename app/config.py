"""Application configuration."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings."""
    
    # Hugging Face Space OpenAI API endpoint
    hf_space_url: str = "https://jeanbaptdzd-open-finance-llm-8b.hf.space"
    
    # OpenAI-compatible API settings
    api_key: str = "not-needed"  # No authentication required
    model_name: str = "DragonLLM/Qwen-Open-Finance-R-8B"
    
    # API configuration
    timeout: float = 120.0
    max_retries: int = 3
    
    # Logfire configuration
    environment: str = "development"  # development, staging, production
    
    # Generation settings for reasoning models
    # Qwen3 uses <think> tags which consume 40-60% of tokens
    # Increase max_tokens to allow complete responses
    max_tokens: int = 1500  # Increased for reasoning models (was default ~800-1000)
    
    # Context window limits for Qwen-3 8B
    # Base context window: 32,768 tokens (32K)
    # Extended with YaRN: up to 128,000 tokens (128K)
    # Current max_tokens is for generation, context input can use up to ~30K tokens
    
    # Generation limits
    # Maximum theoretical generation: 20,000 tokens
    # Practical limit depends on: context_window - input_tokens - safety_margin
    # With typical input (~500 tokens), can generate up to ~30K tokens
    max_generation_limit: int = 20000  # Theoretical maximum (rarely needed)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()

