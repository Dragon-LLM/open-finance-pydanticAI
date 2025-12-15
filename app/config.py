"""Application configuration."""

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
import httpx


# Available endpoints
ENDPOINTS = {
    "koyeb": {
        "url": "https://dragon-llm-dealexmachina-673cae4f.koyeb.app",
        "model": "DragonLLM/Qwen-Open-Finance-R-8B",  # vLLM requires exact model name
    },
    "hf": {
        "url": "https://jeanbaptdzd-open-finance-llm-8b.hf.space",
        "model": "dragon-llm-open-finance",  # HF Space accepts any name
    },
    "llm_pro_finance": {
        "url": "https://demo.llmprofinance.com",
        "model": "DragonLLM/llama3.1-70b-fin-v1.0-fp8",  # Fine-tuned Llama 70B for finance
        "api_path": "/api",  # LLM Pro Finance uses /api instead of /api/v1
    },
    "ollama": {
        "url": "http://localhost:11434",
        "model": "",  # Will use OLLAMA_MODEL environment variable
        "api_path": "/v1",  # Ollama's OpenAI-compatible endpoint
    },
}


def strip_quotes(value: str) -> str:
    """Strip surrounding quotes from a value."""
    if value and len(value) >= 2:
        if (value.startswith("'") and value.endswith("'")) or \
           (value.startswith('"') and value.endswith('"')):
            return value[1:-1]
    return value


class Settings(BaseSettings):
    """Application settings."""
    
    # Endpoint selection: "koyeb", "hf", "llm_pro_finance", or "ollama"
    endpoint: str = "koyeb"
    
    @property
    def base_url(self) -> str:
        """Get the base URL for the selected endpoint."""
        return ENDPOINTS.get(self.endpoint, ENDPOINTS["koyeb"])["url"]
    
    @property
    def model_name(self) -> str:
        """Get the model name for the selected endpoint."""
        if self.endpoint == "ollama":
            return self.ollama_model if self.ollama_model else ""
        return ENDPOINTS.get(self.endpoint, ENDPOINTS["koyeb"])["model"]
    
    # Legacy alias for compatibility
    @property
    def hf_space_url(self) -> str:
        return self.base_url
    
    # OpenAI-compatible API settings
    api_key: str = "not-needed"
    
    # LLM Pro Finance API key for Llama 70B model
    llm_pro_finance_key: str = ""
    
    # LLM Pro Finance API URL (optional, defaults to ENDPOINTS config)
    llm_pro_finance_url: str = ""
    
    # Ollama local model configuration
    ollama_model: str = "qwen2.5:3b-instruct"  # Model name to use (e.g., "dragon-llm", "qwen2.5:7b", "ministral-3:14b-instruct-2512-q4_K_M")
    
    # Validators to strip quotes from env values
    @field_validator('llm_pro_finance_key', 'llm_pro_finance_url', 'api_key', 'ollama_model', mode='before')
    @classmethod
    def strip_quotes_from_value(cls, v):
        if isinstance(v, str):
            return strip_quotes(v)
        return v
    
    @property
    def judge_api_key(self) -> str:
        """Get API key for judge agent (uses LLM_PRO_FINANCE_KEY if available)."""
        if self.llm_pro_finance_key:
            return self.llm_pro_finance_key
        return self.api_key
    
    @property
    def judge_base_url(self) -> str:
        """Get base URL for judge agent (uses LLM_PRO_FINANCE_URL if available)."""
        if self.llm_pro_finance_url:
            return self.llm_pro_finance_url
        return ENDPOINTS.get("llm_pro_finance", {}).get("url", "https://api.llm-pro-finance.com")
    
    # API configuration
    timeout: float = 120.0
    max_retries: int = 3
    
    # Logfire configuration
    environment: str = "development"  # development, staging, production
    
    # Generation settings for reasoning models
    max_tokens: int = 1500
    
    # Generation limits
    max_generation_limit: int = 20000

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()


def get_best_available_endpoint(timeout: float = 3.0) -> str:
    """Get the best available endpoint, preferring Koyeb over HF.
    
    Returns:
        "koyeb" if Koyeb is available (online or can be woken up)
        "hf" if only HuggingFace is available
        "koyeb" as default if neither is available (will fail gracefully)
    """
    koyeb_url = ENDPOINTS.get("koyeb", {}).get("url", "")
    hf_url = ENDPOINTS.get("hf", {}).get("url", "")
    
    # Check Koyeb first (preferred)
    if koyeb_url:
        try:
            with httpx.Client(timeout=timeout, follow_redirects=True) as client:
                # Try multiple endpoints
                for url in [f"{koyeb_url}/v1/models", koyeb_url]:
                    try:
                        r = client.get(url)
                        if r.status_code in [200, 401]:
                            return "koyeb"
                        # 404 with "no active service" means sleeping, but we can try to wake it
                        if r.status_code == 404 and "no active service" in r.text.lower():
                            # Still prefer Koyeb even if sleeping (can be woken up)
                            return "koyeb"
                    except httpx.TimeoutException:
                        # Timeout might mean sleeping, but still prefer Koyeb
                        continue
                    except:
                        continue
        except:
            pass
    
    # Check HF as fallback
    if hf_url:
        try:
            with httpx.Client(timeout=timeout, follow_redirects=True) as client:
                r = client.get(hf_url)
                if r.status_code in [200, 401]:
                    return "hf"
        except:
            pass
    
    # Default to koyeb (preferred, even if not currently available)
    return "koyeb"
