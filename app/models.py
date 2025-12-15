"""PydanticAI model configuration."""

from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from app.config import settings, ENDPOINTS, get_best_available_endpoint

# Create PydanticAI model using OpenAI-compatible endpoint
# Supports both Koyeb (vLLM) and HF Space (Transformers) backends
# Automatically prefers Koyeb when available, falls back to HF
def get_finance_model():
    """Get finance model using the best available endpoint (prefers Koyeb)."""
    # Get the best available endpoint (prefers Koyeb)
    endpoint = get_best_available_endpoint()
    endpoint_config = ENDPOINTS.get(endpoint, ENDPOINTS["koyeb"])
    
    return OpenAIChatModel(
        model_name=endpoint_config["model"],
        provider=OpenAIProvider(
            base_url=f"{endpoint_config['url']}/v1",
            api_key=settings.api_key,
        ),
    )

# Create a function that returns the finance model using the best available endpoint
# This ensures we always prefer Koyeb when available
def get_finance_model_dynamic():
    """Get finance model using the best available endpoint (prefers Koyeb)."""
    return get_finance_model()

# For backward compatibility, create a default instance
# But agents should ideally call get_finance_model_dynamic() for fresh models
finance_model = get_finance_model()

# Judge agent model using Llama 70B (LLM Pro Finance)
# Uses LLM_PRO_FINANCE_KEY from .env if available
# Falls back to finance_model if key not available
def get_judge_model():
    """Get judge model - uses LLM Pro Finance if key available, otherwise uses finance_model."""
    if settings.llm_pro_finance_key:
        # Use Llama 70B via LLM Pro Finance
        base_url = settings.judge_base_url
        # LLM Pro Finance uses /api (not /api/v1)
        api_path = ENDPOINTS.get("llm_pro_finance", {}).get("api_path", "/api")
        
        return OpenAIChatModel(
            model_name=ENDPOINTS.get("llm_pro_finance", {}).get("model", "DragonLLM/llama3.1-70b-fin-v1.0-fp8"),
            provider=OpenAIProvider(
                base_url=f"{base_url}{api_path}",
                api_key=settings.llm_pro_finance_key if settings.llm_pro_finance_key else "not-needed",
            ),
        )
    else:
        # Fallback to finance_model if no key available
        return finance_model

judge_model = get_judge_model()


def get_model_for_endpoint(endpoint: str):
    """Create a model instance for a specific endpoint.
    
    Args:
        endpoint: One of "koyeb", "hf", "llm_pro_finance", or "ollama"
        
    Returns:
        OpenAIChatModel configured for the specified endpoint
        
    Raises:
        ValueError: If endpoint is not recognized or Ollama model is not configured
    """
    if endpoint == "koyeb":
        endpoint_config = ENDPOINTS.get("koyeb", {})
        return OpenAIChatModel(
            model_name=endpoint_config.get("model", "DragonLLM/Qwen-Open-Finance-R-8B"),
            provider=OpenAIProvider(
                base_url=f"{endpoint_config.get('url', '')}/v1",
                api_key=settings.api_key,
            ),
        )
    elif endpoint == "hf":
        endpoint_config = ENDPOINTS.get("hf", {})
        return OpenAIChatModel(
            model_name=endpoint_config.get("model", "dragon-llm-open-finance"),
            provider=OpenAIProvider(
                base_url=f"{endpoint_config.get('url', '')}/v1",
                api_key=settings.api_key,
            ),
        )
    elif endpoint == "llm_pro_finance":
        endpoint_config = ENDPOINTS.get("llm_pro_finance", {})
        base_url = settings.llm_pro_finance_url or endpoint_config.get("url", "")
        api_path = endpoint_config.get("api_path", "/api")
        api_key = settings.llm_pro_finance_key if settings.llm_pro_finance_key else "not-needed"
        
        return OpenAIChatModel(
            model_name=endpoint_config.get("model", "DragonLLM/llama3.1-70b-fin-v1.0-fp8"),
            provider=OpenAIProvider(
                base_url=f"{base_url}{api_path}",
                api_key=api_key,
            ),
        )
    elif endpoint == "ollama":
        endpoint_config = ENDPOINTS.get("ollama", {})
        if not settings.ollama_model:
            raise ValueError(
                "OLLAMA_MODEL environment variable must be set when using Ollama endpoint. "
                "Example: OLLAMA_MODEL=dragon-llm or OLLAMA_MODEL=qwen2.5:7b"
            )
        base_url = endpoint_config.get("url", "http://localhost:11434")
        api_path = endpoint_config.get("api_path", "/v1")
        
        return OpenAIChatModel(
            model_name=settings.ollama_model,
            provider=OpenAIProvider(
                base_url=f"{base_url}{api_path}",
                api_key="ollama",  # Required but ignored by Ollama
            ),
        )
    else:
        raise ValueError(f"Unknown endpoint: {endpoint}. Must be one of: 'koyeb', 'hf', 'llm_pro_finance', 'ollama'")

