"""PydanticAI model configuration."""

from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from app.config import settings, ENDPOINTS

# Create PydanticAI model using OpenAI-compatible endpoint
# Supports both Koyeb (vLLM) and HF Space (Transformers) backends
# Configure via ENDPOINT env var: "koyeb" (default), "hf", or "llm_pro_finance"
finance_model = OpenAIChatModel(
    model_name=settings.model_name,
    provider=OpenAIProvider(
        base_url=f"{settings.base_url}/v1",
        api_key=settings.api_key,
    ),
)

# Judge agent model using Llama 70B (LLM Pro Finance)
# Uses LLM_PRO_FINANCE_KEY from .env if available
# Falls back to finance_model if key not available
def get_judge_model():
    """Get judge model - uses LLM Pro Finance if key available, otherwise uses finance_model."""
    if settings.llm_pro_finance_key:
        # Use Llama 70B via LLM Pro Finance
        return OpenAIChatModel(
            model_name=ENDPOINTS.get("llm_pro_finance", {}).get("model", "llama-70b-finance"),
            provider=OpenAIProvider(
                base_url=f"{settings.judge_base_url}/v1",
                api_key=settings.llm_pro_finance_key,
            ),
        )
    else:
        # Fallback to finance_model if no key available
        return finance_model

judge_model = get_judge_model()

