"""PydanticAI model configuration."""

from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

from app.config import settings

# Create PydanticAI model using OpenAI-compatible endpoint from Hugging Face Space
# The model name will be sent in the request, but the actual model is determined by the HF Space
# Note: max_tokens will be set at the Agent level, not here
finance_model = OpenAIModel(
    model_name="dragon-llm-open-finance",  # Placeholder name sent to the HF Space backend
    provider=OpenAIProvider(
        base_url=f"{settings.hf_space_url}/v1",
        api_key=settings.api_key,
    ),
)

