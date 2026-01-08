"""Prompt management system using Langfuse.

This module provides functions to fetch and cache prompts from Langfuse Prompt Management.
"""

import logging
from typing import Optional, Dict
from functools import lru_cache

from app.langfuse_config import get_langfuse_client

logger = logging.getLogger(__name__)

# Local cache for prompts (fallback if Langfuse unavailable)
_PROMPT_CACHE: Dict[str, str] = {}


def get_prompt(prompt_name: str, version: Optional[int] = None) -> Optional[str]:
    """
    Get a prompt from Langfuse Prompt Management.
    
    Args:
        prompt_name: Name of the prompt (e.g., "finance_agent_system", "agent_1_system")
        version: Optional version number. If None, returns latest version.
        
    Returns:
        Prompt text if found, None otherwise
    """
    langfuse = get_langfuse_client()
    
    # Check cache first
    cache_key = f"{prompt_name}_v{version}" if version else prompt_name
    if cache_key in _PROMPT_CACHE:
        return _PROMPT_CACHE[cache_key]
    
    # Try to fetch from Langfuse
    if langfuse:
        try:
            # Note: Actual Langfuse API may differ - this is a placeholder
            # prompt = langfuse.get_prompt(name=prompt_name, version=version)
            # For now, return None to use fallback
            logger.debug(f"Langfuse prompt fetching not yet implemented for {prompt_name}")
        except Exception as e:
            logger.debug(f"Failed to fetch prompt from Langfuse: {e}")
    
    # Fallback: return None (caller should use hardcoded prompt)
    return None


@lru_cache(maxsize=32)
def get_cached_prompt(prompt_name: str, version: Optional[int] = None) -> Optional[str]:
    """
    Get a prompt with LRU caching.
    
    Same as get_prompt but with additional caching layer.
    """
    return get_prompt(prompt_name, version)


def set_prompt_cache(prompt_name: str, prompt_text: str, version: Optional[int] = None):
    """
    Manually set a prompt in the cache (useful for fallback prompts).
    
    Args:
        prompt_name: Name of the prompt
        prompt_text: Prompt text
        version: Optional version number
    """
    cache_key = f"{prompt_name}_v{version}" if version else prompt_name
    _PROMPT_CACHE[cache_key] = prompt_text
    logger.debug(f"Cached prompt: {cache_key}")


def clear_prompt_cache():
    """Clear the prompt cache."""
    global _PROMPT_CACHE
    _PROMPT_CACHE.clear()
    get_cached_prompt.cache_clear()
    logger.debug("Prompt cache cleared")


