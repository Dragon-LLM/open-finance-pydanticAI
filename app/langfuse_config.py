"""Langfuse configuration for open-finance-pydanticAI project."""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Flag to avoid multiple configurations
_langfuse_configured = False
_langfuse_client: Optional[object] = None


def configure_langfuse() -> Optional[object]:
    """
    Configure Langfuse for the open-finance-pydanticAI project.
    
    Returns:
        Langfuse client instance if configured, None otherwise.
        
    The client is only initialized if:
    - ENABLE_LANGFUSE is True (or not set, defaults to True if keys present)
    - LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY are provided
    """
    global _langfuse_configured, _langfuse_client
    
    if _langfuse_configured:
        return _langfuse_client
    
    try:
        from app.config import settings
        
        # Check if Langfuse is enabled
        enable_langfuse = getattr(settings, 'enable_langfuse', True)
        if not enable_langfuse:
            logger.info("Langfuse is disabled via configuration")
            _langfuse_configured = True
            return None
        
        # Get credentials from settings
        public_key = getattr(settings, 'langfuse_public_key', None)
        secret_key = getattr(settings, 'langfuse_secret_key', None)
        # Support both langfuse_host and langfuse_base_url
        host = getattr(settings, 'langfuse_host_resolved', None) or getattr(settings, 'langfuse_host', 'https://cloud.langfuse.com')
        
        # Only initialize if keys are present
        if not public_key or not secret_key:
            logger.info("Langfuse credentials not provided, skipping initialization")
            _langfuse_configured = True
            return None
        
        from langfuse import Langfuse
        
        _langfuse_client = Langfuse(
            public_key=public_key,
            secret_key=secret_key,
            host=host,
        )
        
        logger.info(f"Langfuse configured successfully (host: {host})")
        _langfuse_configured = True
        return _langfuse_client
        
    except ImportError:
        logger.warning("Langfuse package not installed, skipping initialization")
        _langfuse_configured = True
        return None
    except Exception as e:
        logger.warning(f"Failed to configure Langfuse: {e}. Continuing without Langfuse.")
        _langfuse_configured = True
        return None


def get_langfuse_client() -> Optional[object]:
    """Get the configured Langfuse client instance."""
    if not _langfuse_configured:
        configure_langfuse()
    return _langfuse_client

