"""Configuration Logfire pour le projet open-finance dans l'organisation deal-ex-machina (UE)."""

import logfire
from app.config import settings

# Flag pour éviter la configuration multiple
_logfire_configured = False


def configure_logfire(send_to_logfire: bool | str | None = 'if-token-present'):
    """
    Configure Logfire pour le projet open-finance dans l'organisation deal-ex-machina.
    
    Le projet est configuré pour la région UE pour la conformité RGPD.
    
    Args:
        send_to_logfire: Si 'if-token-present', n'envoie que si un token est présent.
                        Si False, désactive complètement l'envoi.
                        Si True, force l'envoi (nécessite authentification).
    """
    global _logfire_configured
    
    if _logfire_configured:
        return
    
    logfire.configure(
        service_name="open-finance-pydanticai",
        service_version="0.1.0",
        environment=getattr(settings, 'environment', 'development'),
        send_to_logfire=send_to_logfire,
        # Le token est automatiquement récupéré depuis:
        # - Variable d'environnement LOGFIRE_TOKEN
        # - Ou via logfire auth (stocké dans .logfire/)
        # Le projet et l'organisation sont déterminés par le token
        # Pour le projet "open-finance" dans "deal-ex-machina", exécutez:
        # logfire auth
    )
    
    _logfire_configured = True

