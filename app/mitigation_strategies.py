"""
Stratégies de mitigation pour les échecs avec modèles fine-tunés (Qwen 8B).

Ce module fournit des mécanismes pour:
1. Détecter et éviter les échecs de tool calls
2. Valider les formats JSON
3. Valider la sémantique des sorties JSON
4. Implémenter des stratégies de retry intelligentes
5. Fournir des wrappers de validation pour les agents
"""

import copy
import json
import logging
import re
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple, Type, TypeVar
from pydantic import BaseModel, ValidationError
from pydantic_ai import Agent
import asyncio
from functools import wraps

logger = logging.getLogger(__name__)

# Type alias for agent run result (pydantic_ai returns different types)
RunResult = Any  # Will be the actual return type from agent.run()

T = TypeVar('T', bound=BaseModel)


class RunResultProtocol(Protocol):
    """Protocol defining the expected structure of agent run results."""
    def all_messages(self) -> List[Any]: ...
    output: str
    data: Any


# ============================================================================
# UTILITAIRES
# ============================================================================

class EmptyResult:
    """Résultat vide utilisé comme fallback en cas d'échec."""
    
    def __init__(self):
        self.output = ""
        self.data = None
    
    def all_messages(self):
        return []


# ============================================================================
# DÉTECTION DES TOOL CALLS
# ============================================================================

class ToolCallDetector:
    """Détecte et valide les appels d'outils dans les résultats d'agents."""
    
    @staticmethod
    def extract_tool_calls(result: RunResult, include_final_result: bool = False) -> List[Dict[str, Any]]:
        """Extrait tous les tool calls d'un résultat d'agent.
        
        PydanticAI uses message.parts with ToolCallPart/ToolReturnPart objects.
        The 'final_result' tool is PydanticAI's internal mechanism for structured output.
        
        Args:
            result: Agent run result
            include_final_result: If False, excludes 'final_result' (structured output mechanism)
        """
        tool_calls = []
        seen_ids = set()  # Deduplicate by tool_call_id
        
        try:
            for msg in result.all_messages():
                # PydanticAI style - check msg.parts for ToolCallPart
                if hasattr(msg, 'parts'):
                    for part in msg.parts:
                        # Only count ToolCallPart, NOT ToolReturnPart
                        part_type = type(part).__name__
                        if part_type == 'ToolCallPart' and hasattr(part, 'tool_name'):
                            # Skip final_result unless explicitly requested
                            if not include_final_result and part.tool_name == 'final_result':
                                continue
                            
                            # Deduplicate by tool_call_id
                            call_id = getattr(part, 'tool_call_id', None)
                            if call_id and call_id in seen_ids:
                                continue
                            if call_id:
                                seen_ids.add(call_id)
                            
                            tool_info = {
                                "name": part.tool_name,
                                "args": getattr(part, 'args', {}) or {},
                                "tool_call_id": call_id,
                                "raw": part,
                            }
                            tool_calls.append(tool_info)
        except Exception as e:
            logger.debug(f"Error extracting tool calls: {e}")
        
        return tool_calls

    @staticmethod
    def _extract_tool_info(call) -> Optional[Dict[str, Any]]:
        """Extrait les informations d'un appel d'outil."""
        name = None
        args = None

        # Try different ways to access tool call structure
        if hasattr(call, "function"):
            func = call.function
            name = getattr(func, "name", None)
            args = getattr(func, "arguments", None)
        elif hasattr(call, "tool_name"):
            name = call.tool_name
            args = getattr(call, "args", None)
        elif hasattr(call, "name"):
            name = call.name
            args = getattr(call, "args", None)
        elif isinstance(call, dict):
            name = call.get("tool_name") or call.get("name") or call.get("function", {}).get("name")
            args = call.get("args") or call.get("arguments")

        if name:
            parsed_args = args
            if isinstance(args, str):
                try:
                    parsed_args = json.loads(args)
                except json.JSONDecodeError:
                    parsed_args = {}
            elif not isinstance(args, dict):
                parsed_args = {}

            return {
                "name": name,
                "args": parsed_args,
                "raw": call
            }
        return None
    
    @staticmethod
    def validate_tool_calls_required(
        result: RunResult,
        expected_tools: Optional[List[str]] = None,
        min_calls: int = 1
    ) -> Tuple[bool, List[str]]:
        """Valide que les tool calls requis ont été effectués.
        
        Args:
            result: Résultat de l'agent
            expected_tools: Liste des noms d'outils attendus (None = au moins un outil)
            min_calls: Nombre minimum d'appels requis
        
        Returns:
            Tuple (is_valid, errors)
        """
        tool_calls = ToolCallDetector.extract_tool_calls(result)
        errors = []
        
        # Vérifier le nombre minimum
        if len(tool_calls) < min_calls:
            errors.append(
                f"Nombre insuffisant de tool calls: {len(tool_calls)} < {min_calls}"
            )
        
        # Vérifier les outils attendus
        if expected_tools:
            called_tools = {tc["name"] for tc in tool_calls}
            missing_tools = set(expected_tools) - called_tools
            if missing_tools:
                errors.append(
                    f"Outils manquants: {', '.join(missing_tools)}"
                )
        
        return len(errors) == 0, errors


# ============================================================================
# VALIDATION JSON
# ============================================================================

class JSONValidator:
    """Valide les formats et la sémantique des sorties JSON."""
    
    @staticmethod
    def extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
        """Extrait un objet JSON d'un texte (peut contenir du texte autour)."""
        # Essayer de parser directement
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        
        # Chercher un bloc JSON dans le texte
        # Pattern 1: JSON entre accolades {}
        json_patterns = [
            r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}',  # JSON simple
            r'```json\s*(\{.*?\})\s*```',  # JSON dans code block
            r'```\s*(\{.*?\})\s*```',  # JSON dans code block sans json
        ]
        
        for pattern in json_patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            for match in matches:
                try:
                    return json.loads(match)
                except json.JSONDecodeError:
                    continue
        
        return None
    
    @staticmethod
    def validate_json_structure(
        json_data: Dict[str, Any],
        expected_model: Type[BaseModel]
    ) -> Tuple[bool, Optional[ValidationError], Optional[BaseModel]]:
        """Valide la structure JSON contre un modèle Pydantic.
        
        Returns:
            Tuple (is_valid, validation_error, validated_model)
        """
        try:
            validated = expected_model.model_validate(json_data)
            return True, None, validated
        except ValidationError as e:
            return False, e, None
    
    @staticmethod
    def validate_json_semantics(
        json_data: Dict[str, Any],
        expected_model: Type[BaseModel],
        semantic_checks: Optional[List[Callable[[Dict[str, Any]], Tuple[bool, str]]]] = None
    ) -> Tuple[bool, List[str]]:
        """Valide la sémantique des données JSON en utilisant les métadonnées du modèle.
        
        Args:
            json_data: Données JSON à valider
            expected_model: Modèle Pydantic attendu
            semantic_checks: Liste de fonctions de validation sémantique
        
        Returns:
            Tuple (is_valid, errors)
        """
        errors = []
        
        # Vérifications de base utilisant les métadonnées du modèle
        if isinstance(json_data, dict):
            try:
                # Obtenir le schéma JSON du modèle pour extraire les contraintes
                model_schema = expected_model.model_json_schema()
                properties_schema = model_schema.get('properties', {})
            except (AttributeError, TypeError):
                properties_schema = {}
            
            # Vérifier les contraintes numériques basées sur le schéma du modèle
            for key, value in json_data.items():
                if isinstance(value, (int, float)):
                    field_schema = properties_schema.get(key, {})
                    
                    # Vérifier les contraintes minimum (ge)
                    if 'minimum' in field_schema:
                        if value < field_schema['minimum']:
                            errors.append(
                                f"Valeur négative invalide pour {key}: {value} "
                                f"(minimum attendu: {field_schema['minimum']})"
                            )
                    
                    # Vérifier les contraintes maximum (le)
                    if 'maximum' in field_schema:
                        if value > field_schema['maximum']:
                            errors.append(
                                f"Valeur hors limites pour {key}: {value} "
                                f"(maximum attendu: {field_schema['maximum']})"
                            )
                    
                    # Vérifier exclusiveMinimum (gt)
                    if 'exclusiveMinimum' in field_schema:
                        if value <= field_schema['exclusiveMinimum']:
                            errors.append(
                                f"Valeur invalide pour {key}: {value} "
                                f"(doit être > {field_schema['exclusiveMinimum']})"
                            )
                    
                    # Vérifier exclusiveMaximum (lt)
                    if 'exclusiveMaximum' in field_schema:
                        if value >= field_schema['exclusiveMaximum']:
                            errors.append(
                                f"Valeur invalide pour {key}: {value} "
                                f"(doit être < {field_schema['exclusiveMaximum']})"
                            )
        
        # Vérifications sémantiques personnalisées
        if semantic_checks:
            for check in semantic_checks:
                try:
                    is_valid, error_msg = check(json_data)
                    if not is_valid:
                        errors.append(error_msg)
                except Exception as e:
                    logger.warning("Error in semantic check: %s", e, exc_info=True)
                    errors.append(f"Erreur lors de la validation sémantique: {str(e)}")
        
        return len(errors) == 0, errors


# ============================================================================
# STRATÉGIES DE RETRY
# ============================================================================

class RetryStrategy:
    """Stratégies de retry pour les agents."""
    
    @staticmethod
    async def retry_with_validation(
        agent: Agent,
        prompt: str,
        output_type: Optional[Type[BaseModel]] = None,
        max_retries: int = 3,
        tool_call_required: bool = False,
        expected_tools: Optional[List[str]] = None,
        semantic_validator: Optional[Callable[[Any], Tuple[bool, List[str]]]] = None
    ) -> Tuple[RunResult, bool, List[str]]:
        """Réessaie avec validation jusqu'à obtenir un résultat valide.
        
        Args:
            agent: Agent à exécuter
            prompt: Prompt initial
            output_type: Type de sortie attendu (Pydantic model)
            max_retries: Nombre maximum de tentatives
            tool_call_required: Si True, exige des tool calls
            expected_tools: Liste des outils attendus
            semantic_validator: Fonction de validation sémantique
        
        Returns:
            Tuple (result, success, errors)
        """
        errors = []
        last_result = None
        
        for attempt in range(max_retries):
            try:
                # Construire le prompt avec instructions de retry si nécessaire
                current_prompt = prompt
                if attempt > 0:
                    current_prompt = (
                        f"{prompt}\n\n"
                        f"IMPORTANT - Tentative {attempt + 1}: "
                        "Assurez-vous d'utiliser les outils disponibles et de fournir un JSON valide."
                    )
                
                # Exécuter l'agent
                if output_type:
                    result = await agent.run(current_prompt, output_type=output_type)
                else:
                    result = await agent.run(current_prompt)
                
                last_result = result
                
                # Valider les tool calls si requis
                if tool_call_required:
                    is_valid, tool_errors = ToolCallDetector.validate_tool_calls_required(
                        result, expected_tools=expected_tools
                    )
                    if not is_valid:
                        errors.extend(tool_errors)
                        if attempt < max_retries - 1:
                            continue
                
                # Valider le format JSON si output_type est fourni
                if output_type:
                    # Si output_type est utilisé, PydanticAI valide automatiquement
                    # Mais on peut vérifier la sémantique
                    if semantic_validator and hasattr(result, 'data'):
                        is_valid, semantic_errors = semantic_validator(result.data)
                        if not is_valid:
                            errors.extend(semantic_errors)
                            if attempt < max_retries - 1:
                                continue
                
                # Si on arrive ici, c'est valide
                return result, True, []
                
            except ValidationError as e:
                error_msg = f"Erreur de validation (tentative {attempt + 1}): {str(e)}"
                errors.append(error_msg)
                logger.warning("Validation error on attempt %d: %s", attempt + 1, e)
                if attempt < max_retries - 1:
                    continue
            except (ValueError, TypeError, AttributeError) as e:
                error_msg = f"Erreur de type/valeur (tentative {attempt + 1}): {str(e)}"
                errors.append(error_msg)
                logger.warning("Type/value error on attempt %d: %s", attempt + 1, e, exc_info=True)
                if attempt < max_retries - 1:
                    await asyncio.sleep(0.5)  # Petit délai avant retry
                    continue
            except Exception as e:
                error_msg = f"Erreur d'exécution inattendue (tentative {attempt + 1}): {str(e)}"
                errors.append(error_msg)
                logger.error("Unexpected error on attempt %d: %s", attempt + 1, e, exc_info=True)
                if attempt < max_retries - 1:
                    await asyncio.sleep(0.5)  # Petit délai avant retry
                    continue
        
        # Créer un résultat vide si nécessaire
        if last_result is None:
            last_result = EmptyResult()
        
        return last_result, False, errors
    
    @staticmethod
    async def retry_with_fallback_prompt(
        agent: Agent,
        prompts: List[str],
        output_type: Optional[Type[BaseModel]] = None
    ) -> Tuple[RunResult, bool]:
        """Réessaie avec différents prompts en cas d'échec.
        
        Args:
            agent: Agent à exécuter
            prompts: Liste de prompts à essayer (du plus spécifique au plus générique)
            output_type: Type de sortie attendu
        
        Returns:
            Tuple (result, success)
        """
        for prompt in prompts:
            try:
                if output_type:
                    result = await agent.run(prompt, output_type=output_type)
                else:
                    result = await agent.run(prompt)
                
                # Vérifier que le résultat n'est pas vide
                if result.output and result.output.strip():
                    return result, True
            except Exception as e:
                logger.warning("Error with prompt: %s", e, exc_info=True)
                continue
        
        # Créer un résultat vide si nécessaire
        return EmptyResult(), False


# ============================================================================
# WRAPPERS DE VALIDATION POUR AGENTS
# ============================================================================

def with_tool_call_validation(
    expected_tools: Optional[List[str]] = None,
    min_calls: int = 1,
    raise_on_failure: bool = False
):
    """Décorateur pour valider les tool calls après exécution d'un agent.
    
    Args:
        expected_tools: Liste des outils attendus
        min_calls: Nombre minimum d'appels
        raise_on_failure: Si True, lève une exception en cas d'échec
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            result = await func(*args, **kwargs)
            
            # Check if result has the required attributes for tool call validation
            if hasattr(result, "all_messages") and callable(getattr(result, "all_messages", None)):
                try:
                    is_valid, errors = ToolCallDetector.validate_tool_calls_required(
                        result, expected_tools=expected_tools, min_calls=min_calls
                    )
                    
                    if not is_valid:
                        error_msg = f"Tool call validation failed: {', '.join(errors)}"
                        if raise_on_failure:
                            raise ValueError(error_msg)
                        else:
                            logger.warning("Tool call validation failed: %s", ', '.join(errors))
                except (AttributeError, TypeError) as e:
                    logger.warning("Error during tool call validation: %s", e, exc_info=True)
            else:
                logger.debug("Result does not have all_messages method, skipping tool call validation")
            
            return result
        return wrapper
    return decorator


def with_json_validation(
    output_type: Type[BaseModel],
    extract_from_text: bool = True,
    semantic_validator: Optional[Callable[[Any], Tuple[bool, List[str]]]] = None
):
    """Décorateur pour valider les sorties JSON d'un agent.
    
    Args:
        output_type: Modèle Pydantic attendu
        extract_from_text: Si True, essaie d'extraire JSON du texte
        semantic_validator: Fonction de validation sémantique
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            result = await func(*args, **kwargs)
            
            # Check if result has the required attributes for JSON validation
            has_data = hasattr(result, 'data')
            has_output = hasattr(result, 'output')
            
            if has_data or has_output:
                try:
                    # Si output_type est déjà utilisé, PydanticAI a validé
                    if has_data and result.data:
                        # Validation sémantique supplémentaire
                        if semantic_validator:
                            is_valid, errors = semantic_validator(result.data)
                            if not is_valid:
                                logger.warning("Semantic validation errors: %s", ', '.join(errors))
                    else:
                        # Essayer d'extraire et valider JSON du texte
                        if extract_from_text and has_output:
                            output_text = getattr(result, 'output', '')
                            if output_text:
                                json_data = JSONValidator.extract_json_from_text(output_text)
                                if json_data:
                                    is_valid, validation_error, validated = JSONValidator.validate_json_structure(
                                        json_data, output_type
                                    )
                                    if not is_valid:
                                        logger.warning(
                                            "JSON structure validation failed: %s",
                                            validation_error
                                        )
                except (AttributeError, TypeError) as e:
                    logger.warning("Error during JSON validation: %s", e, exc_info=True)
            else:
                logger.debug("Result does not have 'data' or 'output' attributes, skipping JSON validation")
            
            return result
        return wrapper
    return decorator


# ============================================================================
# AGENT WRAPPER AVEC MITIGATION COMPLÈTE
# ============================================================================

class SafeAgent:
    """Wrapper d'agent avec toutes les stratégies de mitigation."""

    def __init__(
        self,
        agent: Agent,
        output_type: Optional[Type[BaseModel]] = None,
        tool_call_required: bool = False,
        expected_tools: Optional[List[str]] = None,
        max_retries: int = 3,
        semantic_validator: Optional[Callable[[Any], Tuple[bool, List[str]]]] = None
    ):
        """Initialise un agent sécurisé.
        
        Args:
            agent: Agent à wrapper
            output_type: Type de sortie attendu
            tool_call_required: Si True, exige des tool calls
            expected_tools: Liste des outils attendus
            max_retries: Nombre maximum de tentatives
            semantic_validator: Fonction de validation sémantique
        """
        self.agent = agent
        self.output_type = output_type
        self.tool_call_required = tool_call_required
        self.expected_tools = expected_tools
        self.max_retries = max_retries
        self.semantic_validator = semantic_validator
    
    async def run_safe(
        self,
        prompt: str,
        **kwargs
    ) -> Tuple[RunResult, bool, List[str]]:
        """Exécute l'agent avec toutes les stratégies de mitigation."""
        errors = []
        success = True

        try:
            # Exécuter l'agent de base
            result = await self.agent.run(prompt, **kwargs)

            # Validation de sortie structurée si spécifiée
            if self.output_type:
                if hasattr(result, 'data') and result.data:
                    # Validation automatique réussie
                    validated_data = result.data
                else:
                    # Essayer de parser le texte de sortie
                    output_text = getattr(result, 'output', '')
                    if output_text:
                        try:
                            # Essayer de parser comme JSON
                            import json
                            json_data = json.loads(output_text)
                            # Valider avec Pydantic
                            validated_data = self.output_type.model_validate(json_data)
                            success = True
                        except (json.JSONDecodeError, ValidationError) as e:
                            errors.append(f"Échec de validation structurée: {e}")
                            success = False
                    else:
                        errors.append("Aucune sortie produite")
                        success = False

                # Validation sémantique si spécifiée
                if success and self.semantic_validator and validated_data:
                    is_valid, semantic_errors = self.semantic_validator(validated_data)
                    if not is_valid:
                        errors.extend(semantic_errors)
                        success = False

            # Validation des tool calls si requise
            if self.tool_call_required:
                tool_calls = ToolCallDetector.extract_tool_calls(result)
                if not tool_calls:
                    errors.append("Aucun tool call détecté alors que requis")
                    success = False
                elif self.expected_tools:
                    found_tools = {tc.get('name') for tc in tool_calls}
                    expected_set = set(self.expected_tools)
                    if not expected_set.issubset(found_tools):
                        missing = expected_set - found_tools
                        errors.append(f"Outils manquants: {', '.join(missing)}")
                        success = False

        except Exception as e:
            errors.append(f"Erreur lors de l'exécution: {str(e)}")
            success = False
            result = None

        return result, success, errors
    
    async def run(self, prompt: str, **kwargs) -> Any:
        """Exécute l'agent (interface compatible avec Agent.run)."""
        result, success, errors = await self.run_safe(prompt, **kwargs)
        
        if not success and errors and hasattr(result, 'output'):
            # Ajouter les erreurs au résultat si possible
            error_msg = f"\n\n[Erreurs de validation: {', '.join(errors)}]"
            # Créer une copie pour éviter de modifier l'original
            existing_output = getattr(result, 'output', '')
            new_output = existing_output + error_msg
            result = copy.copy(result)
            result.output = new_output
        
        return result


# ============================================================================
# UTILITAIRES DE VALIDATION SÉMANTIQUE
# ============================================================================

def create_portfolio_validator() -> Callable[[Any], Tuple[bool, List[str]]]:
    """Crée un validateur sémantique pour les portfolios."""
    def validate(data: Any) -> Tuple[bool, List[str]]:
        errors = []
        
        if isinstance(data, dict):
            # Vérifier que la valeur totale correspond à la somme des positions
            if "positions" in data and "valeur_totale" in data:
                positions = data.get("positions", [])
                valeur_totale = data.get("valeur_totale", 0)
                
                if isinstance(positions, list):
                    somme_positions = sum(
                        p.get("quantite", 0) * p.get("prix_achat", 0)
                        for p in positions
                        if isinstance(p, dict)
                    )
                    
                    # Tolérance de 1% pour les arrondis
                    if abs(somme_positions - valeur_totale) > valeur_totale * 0.01:
                        errors.append(
                            f"Valeur totale ({valeur_totale}) ne correspond pas à la somme des positions ({somme_positions})"
                        )
        
        return len(errors) == 0, errors
    
    return validate


def create_calculation_validator() -> Callable[[Any], Tuple[bool, List[str]]]:
    """Crée un validateur sémantique pour les calculs financiers."""
    def validate(data: Any) -> Tuple[bool, List[str]]:
        errors = []
        
        if isinstance(data, dict):
            # Vérifier la cohérence des calculs
            if "calculs" in data and isinstance(data["calculs"], list):
                for calc in data["calculs"]:
                    if isinstance(calc, dict):
                        calc_type = calc.get("type_calcul", "")
                        params = calc.get("parametres", {})
                        result = calc.get("resultat", 0)
                        validation = calc.get("validation", False)
                        
                        # Vérifier que validation est cohérente
                        if calc_type == "valeur_future":
                            capital = params.get("capital_initial", 0)
                            taux = params.get("taux_annuel", 0)
                            duree = params.get("duree_annees", 0)
                            
                            if capital > 0 and taux > 0 and duree > 0:
                                expected = capital * ((1 + taux) ** duree)
                                # Tolérance de 1%
                                if abs(result - expected) > expected * 0.01:
                                    errors.append(
                                        f"Calcul valeur future incohérent: attendu ~{expected:.2f}, obtenu {result:.2f}"
                                    )
        
        return len(errors) == 0, errors
    
    return validate

