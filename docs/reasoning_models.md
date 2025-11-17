# Gestion des modèles de raisonnement avec PydanticAI

## Problème: "finish on length"

Quand vous voyez `finish_reason: "length"`, cela signifie que le modèle a atteint la limite de `max_tokens` avant de terminer sa réponse.

## Pourquoi c'est fréquent avec les modèles de raisonnement?

Les modèles comme Qwen3 utilisent des balises `<think>` (ou `<think>`) pour le raisonnement en chaîne:

```
<think>
1. L'utilisateur demande un message SWIFT MT103
2. Je dois identifier les champs requis
3. Format: :20: référence, :32A: date/devise/montant...
</think>

Voici le message SWIFT généré:
:20:NONREF
:23B:CRED
...
```

**Le raisonnement peut consommer 40-60% du budget de tokens!**

## Solution: Augmenter max_tokens

Nous avons configuré `max_tokens=1500` dans `app/config.py` pour permettre:
- ~600-900 tokens pour le raisonnement (`<think>` tags)
- ~600-900 tokens pour la réponse finale
- Total: ~1500 tokens pour des réponses complètes

## Configuration actuelle

```python
# app/config.py
max_tokens: int = 1500  # Pour modèles de raisonnement

# app/models.py
model_settings = ModelSettings(
    max_output_tokens=settings.max_tokens,
)
finance_model = OpenAIModel(
    ...,
    model_settings=model_settings,
)
```

## Recommandations par type de requête

| Type de requête | max_tokens recommandé |
|----------------|----------------------|
| Questions simples | 800-1000 |
| Génération SWIFT | 1200-1500 |
| Analyse complexe | 1500-2000 |
| Extraction structurée | 1000-1200 |

## Comment ajuster pour un agent spécifique?

Vous pouvez créer des agents avec des settings différents:

```python
from pydantic_ai import ModelSettings, Agent

# Agent pour tâches courtes
short_agent = Agent(
    finance_model,
    model_settings=ModelSettings(max_output_tokens=800),
    system_prompt="..."
)

# Agent pour tâches longues (SWIFT, analyses)
long_agent = Agent(
    finance_model,
    model_settings=ModelSettings(max_output_tokens=2000),
    system_prompt="..."
)
```

## Vérifier si la réponse est complète

Notre utilitaire `extract_answer_from_reasoning()` dans `app/utils.py` gère automatiquement:
- Extraction de la réponse après les balises `<think>`
- Détection si la réponse est tronquée
- Nettoyage des balises de raisonnement









