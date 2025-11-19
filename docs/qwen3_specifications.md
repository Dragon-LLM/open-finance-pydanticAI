# Spécifications Qwen-3 8B - Fenêtre de contexte

## Fenêtre de contexte maximale

Le modèle **DragonLLM/Qwen-Open-Finance-R-8B** (basé sur Qwen-3 8B) supporte:

### Fenêtre de base
- **32 768 tokens** (32K tokens)
- Support natif pour la plupart des cas d'usage

### Fenêtre étendue (avec YaRN)
- **128 000 tokens** (128K tokens) 
- Extension via le mécanisme YaRN (Yet another RoPE extensioN)
- Nécessite une configuration spécifique pour activer

## Composition du contexte

Quand vous envoyez une requête, le contexte total inclut:

```
Contexte total = Prompt système + Messages conversation + Réponse générée
```

### Exemples pratiques:

| Type de requête | Prompt + Messages | Réponse max | Total |
|----------------|-------------------|-------------|-------|
| Question simple | ~100 tokens | 800 tokens | ~900 tokens |
| Analyse complexe | ~500 tokens | 1500 tokens | ~2000 tokens |
| Document long | ~5000 tokens | 2000 tokens | ~7000 tokens |
| Analyse très longue | ~15000 tokens | 4000 tokens | ~19000 tokens |

**Limite pratique recommandée:** 30 000 tokens pour laisser de la marge.

## Limite de génération (max_tokens)

**Limite théorique maximale:** **20 000 tokens** en sortie

**Limite pratique:** Dépend de la fenêtre de contexte disponible:
- Si contexte d'entrée = 2K tokens → peut générer jusqu'à ~30K tokens
- Si contexte d'entrée = 10K tokens → peut générer jusqu'à ~22K tokens  
- Si contexte d'entrée = 30K tokens → peut générer jusqu'à ~2K tokens

**Formule:** `max_tokens_generable = fenêtre_contexte - tokens_entrée - marge_sécurité`

## Configuration actuelle

Dans notre application PydanticAI:
- `max_tokens` (génération): **1500 tokens** (configurable)
- Contexte d'entrée: Illimité jusqu'à ~30K tokens (pour laisser de la marge)
- Contexte total: Jusqu'à 32K tokens (base) ou 128K (avec YaRN)
- Limite théorique max: 20K tokens en sortie (mais contrainte par contexte disponible)

## Recommandations

### Pour des requêtes simples:
```python
max_tokens = 800-1000  # Suffisant pour la plupart des réponses
```

### Pour des requêtes complexes (SWIFT, analyses):
```python
max_tokens = 1500-2000  # Permet raisonnement + réponse complète
```

### Pour des documents longs:
- Utilisez le contexte jusqu'à ~30K tokens pour le prompt
- Réservez 2-5K tokens pour la réponse
- Total: jusqu'à 32K tokens (base)

### Activation de YaRN pour contexte étendu:
Si vous avez besoin de plus de 32K tokens:
1. Vérifiez que le backend Transformers supporte YaRN
2. Configurez les paramètres de RoPE scaling
3. La fenêtre peut être étendue jusqu'à 128K tokens

## Références

- Qwen-3 models: Fenêtre de 32K tokens (base), 128K avec YaRN
- YaRN: Yet another RoPE extensioN - méthode d'extension de contexte
- Documentation technique Qwen: https://huggingface.co/Qwen/Qwen2.5

