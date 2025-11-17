# Limites de génération - Qwen-3 8B

## Limite théorique maximale

**20 000 tokens** peuvent être générés en sortie (selon les spécifications Qwen-3 8B).

## Limite pratique

La limite pratique dépend de la **fenêtre de contexte disponible**:

```
max_tokens_generable = fenêtre_contexte - tokens_entrée - marge_sécurité
```

### Exemples pratiques

| Contexte d'entrée | Fenêtre totale | Max génération | Marge |
|-------------------|----------------|----------------|-------|
| 2K tokens | 32K | ~30K tokens | ✅ Large |
| 10K tokens | 32K | ~22K tokens | ✅ Bonne |
| 20K tokens | 32K | ~12K tokens | ✅ Suffisant |
| 30K tokens | 32K | ~2K tokens | ⚠️ Limite |
| 50K tokens | 128K (YaRN) | ~78K tokens | ✅ Très large |

## Pour notre application

### Configuration actuelle
- **max_tokens configuré:** 1500 tokens
- **Typique contexte entrée:** ~100-500 tokens (messages conversation)
- **Disponible pour génération:** ~30K tokens

### Pourquoi 1500 tokens est suffisant?

1. **Questions simples:** 800-1000 tokens suffisent
2. **Analyses complexes:** 1500 tokens couvrent raisonnement + réponse
3. **Messages SWIFT:** 1200-1500 tokens pour format complet
4. **Marge de sécurité:** Reste bien en dessous de la limite pratique

## Ajuster max_tokens selon les besoins

### Questions simples (max_tokens=800)
```python
agent_short = Agent(
    finance_model,
    model_settings=ModelSettings(max_output_tokens=800),
)
```

### Analyses complexes (max_tokens=2000)
```python
agent_long = Agent(
    finance_model,
    model_settings=ModelSettings(max_output_tokens=2000),
)
```

### Documents très longs (max_tokens=5000)
```python
agent_very_long = Agent(
    finance_model,
    model_settings=ModelSettings(max_output_tokens=5000),
)
# Nécessite que l'entrée soit < 27K tokens
```

## Recommandations

| Cas d'usage | max_tokens recommandé | Notes |
|-------------|----------------------|-------|
| Questions rapides | 800-1000 | Suffisant pour la plupart |
| Réponses détaillées | 1500-2000 | Inclut raisonnement |
| Messages SWIFT | 1200-1500 | Format structuré |
| Analyses longues | 2000-4000 | Si nécessaire |
| Génération de code/docs | 3000-5000 | Documents complets |

**Note:** Au-delà de 5000 tokens, vérifiez que votre contexte d'entrée n'est pas trop volumineux.









