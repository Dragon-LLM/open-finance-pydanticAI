# Open Finance PydanticAI

Application Python utilisant PydanticAI pour créer des agents intelligents spécialisés en finance, connectés au modèle Qwen3-8B-Fin via une API OpenAI-compatible déployée sur Hugging Face Spaces.

## Qu'est-ce que PydanticAI ?

PydanticAI est un framework Python moderne pour construire des applications basées sur des LLM (Large Language Models). Il combine la puissance de Pydantic pour la validation de types et la génération de structures de données avec un système d'agents facile à utiliser.

### Points forts

- Validation de types stricte avec Pydantic
- Agents modulaires et extensibles avec intégration native d'outils Python
- Compatibilité avec tous les modèles (OpenAI, Anthropic, etc.)
- Production-ready avec gestion automatique des erreurs
- Développement rapide avec syntaxe claire et Pythonic

## Architecture

```
open-finance-pydanticAI/
├── app/
│   ├── config.py          # Configuration de l'application
│   ├── models.py          # Configuration du modèle PydanticAI
│   ├── agents.py          # Définition des agents financiers
│   ├── main.py            # API FastAPI
│   └── utils.py           # Utilitaires (parsing, extraction)
├── examples/
│   ├── agent_1_structured_data.py      # Extraction de données structurées
│   ├── agent_2_tools_improved.py      # Agents avec outils Python
│   ├── agent_3_multi_step.py           # Workflows multi-étapes
│   └── test_tool_calls_simple.py       # Tests de vérification des tool calls
└── docs/
    ├── qwen3_specifications.md         # Spécifications du modèle
    ├── reasoning_models.md              # Gestion des modèles de raisonnement
    └── generation_limits.md             # Limites de génération
```

## Installation

### Prérequis

- Python 3.10+
- Accès à l'espace Hugging Face `jeanbaptdzd/open-finance-llm-8b`

### Installation des dépendances

```bash
pip install -e ".[dev]"
```

### Variables d'environnement

Créez un fichier `.env` :

```env
HF_SPACE_URL=https://jeanbaptdzd-open-finance-llm-8b.hf.space
API_KEY=not-needed
MODEL_NAME=gpt-3.5-turbo
MAX_TOKENS=1500
TIMEOUT=120
```

## Utilisation

### API FastAPI

Démarrer le serveur :

```bash
uvicorn app.main:app --reload
```

Endpoints disponibles :

- `GET /` - Informations sur le service
- `GET /health` - Health check
- `POST /ask` - Poser une question financière

Exemple de requête :

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Qu'est-ce qu'une date de valeur?"}'
```

### Exemples d'agents

#### Extraction de données structurées

```python
from app.agents import finance_agent

result = await finance_agent.run(
    "Mon portfolio: 50 actions AIR.PA à 120€, 30 actions SAN.PA à 85€"
)
```

#### Agent avec outils de calcul

```python
from examples.agent_2_tools_improved import finance_calculator_agent

result = await finance_calculator_agent.run(
    "J'ai 50 000€ à placer à 4% par an pendant 10 ans. Combien aurai-je?"
)
```

Voir le répertoire `examples/` pour plus d'exemples détaillés.

## Configuration du modèle

Le projet est configuré pour utiliser le modèle `DragonLLM/qwen3-8b-fin-v1.0` via l'espace Hugging Face qui expose une API compatible OpenAI.

Caractéristiques du modèle :

- Spécialisé en terminologie financière française
- Fenêtre de contexte : 32K tokens (base), 128K avec YaRN
- Limite de génération : ~20K tokens (théorique), pratique 2-3K recommandé
- Support du raisonnement avec tags `<think>`
- Support des tool calls (fonctionnalité activée)

Voir `docs/qwen3_specifications.md` pour plus de détails.

## Exemples disponibles

1. **Extraction de données structurées** (`agent_1_structured_data.py`)
   - Extraction de données financières depuis du texte non structuré

2. **Agents avec outils** (`agent_2_tools_improved.py`)
   - Intégration d'outils Python pour calculs financiers précis (intérêts composés, prêts, performance)
   - Utilise numpy-financial pour des calculs testés et précis

3. **Workflows multi-étapes** (`agent_3_multi_step.py`)
   - Coordination de plusieurs agents spécialisés

4. **Tests** (`test_tool_calls_simple.py`)
   - Vérification du fonctionnement des tool calls

## Développement

### Formatage et linting

```bash
black .
ruff check .
mypy app
```

### Tests

Les exemples dans `examples/` servent également de tests d'intégration :

```bash
python examples/test_tool_calls_simple.py
python examples/agent_2_tools_improved.py
```

## Limitations et bonnes pratiques

Limites de génération :

- Recommandé : 1500-2000 tokens pour la plupart des cas
- Maximum pratique : ~3000 tokens selon le contexte

Meilleures pratiques :

- Utiliser des `max_tokens` adaptés à chaque type d'agent
- Implémenter une gestion de mémoire pour les conversations longues
- Valider les réponses structurées avec Pydantic
- Utiliser des outils Python pour les calculs critiques plutôt que de compter sur le LLM

## Documentation technique

- `docs/qwen3_specifications.md` - Spécifications détaillées du modèle Qwen3
- `docs/reasoning_models.md` - Gestion des modèles avec raisonnement
- `docs/generation_limits.md` - Limites de génération et optimisation
- `docs/financial_libraries_recommendations.md` - Recommandations de bibliothèques financières

## Licence

Ce projet est fourni à des fins éducatives et de démonstration.
