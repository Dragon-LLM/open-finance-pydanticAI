# Open Finance PydanticAI

Application Python utilisant PydanticAI pour créer des agents intelligents spécialisés en finance, connectés au modèle DragonLLM/Qwen-Open-Finance-R-8B via une API OpenAI-compatible déployée sur Hugging Face Spaces.

## Qu'est-ce que PydanticAI ?

PydanticAI est un framework agentique Python conçu pour simplifier le développement d'applications de production basées sur l'intelligence artificielle générative. En tant que framework agentique, PydanticAI permet de créer des agents capables d'interagir avec leur environnement, d'appeler des fonctions et de communiquer avec d'autres agents.

Le framework met l'accent sur la sécurité des types et la validation des données en s'appuyant sur les capacités de validation de Pydantic, tout en facilitant la construction de systèmes d'IA complexes et modulaires.

### Points forts

- **Framework agentique** : Création d'agents capables d'interagir avec leur environnement et de communiquer entre eux
- Validation de types stricte avec Pydantic pour garantir la fiabilité des applications
- Agents modulaires et extensibles avec intégration native d'outils Python
- Compatibilité avec tous les modèles (OpenAI, Anthropic, Gemini, etc.)
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
│   ├── agent_2_tools.py                # Calculs financiers (numpy-financial)
│   ├── agent_2_tools_quant.py          # Analyse de risque avancée (QuantLib)
│   ├── agent_option_pricing.py         # Pricing d'options via QuantLib
│   ├── agent_2_compliance.py           # Contrôle compliance (vérifie les tool calls)
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
MODEL_NAME=dragon-llm-open-finance
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
from examples.agent_2_tools import finance_calculator_agent

result = await finance_calculator_agent.run(
    "J'ai 50 000€ à placer à 4% par an pendant 10 ans. Combien aurai-je?"
)
```

Voir le répertoire `examples/` pour plus d'exemples détaillés.

## Configuration du modèle

Le projet est configuré pour utiliser le modèle `DragonLLM/Qwen-Open-Finance-R-8B` via l'espace Hugging Face qui expose une API compatible OpenAI.

Caractéristiques du modèle :

- Spécialisé en terminologie financière française
- Fenêtre de contexte : 32K tokens (base), 128K avec YaRN
- Limite de génération : ~20K tokens (théorique), pratique 2-3K recommandé
- Support du raisonnement avec tags `<think>`
- Support des tool calls (fonctionnalité activée)

Voir `docs/qwen3_specifications.md` pour plus de détails.

## Exemples disponibles

1. **Extraction de données structurées** (`agent_1_structured_data.py`)
   - Extraction et validation automatique via Pydantic

2. **Agent 2 – Calculs financiers** (`agent_2_tools.py`)
   - Intérêts composés, prêts immobiliers, performances de portefeuille
   - Outils numpy-financial testés et surveillés

3. **Agent 2 Quant – Risk Ops** (`agent_2_tools_quant.py`)
   - VaR paramétrique / historique / Monte Carlo, analyse de portfolio, métriques ajustées
   - S'adresse aux risk managers et gérants institutionnels

4. **Agent option pricing** (`agent_option_pricing.py`)
   - Pricing d’un call européen + Greeks via QuantLib
   - Exemple compact compatible modèles 8 B

5. **Agent 2 Compliance** (`agent_2_compliance.py`)
   - Enveloppe l’agent financier et vérifie l’usage effectif des outils
   - Génère un avis “Conforme / Non conforme”

6. **Workflows multi-étapes** (`agent_3_multi_step.py`)
   - Coordination d’agents spécialisés (risque, fiscalité, optimisation)

7. **Tests de tool calls** (`test_tool_calls_simple.py`)
   - Harness minimal pour valider la chaîne tool-calling

> **Note importante** : Les exemples fournis dans ce projet sont générés à des fins de démonstration et d'apprentissage. Ils ne sont pas issus de cas d'usage réels de production. Nous encourageons vivement les utilisateurs à proposer des exemples plus pertinents et réalistes basés sur leurs propres expériences via des [Issues GitHub](https://github.com/DealExMachina/open-finance-pydanticAI/issues). Vos contributions aideront à améliorer la qualité et la pertinence des exemples pour la communauté.

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
python examples/agent_2_tools.py
python examples/agent_2_tools_quant.py
python examples/agent_option_pricing.py
python examples/agent_2_compliance.py
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

## Références

### Modèle de langage

- **DragonLLM/Qwen-Open-Finance-R-8B**: Modèle de langage spécialisé en finance, partie de la suite LLM Pro Finance. Caillaut, G., Qader, R., Liu, J., Nakhlé, M., Sadoune, A., Ahmim, M., & Barthelemy, J.-G. (2025). The LLM Pro Finance Suite: Multilingual Large Language Models for Financial Applications. *arXiv preprint arXiv:2511.08621*. [https://arxiv.org/abs/2511.08621](https://arxiv.org/abs/2511.08621)
- **Qwen**: Modèle de langage de base développé par Alibaba Cloud. [https://huggingface.co/Qwen](https://huggingface.co/Qwen)

### Frameworks et bibliothèques principales

- **PydanticAI**: Framework agentique Python pour construire des applications de production basées sur l'intelligence artificielle générative. Permet de créer des agents capables d'interagir avec leur environnement, d'appeler des fonctions et de communiquer entre eux. [https://ai.pydantic.dev/](https://ai.pydantic.dev/)
- **Hugging Face**: Plateforme pour le machine learning et le déploiement de modèles. [https://huggingface.co/](https://huggingface.co/)
- **FastAPI**: Framework web moderne et rapide pour construire des APIs avec Python. [https://fastapi.tiangolo.com/](https://fastapi.tiangolo.com/)
- **Pydantic**: Validation de données en Python utilisant les annotations de type. [https://docs.pydantic.dev/](https://docs.pydantic.dev/)

### Bibliothèques financières

- **numpy-financial**: Bibliothèque Python pour les calculs financiers (valeur future, prêts, etc.). [https://numpy.org/numpy-financial/](https://numpy.org/numpy-financial/)
- **QuantLib-Python**: Bibliothèque quantitative pour la finance (pricing d'options, VaR, etc.). [https://www.quantlib.org/](https://www.quantlib.org/)

### Observabilité

- **Logfire**: Plateforme d'observabilité pour applications Python basée sur OpenTelemetry. [https://logfire.pydantic.dev/](https://logfire.pydantic.dev/)

### Autres dépendances

- **NumPy**: Bibliothèque fondamentale pour le calcul scientifique en Python. [https://numpy.org/](https://numpy.org/)
- **Pandas**: Bibliothèque d'analyse de données. [https://pandas.pydata.org/](https://pandas.pydata.org/)
- **SciPy**: Bibliothèque scientifique pour Python. [https://scipy.org/](https://scipy.org/)

## Licence

Ce projet est fourni à des fins éducatives et de démonstration.
