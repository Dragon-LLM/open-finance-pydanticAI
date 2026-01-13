# Open Finance PydanticAI

[![PydanticAI](https://img.shields.io/badge/PydanticAI-1.18+-blue?logo=python)](https://ai.pydantic.dev/)
[![Logfire](https://img.shields.io/badge/Logfire-Observability-orange)](https://logfire.pydantic.dev/)
[![Langfuse](https://img.shields.io/badge/Langfuse-Tracing-green)](https://langfuse.com/)
[![Koyeb](https://img.shields.io/badge/Koyeb-Deploy-purple)](https://koyeb.com/)
[![HuggingFace](https://img.shields.io/badge/HF%20Spaces-Live-yellow)](https://huggingface.co/spaces)
[![Ollama](https://img.shields.io/badge/Ollama-Local-gray)](https://ollama.ai/)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)

🇬🇧 [English version](README.md)

Projet de démonstration explorant les agents PydanticAI pour des tâches financières. Inclut tool calling, sorties structurées, et double observabilité via Langfuse et Logfire.

**Backend**: Nécessite un serveur LLM. Voir [Dragon-LLM/simple-open-finance-8B](https://github.com/Dragon-LLM/simple-open-finance-8B) pour le déploiement.

## À propos de PydanticAI

[PydanticAI](https://ai.pydantic.dev/) est un framework pour construire des agents IA avec sorties structurées type-safe, tool calling et mémoire. Il utilise les schémas Pydantic pour la validation et s'intègre avec les APIs compatibles OpenAI.

**Fonctionnalités principales :**
- Sorties structurées avec validation automatique
- Tool calling avec fonctions Python
- Gestion mémoire et contexte
- Définitions d'agents type-safe

**Exemple : Agent avec outils**

```python
from pydantic_ai import Agent, ModelSettings
from pydantic import BaseModel

# Définir un outil
def calculer_valeur_future(capital: float, taux: float, duree: float) -> str:
    """Calcule la valeur future avec intérêts composés."""
    import numpy_financial as npf
    return f"VF: {npf.fv(taux, duree, 0, -capital):,.2f}€"

# Définir sortie structurée
class Result(BaseModel):
    calculation_type: str
    result: float
    explanation: str

# Créer agent
agent = Agent(
    model,
    tools=[calculer_valeur_future],
    output_type=Result,
    system_prompt="Conseiller financier. Utilise les outils pour les calculs."
)

# Exécuter agent
result = await agent.run("50000€ à 4% sur 10 ans. Valeur future?")
```

Voir `examples/agent_2.py` pour une implémentation complète avec plusieurs outils financiers.

---

## Avertissement

Ce sont des exemples de démonstration à but pédagogique. Les logiciels financiers réels exigent des cadres de conformité, des pistes d'audit, une validation réglementaire et une ingénierie rigoureuse. À utiliser en connaissance de cause.

---

## Interface Gradio

Une interface web pour interagir avec tous les agents sans écrire de code.

![Interface Gradio](docs/screenshot.png)

```bash
python app/gradio_app.py
# Ouvrir http://localhost:7860
```

**Fonctionnalités :**
- Interface à onglets avec un onglet par agent
- Sélecteur d'endpoint pour basculer entre Koyeb, HuggingFace, Ollama ou LLM Pro Finance
- Monitoring santé serveur en temps réel avec réveil des services en veille
- Panneau d'observabilité avec toggles pour Langfuse et Logfire
- Suivi des appels d'outils avec métriques d'exécution

---

## Agents

Six agents de démonstration illustrant différents patterns PydanticAI :

| Agent | Tâche | Outils | Description |
|-------|-------|--------|-------------|
| **1** | Extraction de portefeuille | Schémas Pydantic | Extrait des données structurées de texte libre |
| **2** | Calculatrice financière | numpy-financial | Calcule VF, VAN, TRI, mensualités |
| **3** | Conseil risque & fiscal | Multi-agent | Orchestre analyste risque, conseiller fiscal, optimiseur |
| **4** | Pricing d'options | QuantLib | Black-Scholes et calcul des Greeks |
| **5** | SWIFT/ISO 20022 | Parsers custom | Conversion, validation, scoring risque AML |
| **6** | Juge | Modèle 70B | Évalue les sorties des autres agents |

Toutes les implémentations sont dans `examples/agent_*.py`.

---

## Modèles

| Endpoint | Modèle | Paramètres | Usage |
|----------|--------|------------|-------|
| Koyeb | Dragon LLM Open Finance Qwen 8B | 8B | Défaut pour tous les agents |
| HuggingFace Spaces | Dragon LLM Open Finance Qwen 8B | 8B | Alternative persistante |
| Ollama | Configurable | Variable | Inférence locale |
| LLM Pro Finance | Llama 70B | 70B | Évaluations agent Juge |

Tous les endpoints exposent des APIs compatibles OpenAI. Le modèle 8B gère le tool calling et les sorties structurées. Le modèle 70B fournit une évaluation de meilleure qualité pour l'agent Juge.

---

## Observabilité

L'observabilité est essentielle pour les applications LLM. Ce projet intègre deux plateformes :

**Logfire** (Pydantic)
- Instrumentation automatique de tous les agents PydanticAI
- Trace les runs d'agents, appels d'outils et générations LLM sans modification de code
- Intégration native avec l'écosystème Pydantic

**Langfuse** (orienté LLM)
- Gestion détaillée des traces avec spans hiérarchiques
- Datasets d'évaluation et scoring
- Suivi des coûts et analytics d'usage

### Configuration

```env
# Langfuse
ENABLE_LANGFUSE=true
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com

# Logfire
ENABLE_LOGFIRE=true
LOGFIRE_TOKEN=...  # ou authentification via: logfire auth
```

Les deux plateformes peuvent fonctionner simultanément. L'interface Gradio fournit des toggles pour activer ou désactiver chaque plateforme sans redémarrage.

---

## Installation

```bash
# Installation de base
pip install -e ".[dev]"

# Avec QuantLib pour le pricing d'options (Agent 4)
pip install -e ".[dev,quant]"
```

## Configuration

Créer un fichier `.env` :

```env
ENDPOINT=koyeb
API_KEY=not-needed
MAX_TOKENS=1500

# Optionnel: LLM Pro Finance pour l'agent Juge
LLM_PRO_FINANCE_KEY=votre-clé-api
LLM_PRO_FINANCE_URL=https://demo.llmprofinance.com

# Optionnel: Ollama local
OLLAMA_MODEL=dragon-llm
```

## Exécution

```bash
# Démarrer l'interface Gradio
python app/gradio_app.py

# Lancer la suite d'évaluation
python examples/evaluate_all_agents.py

# Exécuter avec traçage Langfuse
python examples/run_all_evaluations.py --endpoint koyeb --max-items 5
```

---

## Structure du projet

```
app/
├── gradio_app.py       # Interface web
├── observability.py    # Handler unifié Langfuse + Logfire
├── config.py           # Paramètres et configuration endpoints
└── models.py           # Instanciation des modèles par endpoint

examples/
├── agent_1.py          # Extraction de portefeuille
├── agent_2.py          # Calculs financiers
├── agent_3.py          # Conseil risque et fiscal
├── agent_4.py          # Pricing d'options
├── agent_5.py          # Conversion SWIFT/ISO 20022
├── agent_5_validator.py
├── agent_5_risk.py
├── judge_agent.py      # Évaluation des sorties
└── evaluate_*.py       # Scripts d'évaluation
```

---

## Références

- [PydanticAI](https://ai.pydantic.dev/) — Framework d'agents
- [Logfire](https://logfire.pydantic.dev/) — Observabilité Pydantic
- [Langfuse](https://langfuse.com/) — Traçage et évaluation LLM
- [Dragon-LLM/simple-open-finance-8B](https://github.com/Dragon-LLM/simple-open-finance-8B) — Déploiement serveur
- [vLLM](https://github.com/vllm-project/vllm) — Moteur d'inférence
- [numpy-financial](https://numpy.org/numpy-financial/) — Calculs financiers
- [QuantLib](https://www.quantlib.org/) — Pricing d'options

---

Licence MIT
