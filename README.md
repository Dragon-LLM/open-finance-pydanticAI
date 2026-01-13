# Open Finance PydanticAI

[![PydanticAI](https://img.shields.io/badge/PydanticAI-1.18+-blue?logo=python)](https://ai.pydantic.dev/)
[![Logfire](https://img.shields.io/badge/Logfire-Observability-orange)](https://logfire.pydantic.dev/)
[![Langfuse](https://img.shields.io/badge/Langfuse-Tracing-green)](https://langfuse.com/)
[![Koyeb](https://img.shields.io/badge/Koyeb-Deploy-purple)](https://koyeb.com/)
[![HuggingFace](https://img.shields.io/badge/HF%20Spaces-Live-yellow)](https://huggingface.co/spaces)
[![Ollama](https://img.shields.io/badge/Ollama-Local-gray)](https://ollama.ai/)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)

🇫🇷 [Version française](README_FR.md)

Demo project exploring PydanticAI agents for financial tasks. Features tool calling, structured outputs, and dual observability with Langfuse and Logfire.

**Backend**: Requires an LLM server. See [Dragon-LLM/simple-open-finance-8B](https://github.com/Dragon-LLM/simple-open-finance-8B) for deployment instructions.

## About PydanticAI

[PydanticAI](https://ai.pydantic.dev/) is a framework for building AI agents with type-safe structured outputs, tool calling, and memory. It leverages Pydantic schemas for validation and integrates seamlessly with OpenAI-compatible APIs.

**Key features:**
- Structured outputs with automatic validation
- Tool calling with Python functions
- Memory and context management
- Type-safe agent definitions

**Example: Agent with tools**

```python
from pydantic_ai import Agent, ModelSettings
from pydantic import BaseModel

# Define a tool
def calculer_valeur_future(capital: float, taux: float, duree: float) -> str:
    """Calculate future value with compound interest."""
    import numpy_financial as npf
    return f"FV: {npf.fv(taux, duree, 0, -capital):,.2f}€"

# Define structured output
class Result(BaseModel):
    calculation_type: str
    result: float
    explanation: str

# Create agent
agent = Agent(
    model,
    tools=[calculer_valeur_future],
    output_type=Result,
    system_prompt="Financial advisor. Use tools for calculations."
)

# Run agent
result = await agent.run("50000€ at 4% for 10 years. Future value?")
```

See `examples/agent_2.py` for a complete implementation with multiple financial tools.

---

## Disclaimer

These are toy examples for learning and experimentation. Real financial software requires compliance frameworks, audit trails, regulatory validation, and rigorous engineering. Use accordingly.

---

## Gradio Interface

A web UI for interacting with all agents without writing code.

![Gradio Interface](docs/screenshot.png)

```bash
python app/gradio_app.py
# Open http://localhost:7860
```

**Features:**
- Tabbed interface with one tab per agent
- Endpoint selector to switch between Koyeb, HuggingFace, Ollama, or LLM Pro Finance
- Real-time server health monitoring with wake-up for sleeping services
- Observability panel with toggles for Langfuse and Logfire
- Tool call tracking showing which tools were invoked and execution metrics

---

## Agents

Six demo agents showcasing different PydanticAI patterns:

| Agent | Task | Tools | Description |
|-------|------|-------|-------------|
| **1** | Portfolio Extraction | Pydantic schemas | Extracts structured portfolio data from unstructured text |
| **2** | Financial Calculator | numpy-financial | Computes FV, NPV, IRR, loan payments |
| **3** | Risk & Tax Advisor | Multi-agent | Orchestrates risk analyst, tax advisor, portfolio optimizer |
| **4** | Option Pricing | QuantLib | Black-Scholes pricing and Greeks calculation |
| **5** | SWIFT/ISO 20022 | Custom parsers | Message conversion, validation, AML risk scoring |
| **6** | Judge | 70B model | Evaluates outputs from other agents |

All agent implementations are in `examples/agent_*.py`.

---

## Models

| Endpoint | Model | Parameters | Use Case |
|----------|-------|------------|----------|
| Koyeb | Dragon LLM Open Finance Qwen 8B | 8B | Default for all agents |
| HuggingFace Spaces | Dragon LLM Open Finance Qwen 8B | 8B | Persistent alternative |
| Ollama | User-configured | Variable | Local inference |
| LLM Pro Finance | Llama 70B | 70B | Judge agent evaluations |

All endpoints expose OpenAI-compatible APIs. The 8B model handles tool calling and structured outputs. The 70B model provides higher-quality evaluation for the Judge agent.

---

## Observability

Observability is essential for LLM applications. This project integrates two platforms:

**Logfire** (Pydantic)
- Automatic instrumentation of all PydanticAI agents
- Traces agent runs, tool calls, and LLM generations without code changes
- Native integration with Pydantic ecosystem
- **[Logfire Evals](https://ai.pydantic.dev/evals/)**: New evaluation framework for systematic agent testing

**Langfuse** (LLM-focused)
- Detailed trace management with hierarchical spans
- Evaluation datasets and scoring
- Cost tracking and usage analytics

### What's Tracked

| Metric | Logfire | Langfuse | Description |
|--------|---------|----------|-------------|
| Agent runs | ✅ | ✅ | Start/end, duration, success/failure |
| Tool calls | ✅ | ✅ | Which tools, arguments, results |
| Token usage | ✅ | ✅ | Input/output tokens per generation |
| Latency | ✅ | ✅ | Response times per span |
| Structured outputs | ✅ | ✅ | Pydantic model validation |
| Context overflow | ✅ | — | Detects when context limit exceeded |
| Tool call anomalies | ✅ | — | Flags excessive tool loops |
| Evaluation scores | ✅ | ✅ | Correctness, efficiency metrics |

### Alerts & Dashboards

With Logfire, you can configure alerts for:
- **Context overflow**: Agent exceeds model's context window
- **Tool call anomalies**: Unusual tool invocation patterns (loops, retries)
- **High latency**: Response times exceeding thresholds

See `docs/logfire_setup.md` for SQL queries to set up alerts and dashboards.

### Configuration

```env
# Langfuse
ENABLE_LANGFUSE=true
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com

# Logfire
ENABLE_LOGFIRE=true
LOGFIRE_TOKEN=...  # or authenticate via: logfire auth
```

Both platforms can run simultaneously. The Gradio UI provides runtime toggles to enable or disable each platform without restarting.

---

## Installation

```bash
# Base installation
pip install -e ".[dev]"

# With QuantLib for option pricing (Agent 4)
pip install -e ".[dev,quant]"
```

## Configuration

Create a `.env` file:

```env
ENDPOINT=koyeb
API_KEY=not-needed
MAX_TOKENS=1500

# Optional: LLM Pro Finance for Judge agent
LLM_PRO_FINANCE_KEY=your-api-key
LLM_PRO_FINANCE_URL=https://demo.llmprofinance.com

# Optional: Local Ollama
OLLAMA_MODEL=dragon-llm
```

## Running

```bash
# Start the Gradio interface
python app/gradio_app.py

# Run Logfire evaluations
python examples/run_logfire_evaluation.py --all --max-items 3

# Run Langfuse evaluations
python examples/run_langfuse_evaluation.py --agents agent_1 agent_2 --max-items 3

# Run Pydantic Evals (official framework)
python examples/run_pydantic_evals.py --all --max-cases 3
```

---

## Project Structure

```
app/
├── gradio_app.py          # Web interface
├── observability.py       # Unified Langfuse + Logfire handler
├── config.py              # Settings and endpoint configuration
├── models.py              # Model instantiation per endpoint
├── langfuse_*.py          # Langfuse integration
├── logfire_*.py           # Logfire integration and metrics

examples/
├── agent_1.py             # Portfolio extraction
├── agent_2.py             # Financial calculations
├── agent_3.py             # Multi-agent risk/tax workflow
├── agent_4.py             # Option pricing (QuantLib)
├── agent_5.py             # SWIFT/ISO 20022 conversion
├── agent_5_validator.py   # Message validation
├── agent_5_risk.py        # AML risk assessment
├── judge_agent.py         # 70B evaluation agent
├── run_langfuse_evaluation.py
├── run_logfire_evaluation.py
└── run_pydantic_evals.py
```

---

## References

- [PydanticAI](https://ai.pydantic.dev/) — Agent framework
- [Logfire](https://logfire.pydantic.dev/) — Pydantic observability
- [Langfuse](https://langfuse.com/) — LLM tracing and evaluation
- [Dragon-LLM/simple-open-finance-8B](https://github.com/Dragon-LLM/simple-open-finance-8B) — Server deployment
- [vLLM](https://github.com/vllm-project/vllm) — Inference engine
- [numpy-financial](https://numpy.org/numpy-financial/) — Financial calculations
- [QuantLib](https://www.quantlib.org/) — Option pricing

---

MIT License
