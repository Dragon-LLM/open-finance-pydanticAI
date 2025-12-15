# Open Finance PydanticAI

Research project evaluating small language models (8B parameters) for tool calling and structured output generation in financial applications using PydanticAI.

## Overview

This system demonstrates that 8B parameter models can reliably perform financial tasks when:
- Tool calling is explicitly enforced via system prompts
- Structured outputs are validated using Pydantic schemas
- Client-side verification is performed for critical calculations

The application provides six specialized agents for financial analysis, accessible via a Gradio web interface or programmatic API. All agents use OpenAI-compatible endpoints with support for multiple deployment backends.

## Architecture

### Agents

**Agent 1: Portfolio Extractor** (`examples/agent_1.py`)
- Extracts structured portfolio data from unstructured text
- Identifies stock symbols, quantities, purchase prices, and dates
- Outputs validated `Portfolio` schema with calculated total value
- Uses client-side arithmetic verification to correct model calculation errors

**Agent 2: Financial Calculator** (`examples/agent_2.py`)
- Performs financial calculations using numpy-financial library
- Tools: future value, loan payments, NPV, IRR, interest rate calculations
- Handles rate normalization (percentage vs decimal) automatically
- Compliance wrapper available (`agent_2_compliance.py`) for tool usage verification

**Agent 3: Risk and Tax Advisor** (`examples/agent_3.py`)
- Multi-agent workflow orchestrating specialized sub-agents
- Risk Analyst: Investment risk evaluation (1-5 scale)
- Tax Advisor: French tax implications (PEA, assurance-vie, compte-titres)
- Portfolio Optimizer: Asset allocation recommendations

**Agent 4: Option Pricing** (`examples/agent_4.py`)
- Calculates European option prices using QuantLib Black-Scholes model
- Computes Greeks: Delta, Gamma, Theta, Vega, Rho
- Requires QuantLib installation (`pip install -e ".[quant]"`)
- Compliance wrapper available (`agent_4_compliance.py`)

**Agent 5: SWIFT/ISO 20022 Processing**
Three specialized variants for financial message processing:

- **Convert** (`examples/agent_5.py`): Bidirectional conversion between SWIFT MT103 and ISO 20022 pacs.008 formats
- **Validate** (`examples/agent_5_validator.py`): Message structure, format, and field validation
- **Risk Assessment** (`examples/agent_5_risk.py`): AML/KYC risk scoring with PEP/sanctions checking

**Agent 6: Judge Agent** (`examples/judge_agent.py`)
- Critical evaluation using Llama 70B model (via LLM Pro Finance endpoint)
- Reviews agent outputs for correctness, completeness, and quality
- Provides improvement suggestions and tool usage analysis
- Falls back to 8B model if LLM Pro Finance key unavailable

### Tool Calling

All agents use PydanticAI's tool calling mechanism with explicit tool definitions. Tools are Python functions wrapped with `Tool()` decorator, providing:
- Type-safe parameter validation
- Automatic schema generation
- Error handling and retry logic

Key tool libraries:
- **numpy-financial**: Financial calculations (FV, PV, PMT, NPV, IRR)
- **QuantLib**: Option pricing and Greeks calculation
- **Custom SWIFT/ISO 20022 parsers**: Message format conversion and validation

Tool calling is enforced via system prompts that explicitly require tool usage before responding. Agents track tool call metadata including count, names, and execution results.

### Gradio Interface

The Gradio application (`app/gradio_app.py`) provides a web-based interface with:

- **Tabbed interface**: One tab per agent with dedicated input/output areas
- **Endpoint selection**: Dynamic endpoint switching (Koyeb, HuggingFace, Ollama, LLM Pro Finance)
- **Health monitoring**: Real-time endpoint status checks with wake-up capability for sleeping services
- **Result storage**: Persistent storage of agent outputs with metadata
- **Tool usage tracking**: Display of tool calls, execution time, and endpoint used
- **Metadata display**: Endpoint tracking, fallback detection, and performance metrics

Features:
- Automatic endpoint health checking
- Koyeb service wake-up for sleeping instances
- Endpoint-specific error handling
- Result history and export capabilities

### Endpoints

The system supports four endpoint types, all using OpenAI-compatible APIs:

**Koyeb** (default, recommended)
- Backend: vLLM with Flash Attention
- Model: `DragonLLM/Qwen-Open-Finance-R-8B`
- URL: `https://dragon-llm-dealexmachina-673cae4f.koyeb.app`
- Features: Tool calling enabled, auto-wake for sleeping services
- API path: `/v1`

**HuggingFace Spaces**
- Backend: Text Generation Inference (TGI)
- Model: `dragon-llm-open-finance`
- URL: `https://jeanbaptdzd-open-finance-llm-8b.hf.space`
- Features: Persistent availability, tool calling support
- API path: `/v1`

**Ollama** (local)
- Backend: Local Ollama server
- Model: Configurable via `OLLAMA_MODEL` environment variable
- URL: `http://localhost:11434`
- Features: Local quantized models, full tool calling support
- API path: `/v1`
- Setup: Requires Ollama installation and model import

**LLM Pro Finance**
- Backend: Custom API
- Model: `DragonLLM/llama3.1-70b-fin-v1.0-fp8` (70B parameters)
- URL: `https://demo.llmprofinance.com`
- Features: Larger model for Judge Agent, requires API key
- API path: `/api`
- Limitations: Tool calling not yet supported (coming soon)

Endpoint selection is dynamic per-agent. The system automatically detects endpoint availability and can fall back to default endpoints if agent recreation fails. Metadata tracks actual endpoint used vs requested endpoint.

## Installation

```bash
pip install -e ".[dev]"
```

For option pricing (Agent 4):
```bash
pip install -e ".[dev,quant]"
```

## Configuration

Create `.env` file:

```env
ENDPOINT=koyeb
API_KEY=not-needed
MAX_TOKENS=1500

# Optional: Llama 70B for Judge Agent
LLM_PRO_FINANCE_KEY=your-api-key-here
LLM_PRO_FINANCE_URL=https://api.llm-pro-finance.com

# Optional: Ollama local endpoint
OLLAMA_MODEL=dragon-llm
```

## Usage

### Gradio Interface

```bash
python app/gradio_app.py
```

Access at `http://localhost:7860`. Select endpoint and agent tab, enter prompt, view results with tool usage metadata.

### Programmatic API

```python
from examples.agent_1 import agent_1, Portfolio
from examples.agent_2 import agent_2

# Structured extraction
result = await agent_1.run("Portfolio: 50 AIR.PA at 120€", output_type=Portfolio)
portfolio = result.output

# Financial calculations
result = await agent_2.run("50000€ at 4% for 10 years. Future value?")
```

### Endpoint Selection

```python
from app.models import get_model_for_endpoint
from pydantic_ai import Agent

# Create agent with specific endpoint
model = get_model_for_endpoint("ollama")
agent = Agent(model, system_prompt="...", tools=[...])
```

## Evaluation

Run comprehensive evaluation suite:

```bash
python examples/evaluate_all_agents.py
```

Results saved to `examples/evaluate_all_agents_results.json` with:
- Token usage (input/output/total)
- Tool call verification
- Correctness validation
- Inference speed metrics
- Endpoint metadata

## Model Deployment

Requires deployment of `DragonLLM/Qwen-Open-Finance-R-8B` on an OpenAI-compatible endpoint. See [simple-llm-pro-finance](https://github.com/DealExMachina/simple-llm-pro-finance) for deployment instructions.

### Ollama Setup

1. Install from [ollama.ai](https://ollama.ai)
2. Import model:
   ```bash
   # Create Modelfile
   cat > Modelfile << EOF
   FROM /path/to/model.gguf
   PARAMETER temperature 0.7
   PARAMETER num_ctx 8192
   EOF
   
   ollama create dragon-llm -f Modelfile
   ```
3. Configure `OLLAMA_MODEL=dragon-llm` in `.env`
4. Verify: `curl http://localhost:11434/v1/models`

## Best Practices

- Use `max_output_tokens` (600-1500) to stay within context limits
- Calculate totals client-side (model arithmetic unreliable)
- Use structured outputs (Pydantic) for validation
- Implement explicit tool calling instructions in system prompts
- Verify tool usage in compliance-critical applications
- Monitor endpoint metadata for fallback detection

## Technical Specifications

- **Model**: DragonLLM/Qwen-Open-Finance-R-8B (8B parameters)
- **Context Window**: 8192 tokens (base), up to 128K with YaRN
- **Framework**: PydanticAI 1.18.0+
- **Tool Calling**: OpenAI-compatible function calling
- **Output Format**: Pydantic schemas with automatic validation
- **Language**: French financial terminology optimized

## References

- **Qwen Model**: Qwen Team. "Qwen2.5: A Party of Foundation Models" arXiv:2511.08621 (2024). [arXiv:2511.08621](https://arxiv.org/abs/2511.08621)
- **PydanticAI**: Pydantic AI Framework. [https://ai.pydantic.dev/](https://ai.pydantic.dev/)
- **numpy-financial**: NumPy Financial Functions. [https://numpy.org/numpy-financial/](https://numpy.org/numpy-financial/)
- **QuantLib**: QuantLib - Quantitative Finance Library. [https://www.quantlib.org/](https://www.quantlib.org/)
- **SWIFT Standards**: SWIFT MT Message Standards. [https://www.swift.com/](https://www.swift.com/)
- **ISO 20022**: ISO 20022 Financial Services Messaging. [https://www.iso20022.org/](https://www.iso20022.org/)
- **vLLM**: vLLM: Easy, Fast, and Cheap LLM Serving. [https://github.com/vllm-project/vllm](https://github.com/vllm-project/vllm)
- **Ollama**: Ollama - Local LLM Runtime. [https://ollama.ai/](https://ollama.ai/)
