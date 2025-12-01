# Open Finance PydanticAI

Research project studying tool calling and structured outputs with small language models (8B parameters).

**Note**: Simplified examples for research purposes, not production-ready.

## Research Objective

Investigates whether 8B models can reliably:
1. Trigger tools for financial calculations
2. Generate structured outputs using Pydantic schemas

**Key Finding**: Small models (8B) are viable when tool calling is explicitly enforced, structured outputs are used, and client-side verification is performed.

## Overview

Financial AI agents using PydanticAI with `DragonLLM/Qwen-Open-Finance-R-8B` (8B parameters, French financial terminology). Uses numpy-financial and QuantLib for calculations.

## Agents

This repository contains comprehensive financial AI agents, each demonstrating different capabilities:

### Agent 1: Structured Data Extraction
**File**: `examples/agent_1.py`

Extracts structured portfolio data from unstructured text. Uses Pydantic schemas to validate and structure the output.

**Capabilities**:
- Parses portfolio positions from natural language
- Extracts symbol, quantity, purchase price, and date
- Calculates total portfolio value
- Returns structured `Portfolio` object with validated data

**Example**: Extracts positions from "50 AIR.PA à 120€, 30 SAN.PA à 85€"

### Agent 2: Financial Calculations
**File**: `examples/agent_2.py`

Performs financial calculations using numpy-financial library. Enforces tool usage for all calculations.

**Capabilities**:
- Future value calculations
- Monthly payment calculations
- Present value calculations
- Interest rate calculations
- Portfolio performance analysis

**Tools**: `calculer_valeur_future`, `calculer_versement_mensuel`, `calculer_valeur_actuelle`, `calculer_taux_interet`, `calculer_performance_portfolio`

**Compliance Version**: `examples/agent_2_compliance.py` - Adds compliance checking to verify tool usage

### Agent 3: Multi-Step Workflow
**File**: `examples/agent_3.py`

Orchestrates multiple specialized agents for comprehensive financial analysis.

**Sub-agents**:
- **Risk Analyst** (`risk_analyst`): Evaluates investment risk (1-5 scale) with structured JSON output
- **Tax Advisor** (`tax_advisor`): Analyzes French tax implications (PEA, assurance-vie, compte-titres)
- **Portfolio Optimizer** (`portfolio_optimizer`): Suggests optimized asset allocations

**Workflow**: Risk analysis → Tax analysis → Portfolio optimization recommendations

### Agent 4: Option Pricing
**File**: `examples/agent_4.py`

Calculates option prices and Greeks using QuantLib library. Enforces QuantLib tool usage for all pricing calculations.

**Capabilities**:
- European call option pricing via Black-Scholes
- Calculates Delta, Gamma, Vega, Theta (Greeks)
- Uses QuantLib for validated financial calculations
- Returns structured `OptionPricingResult` with all metrics

**Compliance Version**: `examples/agent_4_compliance.py` - Verifies QuantLib tool usage for compliance

### Agent 5: SWIFT/ISO 20022 Message Processing
**Files**: 
- `examples/agent_5.py` - Main converter
- `examples/agent_5_validator.py` - Message validation
- `examples/agent_5_risk.py` - Risk assessment
- `examples/agent_5_swift_iso20022.py` - Alternative implementation

**Capabilities**:
- **Conversion**: Bidirectional SWIFT MT103 ↔ ISO 20022 pacs.008 conversion
- **Parsing**: Parses SWIFT MT and ISO 20022 XML messages
- **Validation**: Validates message structure, format, and field requirements
- **Risk Assessment**: AML/KYC risk scoring for financial messages

**Tools**: `parser_swift_mt`, `parser_iso20022`, `convertir_swift_vers_iso20022`, `convertir_iso20022_vers_swift`, `valider_swift_message`, `valider_iso20022_message`, `evaluer_risque_message`

### Agent with Mitigation Strategies
**File**: `examples/agent_with_mitigation.py`

Demonstrates mitigation strategies for unreliable model outputs:
- JSON validation
- Tool call detection
- Retry mechanisms
- Safe agent wrappers

## Installation

```bash
pip install -e ".[dev]"
```

Create `.env`:
```env
ENDPOINT=koyeb
API_KEY=not-needed
MAX_TOKENS=1500
```

## Usage

```python
from examples.agent_1 import agent_1, Portfolio
from examples.agent_2 import agent_2
from examples.agent_5 import agent_5

# Structured extraction
result = await agent_1.run("Portfolio: 50 AIR.PA at 120€", output_type=Portfolio)

# Financial calculations
result = await agent_2.run("50000€ at 4% for 10 years. Future value?")

# SWIFT conversion
result = await agent_5.run("Convert SWIFT MT103 to ISO 20022: ...")
```

## Testing and Evaluation

### Comprehensive Agent Evaluation

Run the full evaluation suite to test all agents:

```bash
python examples/evaluate_all_agents.py
```

This script evaluates all agents with strict correctness checks:
- **Agent 1**: Structured data extraction validation
- **Agent 2**: Financial calculation accuracy
- **Agent 2 Compliance**: Tool usage verification
- **Agent 3**: Multi-step workflow validation
- **Agent 4**: Option pricing accuracy
- **Agent 4 Compliance**: QuantLib tool verification
- **Agent 5**: SWIFT/ISO 20022 conversion
- **Agent 5 Validator**: Message validation
- **Agent 5 Risk**: Risk assessment

### Evaluation Results

Results are saved to `examples/evaluate_all_agents_results.json` with:
- Token usage (input/output/total)
- Tool call verification
- Correctness validation
- Inference speed (tokens/second)
- Detailed error messages
- Suggested improvements

**Latest Results Summary** (from `evaluate_all_agents_results.json`):
- **Total Agents Tested**: 8
- **Agents Using Tools**: 7/8 (87%)
- **Correct Results**: 8/8 (100%)
- **Average Tokens**: ~3,810 per request
- **Average Speed**: ~203 tokens/sec

### Agent 5 Synthetic Test Suite

For detailed Agent 5 testing:

```bash
python examples/test_agent_5_synthetic.py
```

Tests 10 synthetic cases for SWIFT/ISO 20022 conversion, validation, and risk assessment.

Results saved to:
- `examples/evaluate_all_agents_results.json` - Full agent outputs and metrics
- `examples/test_agent_5_results.json` - Agent 5 detailed test results

## Performance

**Synthetic Test Suite (10 cases per agent):**
- Total: 50 tests, 100% success rate
- Tool calling: 100% success
- Structured outputs: 100% validation

| Agent | Avg Tokens | Avg Time (s) |
|-------|------------|--------------|
| Agent 1 | 707 | 4.49 |
| Agent 2 | 4,121 | 15.29 |
| Agent 3 | 3,132 | 16.90 |
| Agent 4 | 3,053 | 22.50 |
| Agent 5 | 9,195 | 79.11 |

## Model Deployment

**Important**: This repository requires a running model instance. The model must be deployed on a GPU provider before using the agents.

### Deployment Options

The model instance should be deployed on one of the following platforms:

1. **Koyeb** (recommended) - vLLM backend
2. **Hugging Face Spaces** - TGI (Text Generation Inference)
3. **Other GPU providers** - Any OpenAI-compatible API endpoint

### Using simple-llm-pro-finance for Deployment

The **[simple-llm-pro-finance](https://github.com/DealExMachina/simple-llm-pro-finance)** repository is designed specifically for deploying the model instance. Use it to:

- Deploy the `DragonLLM/Qwen-Open-Finance-R-8B` model on Koyeb or HuggingFace
- Set up the OpenAI-compatible API endpoint
- Configure the model with proper tool calling support

After deployment, configure the endpoint in this repository's `.env` file:

```env
ENDPOINT=koyeb  # or "hf" for HuggingFace
BASE_URL=https://your-deployed-model-url
API_KEY=your-api-key-if-needed
```

### Model Specifications

- **Model**: `DragonLLM/Qwen-Open-Finance-R-8B`
- **Context**: 8192 tokens (requires careful management)
- **Tool Calling**: Requires explicit configuration for vLLM
- **Language**: French financial terminology optimized

## Best Practices

- Use `max_output_tokens` (600-1500) to stay within context limits
- Calculate totals client-side (model arithmetic unreliable)
- Use structured outputs (Pydantic) for validation
- Implement explicit tool calling instructions in prompts

## Documentation

- `docs/model_capabilities_8b.md` - Model capabilities
- `docs/qwen3_specifications.md` - Model specifications
- `docs/swift_iso20022_tools_evaluation.md` - SWIFT/ISO 20022 tools

## References

- **Model**: DragonLLM/Qwen-Open-Finance-R-8B - [arXiv:2511.08621](https://arxiv.org/abs/2511.08621)
- **PydanticAI**: [https://ai.pydantic.dev/](https://ai.pydantic.dev/)
- **numpy-financial**: [https://numpy.org/numpy-financial/](https://numpy.org/numpy-financial/)
- **QuantLib**: [https://www.quantlib.org/](https://www.quantlib.org/)
