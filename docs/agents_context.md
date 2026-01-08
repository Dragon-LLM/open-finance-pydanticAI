# Open Finance PydanticAI - Agents Documentation

This document provides complete context about all financial agents available in the system.

## Overview

The Open Finance PydanticAI system consists of 6 specialized agents for financial analysis and processing. Each agent is designed to handle specific financial tasks using structured outputs and validated calculations.

---

## Agent 1: Portfolio Extraction

### Description
Extracts structured portfolio data from unstructured financial text. Identifies stock symbols, quantities, purchase prices, and dates from natural language descriptions.

### Capabilities
- Parses unstructured text describing portfolio holdings
- Extracts stock symbols (e.g., AIR.PA, SAN.PA, TTE.PA)
- Identifies quantities, purchase prices, and dates
- Calculates total portfolio value
- Validates date formats (YYYY-MM-DD)

### Input Format
Natural language text describing portfolio holdings. Example:
```
Mon portfolio actuel :
- J'ai acheté 50 actions Airbus (AIR.PA) à 120€ le 15 mars 2024
- 30 actions Sanofi (SAN.PA) à 85€ le 20 février 2024  
- 100 actions TotalEnergies (TTE.PA) à 55€ le 10 janvier 2024

Date d'évaluation : 1er novembre 2024
```

### Output Format
Structured `Portfolio` object containing:
- `positions`: List of `PositionBoursiere` objects (symbol, quantity, purchase_price, purchase_date)
- `valeur_totale`: Total portfolio value in euros (calculated as sum of quantity × purchase_price)
- `date_evaluation`: Evaluation date

### Example Output
```json
{
  "positions": [
    {"symbole": "AIR.PA", "quantite": 50, "prix_achat": 120.0, "date_achat": "2024-03-15"},
    {"symbole": "SAN.PA", "quantite": 30, "prix_achat": 85.0, "date_achat": "2024-02-20"},
    {"symbole": "TTE.PA", "quantite": 100, "prix_achat": 55.0, "date_achat": "2024-01-10"}
  ],
  "valeur_totale": 14050.0,
  "date_evaluation": "2024-11-01"
}
```

### Performance Characteristics
- Typical tokens: 200-400 input, 300-600 output
- Processing time: 2-5 seconds
- Accuracy: High for well-structured input text

---

## Agent 2: Financial Calculations

### Description
Performs precise financial calculations using numpy-financial library. Supports various financial computations including future value, loan payments, NPV, IRR, and more.

### Capabilities
- Future value calculations (compound interest)
- Loan payment calculations (monthly payments)
- Net Present Value (NPV)
- Internal Rate of Return (IRR)
- Interest rate calculations
- All calculations use validated numpy-financial functions

### Input Format
Natural language questions about financial calculations. Example:
```
J'ai un capital de 50 000€ que je veux placer à 4% par an pendant 10 ans. 
Combien aurai-je à la fin ? Et si j'emprunte 200 000€ sur 20 ans à 3.5% 
pour acheter un appartement, combien paierai-je par mois ?
```

### Output Format
Structured `FinancialCalculationResult` object containing:
- `calculation_type`: Type of calculation performed (e.g., 'future_value', 'loan_payment')
- `result`: The calculated result value
- `input_parameters`: Dictionary of input parameters used
- `explanation`: Brief explanation of the calculation and result

### Tools Used
- `calculer_valeur_future`: Future value with compound interest
- `calculer_versement_mensuel`: Monthly loan payment
- `calculer_npv`: Net Present Value
- `calculer_irr`: Internal Rate of Return
- `calculer_taux_interet`: Interest rate calculation

### Example Output
```json
{
  "calculation_type": "future_value",
  "result": 74012.21,
  "input_parameters": {
    "capital_initial": 50000,
    "taux_annuel": 0.04,
    "duree_annees": 10
  },
  "explanation": "Avec un capital de 50,000€ placé à 4% par an pendant 10 ans, vous aurez 74,012.21€ à la fin."
}
```

### Performance Characteristics
- Typical tokens: 150-300 input, 200-400 output
- Processing time: 3-6 seconds (includes tool calls)
- Accuracy: Very high (uses validated financial libraries)

---

## Agent 3: Risk Analysis & Tax Advice

### Description
Multi-agent workflow providing risk assessment and tax optimization recommendations. Consists of two specialized sub-agents: risk analyst and tax advisor.

### Capabilities

#### Risk Analyst
- Evaluates investment risk levels (1-5 scale)
- Identifies risk factors
- Provides risk-based recommendations
- Calculates expected portfolio returns

#### Tax Advisor
- Analyzes tax implications for French tax regimes
- Recommends optimal tax structures (PEA, assurance-vie, compte-titres)
- Explains tax advantages and disadvantages
- Provides actionable tax optimization advice

### Input Format
Portfolio allocation description. Example:
```
40% actions, 30% obligations, 20% immobilier, 10% autres. 
Investissement 100k€, 30 ans.
```

### Output Format

#### Risk Analysis
Structured `AnalyseRisque` object:
- `niveau_risque`: Integer 1-5 (1=very low, 5=very high)
- `facteurs_risque`: List of identified risk factors
- `recommandation`: Risk-based recommendation
- `justification`: Detailed justification

#### Tax Analysis
Structured `AnalyseFiscale` object:
- `regime_fiscal`: Tax regime name (PEA, assurance-vie, etc.)
- `implications`: List of tax implications
- `avantages`: List of tax advantages
- `inconvenients`: List of tax disadvantages
- `recommandation`: Tax optimization recommendation

### Tools Used
- `calculer_rendement_portfolio`: Calculates expected portfolio returns based on asset allocation

### Example Output
```json
{
  "risk_analysis": {
    "niveau_risque": 3,
    "facteurs_risque": ["Volatilité élevée des actions", "Diversification modérée"],
    "recommandation": "Diversifier davantage le portfolio",
    "justification": "Portfolio modéré-élevé avec 40% actions..."
  },
  "tax_analysis": {
    "regime_fiscal": "Mixte (PEA + Assurance-vie)",
    "implications": ["PEA: Exonération après 5 ans", "Assurance-vie: Abattement après 8 ans"],
    "avantages": ["Optimisation fiscale", "Transmission avantageuse"],
    "inconvenients": ["Plafonds limités", "Complexité de gestion"],
    "recommandation": "Privilégier PEA pour actions, assurance-vie pour diversification"
  }
}
```

### Performance Characteristics
- Typical tokens: 200-400 input, 800-1500 output (both agents)
- Processing time: 5-10 seconds (two sequential agent calls)
- Accuracy: High for standard French tax scenarios

---

## Agent 4: Option Pricing

### Description
Calculates option prices using QuantLib with Black-Scholes model. Provides option Greeks (delta, gamma, vega, theta) for risk management.

### Capabilities
- European call/put option pricing
- Black-Scholes model implementation
- Greeks calculation (delta, gamma, vega, theta)
- Uses QuantLib for validated calculations

### Input Format
Option parameters. Example:
```
Price a call option: spot=100€, strike=105€, maturity=0.5 years, 
rate=0.03, volatility=0.2
```

### Output Format
Structured `OptionPricingResult` object:
- `option_price`: Calculated option price
- `delta`: Price sensitivity to underlying (Δ)
- `gamma`: Delta sensitivity (Γ)
- `vega`: Volatility sensitivity (ν)
- `theta`: Time decay (Θ)
- `input_parameters`: Dictionary of input parameters
- `calculation_method`: Method used (e.g., 'Black-Scholes')
- `greeks_explanations`: Explanations of each Greek

### Tools Used
- `calculer_prix_call_black_scholes`: Black-Scholes call option pricing

### Example Output
```json
{
  "option_price": 3.2456,
  "delta": 0.4234,
  "gamma": 0.0123,
  "vega": 18.5678,
  "theta": -0.1234,
  "input_parameters": {
    "spot": 100.0,
    "strike": 105.0,
    "maturite_annees": 0.5,
    "taux_sans_risque": 0.03,
    "volatilite": 0.2
  },
  "calculation_method": "Black-Scholes",
  "greeks_explanations": {
    "delta": "Sensibilité du prix de l'option à une variation de 1€ du prix du sous-jacent",
    "gamma": "Sensibilité du delta à une variation du prix du sous-jacent (convexité)",
    "vega": "Sensibilité du prix de l'option à une variation de 1% de la volatilité",
    "theta": "Décroissance du prix de l'option par jour (décroissance temporelle)"
  }
}
```

### Performance Characteristics
- Typical tokens: 100-200 input, 400-800 output
- Processing time: 2-4 seconds
- Accuracy: Very high (uses QuantLib validated models)

---

## Agent 5: SWIFT/ISO20022 Converter

### Description
Converts between SWIFT MT messages and ISO 20022 XML format. Supports bidirectional conversion for common payment messages.

### Capabilities
- SWIFT MT103 → ISO 20022 pacs.008 (Customer Credit Transfer)
- ISO 20022 pacs.008 → SWIFT MT103
- SWIFT MT940 → ISO 20022 camt.053 (Bank Statement)
- ISO 20022 camt.053 → SWIFT MT940
- Message parsing and validation
- Field mapping between formats

### Input Format
SWIFT MT message or ISO 20022 XML. Example SWIFT MT103:
```
{1:F01BANKFRPPAXXX1234567890}{2:O1031200240101BANKFRPPAXXX123456789012345678901234567890}{3:{108:REF123456789}}{4:
:20:REF123456789
:32A:240101EUR1000.00
:50K:/FR1420041010050500013M02606
COMPAGNIE ABC
:59:/DE89370400440532013000
COMPAGNIE XYZ
:70:PAYMENT FOR INVOICE 12345
-}
```

### Output Format
Structured message objects:
- `SwiftMTMessage`: Parsed SWIFT MT structure
- `ISO20022Message`: Parsed ISO 20022 structure

### Tools Used
- `parser_swift_mt`: Parses SWIFT MT messages
- `parser_iso20022`: Parses ISO 20022 XML
- `convertir_swift_vers_iso20022`: Converts SWIFT to ISO 20022
- `convertir_iso20022_vers_swift`: Converts ISO 20022 to SWIFT

### Example Output
```json
{
  "message_type": "pacs.008",
  "fields": {
    "MsgId": "MSG20240101120000",
    "Amt": "1000.00",
    "Ccy": "EUR",
    "Dbtr": "COMPAGNIE ABC",
    "Cdtr": "COMPAGNIE XYZ"
  },
  "raw_message": "..."
}
```

### Performance Characteristics
- Typical tokens: 500-2000 input, 800-2500 output
- Processing time: 4-8 seconds
- Accuracy: High for standard message formats

---

## Agent 6: Judge Agent

### Description
Critical evaluation agent that reviews outputs from other agents (1-5) and provides comprehensive assessments. Uses a larger 70B model (Llama) for deeper analysis.

### Capabilities
- Correctness assessment (mathematical accuracy)
- Quality evaluation (structure, completeness)
- Tool usage analysis (correct tool calls, duplicates)
- Input validation review
- Response format consistency check
- Compliance verification
- Best practices identification
- Improvement suggestions

### Input Format
JSON output from one or more agents (1-5). The judge agent analyzes these outputs.

### Output Format
Structured `ComprehensiveJudgment` object:
- `overall_assessment`: Overall assessment text
- `agent_reviews`: List of `AgentOutputReview` objects (one per agent)
- `common_issues`: Issues common across multiple agents
- `best_practices_identified`: Best practices observed
- `priority_improvements`: High-priority improvement recommendations
- `overall_score`: Overall score (0.0-1.0)

Each `AgentOutputReview` contains:
- `agent_name`: Name of reviewed agent
- `correctness_score`: 0.0-1.0
- `quality_score`: 0.0-1.0
- `strengths`: List of identified strengths
- `weaknesses`: List of identified weaknesses
- `critical_issues`: Critical issues requiring attention
- `improvement_suggestions`: Specific improvement suggestions

### Evaluation Criteria
1. **Correctness**: Mathematical accuracy, calculation precision
2. **Quality**: Structure, completeness, data coherence
3. **Tool Usage**: Correct tool calls, no unauthorized manual calculations
4. **Input Validation**: Format checks, range validation, required fields
5. **Response Format**: Consistency between agents, standardized structures
6. **Compliance**: Regulatory compliance
7. **Best Practices**: Industry best practices adherence

### Example Output
```json
{
  "overall_assessment": "Overall, the agents perform well with structured outputs. Some arithmetic errors detected in Agent 1.",
  "agent_reviews": [
    {
      "agent_name": "Agent 1",
      "correctness_score": 0.85,
      "quality_score": 0.90,
      "strengths": ["Good structured output", "Proper date formatting"],
      "weaknesses": ["Arithmetic error in total calculation"],
      "critical_issues": ["Total value calculation incorrect"],
      "improvement_suggestions": ["Verify arithmetic calculations", "Use tool for sum calculation"]
    }
  ],
  "common_issues": ["Some agents skip tool validation"],
  "best_practices_identified": ["Structured outputs", "Proper error handling"],
  "priority_improvements": ["Add arithmetic validation", "Enforce tool usage"],
  "overall_score": 0.88
}
```

### Performance Characteristics
- Typical tokens: 1000-5000 input, 2000-4000 output
- Processing time: 8-15 seconds (uses larger 70B model)
- Accuracy: High (comprehensive analysis)

---

## System Architecture

### Model Configuration
- **Agents 1-5**: Use Qwen-Open-Finance-R-8B model (8B parameters)
  - Endpoints: Koyeb (vLLM) or HuggingFace Space
  - Context window: 32K tokens (extendable to 128K with YaRN)
  - Max generation: ~20K tokens

- **Agent 6 (Judge)**: Uses Llama 70B model via LLM Pro Finance API
  - Larger model for comprehensive analysis
  - Requires API key configuration
  - Higher accuracy for complex evaluations

### Tool Integration
All agents use Python tools for validated calculations:
- numpy-financial: Financial calculations
- QuantLib: Option pricing and derivatives
- Custom parsers: SWIFT/ISO20022 message parsing

### Structured Outputs
All agents use Pydantic models for:
- Type validation
- Data consistency
- Automatic serialization
- Error handling

---

## Usage Guidelines

### Best Practices
1. **Be Specific**: Provide clear, detailed input descriptions
2. **Use Defaults**: Start with default examples to understand agent behavior
3. **Validate Outputs**: Always verify calculations, especially for Agent 1
4. **Check Tool Usage**: Ensure agents use tools for calculations (not manual)
5. **Review Judge Feedback**: Use Agent 6 to validate other agents' outputs

### Common Issues
1. **Arithmetic Errors**: Some agents may miscalculate - always verify totals
2. **Tool Skipping**: Agents sometimes skip required tools - check tool calls
3. **Format Inconsistencies**: Date formats, number formats may vary
4. **Token Limits**: Very long inputs may be truncated

### Performance Tips
1. **Batch Processing**: Run multiple agents in parallel when possible
2. **Cache Results**: Store results for judge agent analysis
3. **Monitor Tokens**: Track token usage for cost optimization
4. **Error Handling**: Always check for errors in agent outputs

---

## API Integration

### Endpoint Configuration
- **Koyeb**: Default endpoint (vLLM backend)
- **HuggingFace Space**: Alternative endpoint
- **LLM Pro Finance**: Required for Judge Agent (70B model)

### Authentication
- Most endpoints: No API key required (public)
- LLM Pro Finance: Requires API key for 70B model access

---

## Version Information
- System Version: 0.1.0
- PydanticAI: >=1.18.0
- Last Updated: 2024









