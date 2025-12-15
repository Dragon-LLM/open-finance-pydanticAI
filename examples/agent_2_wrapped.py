"""
Agent 2 with wrapper functions that match the model's parameter naming.

The fine-tuned model generates tool calls with:
- capital, taux, duree (short names)

But our functions expect:
- capital_initial, taux_annuel, duree_annees (descriptive names)

This module provides wrappers that bridge the gap.
"""

import asyncio
from typing import Dict, Any
from pydantic import BaseModel, Field
from pydantic_ai import Agent, ModelSettings
import numpy_financial as npf

from app.models import finance_model


# ============================================================================
# STRUCTURED OUTPUT MODEL
# ============================================================================

class FinancialCalculationResult(BaseModel):
    """Result."""
    calculation_type: str = Field(description="Type")
    result: float = Field(description="Result")
    input_parameters: Dict[str, Any] = Field(description="Params")
    explanation: str = Field(description="Info")


# ============================================================================
# WRAPPER FUNCTIONS - Match model's parameter names
# ============================================================================

def calculer_valeur_future(capital: float, taux: float, duree: float) -> str:
    """Calcule valeur future.
    
    Args:
        capital: Montant initial
        taux: Taux (ex: 4 pour 4% ou 0.04)
        duree: Durée en années
    """
    capital_abs = abs(capital)
    
    # Normalize rate: if > 1, it's a percentage
    if taux > 1.0:
        taux = taux / 100.0
    
    valeur_future = npf.fv(rate=taux, nper=duree, pmt=0, pv=-capital_abs)
    interets = valeur_future - capital_abs
    
    return f"FV:{valeur_future:,.2f}€"


def calculer_versement_mensuel(capital: float, taux: float, duree: float) -> str:
    """Calcule versement mensuel.
    
    Args:
        capital: Montant emprunté
        taux: Taux annuel (ex: 3.5 pour 3.5% ou 0.035)
        duree: Durée en années (sera convertie en mois)
    """
    capital_abs = abs(capital)
    
    # Normalize rate
    if taux > 1.0:
        taux = taux / 100.0
    
    # Convert years to months
    duree_mois = int(duree * 12)
    taux_mensuel = taux / 12
    
    versement = -npf.pmt(rate=taux_mensuel, nper=duree_mois, pv=capital_abs)
    total = versement * duree_mois
    cout = total - capital_abs
    
    return f"{versement:,.2f}"


def calculer_valeur_actuelle(capital: float, taux: float, duree: float) -> str:
    """Calcule valeur actuelle.
    
    Args:
        capital: Valeur future souhaitée
        taux: Taux d'actualisation
        duree: Durée en années
    """
    capital_abs = abs(capital)
    
    # Normalize rate
    if taux > 1.0:
        taux = taux / 100.0
    
    valeur_actuelle = -npf.pv(rate=taux, nper=duree, pmt=0, fv=-capital_abs)
    actualisation = capital_abs - valeur_actuelle
    
    return f"{valeur_actuelle:,.2f}"


def calculer_taux_interet(capital_initial: float, valeur_future: float, duree: float) -> str:
    """Calcule taux requis.
    
    Args:
        capital_initial: Capital de départ
        valeur_future: Valeur cible
        duree: Durée en années
    """
    capital_abs = abs(capital_initial)
    vf_abs = abs(valeur_future)
    
    taux = npf.rate(nper=duree, pmt=0, pv=-capital_abs, fv=vf_abs)
    
    return f"{taux*100:.4f}"


def calculer_performance_portfolio(valeur_initiale: float, valeur_actuelle: float, duree_jours: int) -> str:
    """Calcule performance portfolio.
    
    Args:
        valeur_initiale: Valeur initiale
        valeur_actuelle: Valeur actuelle
        duree_jours: Durée en jours
    """
    gain = valeur_actuelle - valeur_initiale
    gain_pct = (gain / valeur_initiale) * 100
    rendement_annualise = ((valeur_actuelle / valeur_initiale) ** (365 / duree_jours) - 1) * 100
    
    return f"{gain_pct:+.2f}"


# Ultra-short docstrings - SINGLE WORD to minimize tokens
for fn, doc in {
    calculer_valeur_future: "FV.",
    calculer_versement_mensuel: "PMT.",
    calculer_performance_portfolio: "Perf.",
    calculer_valeur_actuelle: "PV.",
    calculer_taux_interet: "Rate.",
}.items():
    fn.__doc__ = doc


# ============================================================================
# DYNAMIC TOOL SELECTION FOR MINIMAL CONTEXT
# ============================================================================

def select_tool_from_question(question: str):
    """Select the appropriate tool based on question keywords."""
    question_lower = question.lower()
    
    if any(word in question_lower for word in ['valeur finale', 'combien aurai', 'placer', 'investir']):
        return calculer_valeur_future
    elif any(word in question_lower for word in ['emprunt', 'prêt', 'mensualité', 'crédit']):
        return calculer_versement_mensuel
    elif any(word in question_lower for word in ['performance', 'rendement', 'passé de']):
        return calculer_performance_portfolio
    elif any(word in question_lower for word in ['valeur actuelle', 'investir aujourd']):
        return calculer_valeur_actuelle
    elif any(word in question_lower for word in ['quel taux', 'doubler']):
        return calculer_taux_interet
    else:
        return calculer_valeur_future  # Default


def create_minimal_agent(question: str):
    """Create agent with ONLY the needed tool to minimize context."""
    tool = select_tool_from_question(question)
    
    return Agent(
        finance_model,
        model_settings=ModelSettings(
            max_output_tokens=200,  # Ultra-minimal
            temperature=0.0,
        ),
        system_prompt="Calc. 1x outil. JSON.",  # Absolute minimum
        tools=[tool],
        output_type=FinancialCalculationResult,
        retries=0,
    )


# ============================================================================
# AGENT FUNCTION
# ============================================================================

async def run_agent_2_wrapped(question: str):
    """Run agent with dynamic tool selection."""
    agent = create_minimal_agent(question)
    return await agent.run(question)


# For backward compatibility, keep a default agent
agent_2_wrapped = Agent(
    finance_model,
    model_settings=ModelSettings(max_output_tokens=300, temperature=0.0),
    system_prompt="Calc. 1 outil. FinancialCalculationResult.",
    tools=[calculer_valeur_future],  # Single tool default
    output_type=FinancialCalculationResult,
    retries=0,
)


async def test_wrapped():
    """Test the wrapped agent."""
    print("Testing wrapped agent...")
    question = "J'ai 50000 euros a placer a 4% par an pendant 5 ans. Quelle sera la valeur finale?"
    
    result = await agent_2_wrapped.run(question)
    print(f"Result: {result.output}")
    
    # Check tool calls
    from app.mitigation_strategies import ToolCallDetector
    tool_calls = ToolCallDetector.extract_tool_calls(result)
    print(f"Tool calls: {len(tool_calls)}")
    for tc in tool_calls:
        print(f"  - {tc.get('name')}: {tc.get('args')}")


if __name__ == "__main__":
    asyncio.run(test_wrapped())

