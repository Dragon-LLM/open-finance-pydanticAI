"""
Agent 4: Option pricing using QuantLib.

Optimizations:
- Concise system prompt (60 tokens vs 120 tokens, 50% reduction)
- Reduced max_output_tokens (800 vs 1200)
- Structured tool returns (dict instead of formatted strings)
"""

import asyncio
from contextlib import contextmanager
from typing import Dict, Any
from pydantic import BaseModel, Field

from pydantic_ai import Agent, ModelSettings, Tool

try:
    import QuantLib as ql
except ImportError as exc:
    raise RuntimeError("QuantLib-Python est requis pour cet exemple") from exc

from app.models import finance_model


# ============================================================================
# STRUCTURED OUTPUT MODEL
# ============================================================================

class OptionPricingResult(BaseModel):
    """Structured result for option pricing."""
    option_price: float = Field(description="Calculated option price")
    delta: float = Field(description="Delta (price sensitivity to underlying)")
    gamma: float = Field(description="Gamma (delta sensitivity)")
    vega: float = Field(description="Vega (volatility sensitivity)")
    theta: float = Field(description="Theta (time decay)")
    input_parameters: Dict[str, Any] = Field(description="Input parameters (spot, strike, maturity, rate, volatility, dividend)")
    calculation_method: str = Field(description="Method used (e.g., 'Black-Scholes')")
    greeks_explanations: Dict[str, str] = Field(
        description="Brief explanations for each Greek: delta, gamma, vega, theta",
        default_factory=lambda: {
            "delta": "Sensibilité du prix de l'option à une variation de 1€ du prix du sous-jacent",
            "gamma": "Sensibilité du delta à une variation du prix du sous-jacent (convexité)",
            "vega": "Sensibilité du prix de l'option à une variation de 1% de la volatilité",
            "theta": "Décroissance du prix de l'option par jour (décroissance temporelle)"
        }
    )


# ============================================================================
# QUANTLIB TOOLS
# ============================================================================

@contextmanager
def _ql_evaluation_date(evaluation_date: "ql.Date"):
    """Scope QuantLib evaluationDate mutations to avoid leaking global state."""
    settings = ql.Settings.instance()
    previous_date = settings.evaluationDate
    try:
        settings.evaluationDate = evaluation_date
        yield
    finally:
        settings.evaluationDate = previous_date


def calculer_prix_call_black_scholes(
    spot: float,
    strike: float,
    maturite_annees: float,
    taux_sans_risque: float,
    volatilite: float,
    dividende: float = 0.0,
) -> Dict[str, Any]:
    """Prix d'un call européen via Black-Scholes avec QuantLib (OBLIGATOIRE pour tous calculs de pricing).

    Utilise QuantLib pour un calcul précis et validé. Accepte valeurs positives pour tous paramètres.

    Args:
        spot: Prix spot du sous-jacent en devise (doit être > 0, ex: 100.0)
        strike: Prix d'exercice en devise (doit être > 0, ex: 105.0)
        maturite_annees: Temps jusqu'à maturité en années (doit être > 0, ex: 0.5 pour 6 mois)
        taux_sans_risque: Taux sans risque annualisé en décimal (ex: 0.02 pour 2%, 0.05 pour 5%)
        volatilite: Volatilité annualisée en décimal (doit être > 0, ex: 0.25 pour 25%)
        dividende: Rendement de dividende annualisé en décimal (défaut: 0.0, ex: 0.01 pour 1%)

    Returns:
        Dict avec:
        - prix: Prix du call calculé par QuantLib
        - delta: Sensibilité au prix du sous-jacent
        - gamma: Sensibilité du delta
        - vega: Sensibilité à la volatilité
        - theta: Sensibilité au temps
        - spot, strike, maturite_annees: Paramètres d'entrée
        - taux_sans_risque, volatilite: Paramètres en pourcentage
        - details: Résumé formaté avec prix et grecques
    """
    evaluation_date = ql.Date.todaysDate()
    with _ql_evaluation_date(evaluation_date):
        settlement_date = evaluation_date

        calendar = ql.NullCalendar()
        day_count = ql.Actual365Fixed()

        payoff = ql.PlainVanillaPayoff(ql.Option.Call, strike)
        maturity_date = calendar.advance(evaluation_date, ql.Period(round(maturite_annees * 365), ql.Days))
        exercise = ql.EuropeanExercise(maturity_date)

        spot_handle = ql.QuoteHandle(ql.SimpleQuote(spot))
        flat_ts = ql.YieldTermStructureHandle(ql.FlatForward(settlement_date, taux_sans_risque, day_count))
        dividend_ts = ql.YieldTermStructureHandle(ql.FlatForward(settlement_date, dividende, day_count))
        vol_ts = ql.BlackVolTermStructureHandle(ql.BlackConstantVol(settlement_date, calendar, volatilite, day_count))

        process = ql.BlackScholesMertonProcess(spot_handle, dividend_ts, flat_ts, vol_ts)
        option = ql.VanillaOption(payoff, exercise)
        engine = ql.AnalyticEuropeanEngine(process)
        option.setPricingEngine(engine)

        price = option.NPV()
        delta = option.delta()
        gamma = option.gamma()
        vega = option.vega()
        theta = option.theta()

    return {
        "prix": round(price, 4),
        "delta": round(delta, 4),
        "gamma": round(gamma, 6),
        "vega": round(vega, 4),
        "theta": round(theta, 4),
        "spot": spot,
        "strike": strike,
        "maturite_annees": maturite_annees,
        "taux_sans_risque": round(taux_sans_risque * 100, 2),
        "volatilite": round(volatilite * 100, 2),
        "details": f"Prix call: {price:,.4f} | Delta: {delta:.4f} | Gamma: {gamma:.6f} | Vega: {vega:.4f} | Theta: {theta:.4f}"
    }


# ============================================================================
# OPTIMIZED AGENT
# ============================================================================

agent_4 = Agent(
    finance_model,
    model_settings=ModelSettings(max_output_tokens=800),
    system_prompt="""Ingénieur financier spécialisé en pricing d'options avec QuantLib.
RÈGLES ABSOLUES:
1. TOUJOURS utiliser calculer_prix_call_black_scholes pour TOUS les calculs de pricing
2. JAMAIS de calculs manuels - utilisez TOUJOURS l'outil QuantLib
3. Pour un call européen → APPELEZ calculer_prix_call_black_scholes avec spot, strike, maturité, taux, volatilité, dividende
4. Répondez avec un objet OptionPricingResult structuré incluant prix, delta, gamma, vega, theta, paramètres et méthode.
5. Incluez des explications brèves pour chaque grec (delta, gamma, vega, theta) dans le champ greeks_explanations.
N'expliquez pas comment calculer - UTILISEZ L'OUTIL directement.""",
    tools=[
        Tool(
            calculer_prix_call_black_scholes,
            name="calculer_prix_call_black_scholes",
            description="OBLIGATOIRE pour le pricing d'un call européen via Black-Scholes. Fournissez spot, strike, maturité (en années), taux sans risque, volatilité, dividende.",
        )
    ],
    output_type=OptionPricingResult,
)


async def exemple_pricing_call():
    """Exemple de pricing d'un call."""
    question = (
        "Calcule le prix d'un call européen sur l'indice X:\n"
        "- Spot: 100\n"
        "- Strike: 105\n"
        "- Maturité: 0.5 an\n"
        "- Taux sans risque: 0.02\n"
        "- Volatilité: 0.25\n"
        "- Dividende: 0.01"
    )
    
    print("📊 Agent 4: Option Pricing")
    print("=" * 70)
    print(f"Question:\n{question}\n")
    
    import time
    start = time.time()
    result = await agent_4.run(question)
    elapsed = time.time() - start
    
    usage = result.usage()
    
    print("✅ Résultat:\n")
    print(result.output)
    print(f"\n📈 Performance:")
    print(f"  - Temps: {elapsed:.2f}s")
    print(f"  - Tokens: {usage.total_tokens} (input: {usage.input_tokens}, output: {usage.output_tokens})")
    print(f"  - Vitesse: {usage.total_tokens/elapsed:.1f} tokens/sec")
    
    # Check tool calls
    from app.mitigation_strategies import ToolCallDetector
    tool_calls = ToolCallDetector.extract_tool_calls(result) or []
    if tool_calls:
        print(f"\n🔧 Outils utilisés: {len(tool_calls)}")
        for tc in tool_calls:
            print(f"  - {tc.get('name', 'unknown')}")


if __name__ == "__main__":
    asyncio.run(exemple_pricing_call())

