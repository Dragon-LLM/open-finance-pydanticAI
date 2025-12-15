"""
Agent 2: Financial tools with numpy-financial (token-optimized).
"""

import asyncio
from typing import Annotated, Dict, Any
from pydantic import BaseModel, Field
from pydantic_ai import Agent, ModelSettings
import numpy as np
import numpy_financial as npf

from app.models import finance_model


# ============================================================================
# STRUCTURED OUTPUT MODEL
# ============================================================================

class FinancialCalculationResult(BaseModel):
    """Structured result for financial calculations."""
    calculation_type: str = Field(description="Type of calculation performed (e.g., 'future_value', 'loan_payment')")
    result: float = Field(description="The calculated result value")
    input_parameters: Dict[str, Any] = Field(description="Input parameters used for the calculation")
    explanation: str = Field(description="Brief explanation of the calculation and result")


# ============================================================================
# OUTILS FINANCIERS AVEC NUMPY-FINANCIAL
# ============================================================================

def calculer_valeur_future(
    capital_initial: float,
    taux_annuel: float,
    duree_annees: float
) -> str:
    """Calcule la valeur future avec intérêts composés.
    
    Utilise numpy-financial.fv() pour un calcul précis et testé.
    
    Args:
        capital_initial: Montant initial en euros (valeur positive, ex: 50000)
        taux_annuel: Taux d'intérêt annuel (ex: 0.04 pour 4%)
        duree_annees: Durée en années
    
    Returns:
        Valeur future calculée avec détails
    """
    # Normalize input: accept positive value, convert to absolute value
    capital_abs = abs(capital_initial)
    
    # CRITICAL: Normalize rate - if > 1, assume it's a percentage (e.g., 4 means 4%)
    # Models often pass 4 instead of 0.04, causing massive errors
    if taux_annuel > 1.0:
        taux_annuel = taux_annuel / 100.0
    
    # npf.fv(rate, nper, pmt, pv)
    valeur_future = npf.fv(
        rate=taux_annuel,
        nper=duree_annees,
        pmt=0,
        pv=-capital_abs
    )
    
    interets = valeur_future - capital_abs
    rendement_pct = (interets / capital_abs) * 100
    
    return f"FV: {valeur_future:,.2f}€ | Intérêts: {interets:,.2f}€ | Capital: {capital_abs:,.2f}€ | Taux: {taux_annuel*100:.2f}% | {duree_annees}ans"


def calculer_versement_mensuel(
    capital_emprunte: float,
    taux_annuel: float,
    duree_mois: int
) -> str:
    """Calcule le versement mensuel pour un prêt.
    
    Utilise numpy-financial.pmt() pour un calcul précis.
    
    Args:
        capital_emprunte: Montant emprunté en euros (valeur positive, ex: 200000)
        taux_annuel: Taux d'intérêt annuel (ex: 0.035 pour 3.5%)
        duree_mois: Durée du prêt en mois
    
    Returns:
        Versement mensuel calculé avec détails
    """
    # Normalize input: accept positive value
    capital_abs = abs(capital_emprunte)
    
    # CRITICAL: Normalize rate - if > 1, assume it's a percentage (e.g., 4 means 4%)
    if taux_annuel > 1.0:
        taux_annuel = taux_annuel / 100.0
    
    taux_mensuel = taux_annuel / 12
    
    # npf.pmt(rate, nper, pv)
    versement = -npf.pmt(
        rate=taux_mensuel,
        nper=duree_mois,
        pv=capital_abs
    )
    
    total_rembourse = versement * duree_mois
    cout_total = total_rembourse - capital_abs
    
    # Calcul du tableau d'amortissement (première et dernière échéance)
    # Première échéance: principal = versement - intérêts
    interets_premiere = capital_abs * taux_mensuel
    principal_premiere = versement - interets_premiere
    
    return f"Mensualité: {versement:,.2f}€ | Total: {total_rembourse:,.2f}€ | Coût: {cout_total:,.2f}€ | {duree_mois}mois"


def calculer_performance_portfolio(
    valeur_initiale: float,
    valeur_actuelle: float,
    duree_jours: int
) -> str:
    """Calcule la performance d'un portfolio.
    
    Utilise numpy pour des calculs précis de rendement.
    
    Args:
        valeur_initiale: Valeur initiale en euros
        valeur_actuelle: Valeur actuelle en euros
        duree_jours: Durée en jours
    
    Returns:
        Performance calculée avec métriques détaillées
    """
    gain_absolu = valeur_actuelle - valeur_initiale
    gain_pourcentage = (gain_absolu / valeur_initiale) * 100
    
    # Rendement annualisé: (Vf/Vi)^(365/jours) - 1
    rendement_annuelise = ((valeur_actuelle / valeur_initiale) ** (365 / duree_jours) - 1) * 100
    
    # Calcul du rendement mensuel moyen
    duree_mois = duree_jours / 30.44  # Moyenne de jours par mois
    rendement_mensuel = ((valeur_actuelle / valeur_initiale) ** (1 / duree_mois) - 1) * 100
    
    return f"Gain: {gain_absolu:+,.2f}€ ({gain_pourcentage:+.2f}%) | Rdt annualisé: {rendement_annuelise:+.2f}% | {duree_jours}j"


def calculer_valeur_actuelle(
    valeur_future: float,
    taux_annuel: float,
    duree_annees: float
) -> str:
    """Calcule la valeur actuelle (actualisation).
    
    Utilise numpy-financial.pv() pour un calcul précis.
    
    Args:
        valeur_future: Valeur future en euros (valeur positive, ex: 100000)
        taux_annuel: Taux d'actualisation annuel (ex: 0.03 pour 3%)
        duree_annees: Durée en années
    
    Returns:
        Valeur actuelle calculée
    """
    # Normalize input: accept positive value
    valeur_future_abs = abs(valeur_future)
    
    # CRITICAL: Normalize rate - if > 1, assume it's a percentage
    if taux_annuel > 1.0:
        taux_annuel = taux_annuel / 100.0
    
    # npf.pv(rate, nper, pmt, fv)
    valeur_actuelle = -npf.pv(
        rate=taux_annuel,
        nper=duree_annees,
        pmt=0,
        fv=-valeur_future_abs
    )
    
    actualisation = valeur_future_abs - valeur_actuelle
    
    return f"VA: {valeur_actuelle:,.2f}€ | VF: {valeur_future_abs:,.2f}€ | Actualisation: {actualisation:,.2f}€ | {duree_annees}ans"


def calculer_taux_interet(
    capital_initial: float,
    valeur_future: float,
    duree_annees: float
) -> str:
    """Calcule le taux d'intérêt nécessaire pour atteindre un objectif.
    
    Utilise numpy-financial.rate() pour un calcul précis.
    
    Args:
        capital_initial: Montant initial en euros (valeur positive, ex: 25000)
        valeur_future: Valeur future souhaitée en euros (valeur positive, ex: 50000)
        duree_annees: Durée en années
    
    Returns:
        Taux d'intérêt calculé
    """
    # Normalize inputs: accept positive values
    capital_abs = abs(capital_initial)
    valeur_future_abs = abs(valeur_future)
    
    # npf.rate(nper, pmt, pv, fv)
    # nper: nombre de périodes
    # pmt: paiement par période (0)
    # pv: valeur présente (négative car sortie)
    # fv: valeur future (positive car entrée)
    taux = npf.rate(
        nper=duree_annees,
        pmt=0,
        pv=-capital_abs,
        fv=valeur_future_abs
    )
    
    return f"Taux requis: {taux*100:.4f}%/an | Capital: {capital_abs:,.2f}€ | VF: {valeur_future_abs:,.2f}€ | {duree_annees}ans"


# Agent 2: Financial calculations with tools
# Ultra-short tool docstrings to minimize token usage
for fn, doc in {
    calculer_valeur_future: "Valeur future (fv).",
    calculer_versement_mensuel: "Versement mensuel (pmt).",
    calculer_performance_portfolio: "Performance portfolio.",
    calculer_valeur_actuelle: "Valeur actuelle (pv).",
    calculer_taux_interet: "Taux requis (rate).",
}.items():
    fn.__doc__ = doc

agent_2 = Agent(
    finance_model,
    model_settings=ModelSettings(max_output_tokens=400),  # Minimized
    system_prompt=(
        "Conseiller financier.\n"
        "RÈGLE ABSOLUE: Appelle UN outil UNE SEULE FOIS puis STOP.\n"
        "Outils: valeur_future, versement_mensuel, valeur_actuelle, taux_interet.\n"
        "Retourne FinancialCalculationResult immédiatement après l'outil."
    ),
    tools=[
        calculer_valeur_future,
        calculer_versement_mensuel,
        calculer_performance_portfolio,
        calculer_valeur_actuelle,
        calculer_taux_interet,
    ],
    output_type=FinancialCalculationResult,
    retries=0,  # No retries to prevent loops
)


async def exemple_agent_avec_outils():
    """Exemple d'utilisation d'un agent avec outils financiers."""
    print("\n🔧 Agent 2: Financial Calculations with Tools (numpy-financial)")
    print("=" * 60)
    
    # Simple single question to avoid multiple tool calls
    question = "J'ai 50000 euros a placer a 4% par an pendant 10 ans. Quelle sera la valeur finale?"
    
    print(f"Question:\n{question}\n")
    
    result = await agent_2.run(question)
    
    print("✅ Réponse de l'agent avec calculs précis:")
    print(result.output)
    print()
    
    # Vérifier les tool calls
    print("\n" + "=" * 60)
    print("📊 VÉRIFICATION DES TOOL CALLS")
    print("=" * 60)
    
    tool_calls_found = False
    tool_calls_count = 0
    
    # Vérifier dans all_messages()
    if hasattr(result, 'all_messages'):
        try:
            messages = list(result.all_messages())
            for msg in messages:
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    tool_calls_found = True
                    tool_calls_count = len(msg.tool_calls)
                    print(f"✅ {tool_calls_count} tool call(s) détecté(s)!")
                    tools_used = []
                    for i, tc in enumerate(msg.tool_calls, 1):
                        tool_name = None
                        # Try different ways to access tool name
                        if hasattr(tc, 'function'):
                            func = tc.function
                            if hasattr(func, 'name'):
                                tool_name = func.name
                            elif isinstance(func, dict):
                                tool_name = func.get('name', 'unknown')
                        elif hasattr(tc, 'tool_name'):
                            tool_name = tc.tool_name
                        elif hasattr(tc, 'name'):
                            tool_name = tc.name
                        elif isinstance(tc, dict):
                            tool_name = tc.get('tool_name') or tc.get('name') or tc.get('function', {}).get('name', 'unknown')
                        else:
                            tool_name = str(tc)
                        
                        if tool_name and tool_name != 'unknown':
                            tools_used.append(tool_name)
                            print(f"  {i}. Tool: {tool_name}")
                            
                            # Try to get arguments
                            args = {}
                            if hasattr(tc, 'function') and hasattr(tc.function, 'arguments'):
                                args = tc.function.arguments if isinstance(tc.function.arguments, dict) else {}
                            elif hasattr(tc, 'args'):
                                args = tc.args if isinstance(tc.args, dict) else {}
                            elif isinstance(tc, dict):
                                args = tc.get('args', tc.get('arguments', {}))
                            
                            if args:
                                print(f"     Arguments: {args}")
                            
                            # Check for tool result
                            if hasattr(tc, 'result'):
                                result_text = str(tc.result)
                                print(f"     Result: {result_text[:100]}...")
                    
                    if tools_used:
                        print(f"\n📋 Outils utilisés: {', '.join(tools_used)}")
                        # Check for duplicate tool calls
                        from collections import Counter
                        tool_counts = Counter(tools_used)
                        duplicates = {tool: count for tool, count in tool_counts.items() if count > 1}
                        if duplicates:
                            print(f"⚠️  APPELS DUPLIQUÉS DÉTECTÉS:")
                            for tool, count in duplicates.items():
                                print(f"     - {tool}: appelé {count} fois")
                        else:
                            print(f"✅ Aucun appel dupliqué détecté")
                    else:
                        print(f"  [Debug] Tool calls structure: {type(msg.tool_calls[0]) if msg.tool_calls else 'empty'}")
        except Exception as e:
            print(f"  [Debug] Erreur lors de l'inspection: {e}")
    
    if not tool_calls_found:
        print("⚠️  AUCUN TOOL CALL DÉTECTÉ")
        print("   Le modèle mentionne les outils dans sa réponse mais ne les appelle pas réellement.")
        print("   Cela peut être dû au fait que le modèle fine-tuné ne génère pas de tool calls.")
    
    # Afficher les statistiques de tokens
    if hasattr(result, 'usage') and result.usage:
        print(f"\n💾 Tokens utilisés: {result.usage.total_tokens if hasattr(result.usage, 'total_tokens') else 'N/A'}")
    
    print("=" * 60)


async def exemple_calculs_avances():
    """Exemples de calculs plus avancés."""
    print("\n\n📊 Exemples de calculs avancés")
    print("=" * 60)
    
    # Exemple 1: Valeur actuelle
    print("\n1. Calcul de valeur actuelle:")
    question1 = "Quelle est la valeur actuelle de 100 000€ dans 15 ans avec un taux d'actualisation de 3%?"
    result1 = await agent_2.run(question1)
    print(f"Question: {question1}")
    print(f"Réponse: {result1.output[:300]}...")
    
    # Exemple 2: Taux requis
    print("\n2. Calcul de taux requis:")
    question2 = "J'ai 25 000€ aujourd'hui et je veux avoir 50 000€ dans 8 ans. Quel taux d'intérêt me faut-il?"
    result2 = await agent_2.run(question2)
    print(f"Question: {question2}")
    print(f"Réponse: {result2.output[:300]}...")


if __name__ == "__main__":
    asyncio.run(exemple_agent_avec_outils())
    asyncio.run(exemple_calculs_avances())

