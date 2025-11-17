"""
Agent 2 (Improved): Agent avec outils financiers utilisant numpy-financial

Cet agent utilise numpy-financial pour des calculs financiers précis et testés.
Alternative: QuantLib-Python pour des calculs encore plus avancés.

Recommandations de bibliothèques:
1. numpy-financial (recommandé pour ce cas) - Simple, bien testé, suffisant pour la plupart des calculs
2. QuantLib-Python - Plus complet mais plus complexe, idéal pour produits dérivés, options, etc.
3. pandas - Excellent pour analyses de séries temporelles et portfolios
"""

import asyncio
from typing import Annotated
from pydantic import BaseModel
from pydantic_ai import Agent, ModelSettings
import numpy as np
import numpy_financial as npf

from app.models import finance_model


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
        capital_initial: Montant initial en euros (valeur négative pour fv)
        taux_annuel: Taux d'intérêt annuel (ex: 0.05 pour 5%)
        duree_annees: Durée en années
    
    Returns:
        Valeur future calculée avec détails
    """
    # npf.fv(rate, nper, pmt, pv)
    # rate: taux par période
    # nper: nombre de périodes
    # pmt: paiement par période (0 pour investissement unique)
    # pv: valeur présente (négative car sortie de fonds)
    valeur_future = npf.fv(
        rate=taux_annuel,
        nper=duree_annees,
        pmt=0,
        pv=-capital_initial  # Négatif car c'est une sortie
    )
    
    interets = valeur_future - capital_initial
    rendement_pct = (interets / capital_initial) * 100
    
    return (
        f"Valeur future: {valeur_future:,.2f}€\n"
        f"Intérêts générés: {interets:,.2f}€ ({rendement_pct:.2f}%)\n"
        f"Capital initial: {capital_initial:,.2f}€\n"
        f"Taux annuel: {taux_annuel*100:.2f}%\n"
        f"Durée: {duree_annees} ans"
    )


def calculer_versement_mensuel(
    capital_emprunte: float,
    taux_annuel: float,
    duree_mois: int
) -> str:
    """Calcule le versement mensuel pour un prêt.
    
    Utilise numpy-financial.pmt() pour un calcul précis.
    
    Args:
        capital_emprunte: Montant emprunté en euros
        taux_annuel: Taux d'intérêt annuel (ex: 0.04 pour 4%)
        duree_mois: Durée du prêt en mois
    
    Returns:
        Versement mensuel calculé avec détails
    """
    taux_mensuel = taux_annuel / 12
    
    # npf.pmt(rate, nper, pv)
    # rate: taux par période (mensuel)
    # nper: nombre de périodes (mois)
    # pv: valeur présente (montant emprunté, positif car entrée)
    versement = -npf.pmt(
        rate=taux_mensuel,
        nper=duree_mois,
        pv=capital_emprunte
    )  # Négatif car c'est une sortie, on inverse le signe
    
    total_rembourse = versement * duree_mois
    cout_total = total_rembourse - capital_emprunte
    
    # Calcul du tableau d'amortissement (première et dernière échéance)
    # Première échéance: principal = versement - intérêts
    interets_premiere = capital_emprunte * taux_mensuel
    principal_premiere = versement - interets_premiere
    
    return (
        f"Versement mensuel: {versement:,.2f}€\n"
        f"Capital emprunté: {capital_emprunte:,.2f}€\n"
        f"Total remboursé: {total_rembourse:,.2f}€\n"
        f"Coût total du crédit: {cout_total:,.2f}€\n"
        f"Taux mensuel: {taux_mensuel*100:.4f}%\n"
        f"Durée: {duree_mois} mois ({duree_mois/12:.1f} ans)\n"
        f"1ère échéance: {principal_premiere:,.2f}€ principal, {interets_premiere:,.2f}€ intérêts"
    )


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
    
    return (
        f"Performance portfolio:\n"
        f"  Gain absolu: {gain_absolu:+,.2f}€ ({gain_pourcentage:+.2f}%)\n"
        f"  Valeur initiale: {valeur_initiale:,.2f}€\n"
        f"  Valeur actuelle: {valeur_actuelle:,.2f}€\n"
        f"  Rendement annualisé: {rendement_annuelise:+.2f}%\n"
        f"  Rendement mensuel moyen: {rendement_mensuel:+.2f}%\n"
        f"  Durée: {duree_jours} jours ({duree_jours/365:.2f} ans)"
    )


def calculer_valeur_actuelle(
    valeur_future: float,
    taux_annuel: float,
    duree_annees: float
) -> str:
    """Calcule la valeur actuelle (actualisation).
    
    Utilise numpy-financial.pv() pour un calcul précis.
    
    Args:
        valeur_future: Valeur future en euros
        taux_annuel: Taux d'actualisation annuel (ex: 0.05 pour 5%)
        duree_annees: Durée en années
    
    Returns:
        Valeur actuelle calculée
    """
    # npf.pv(rate, nper, pmt, fv)
    valeur_actuelle = -npf.pv(
        rate=taux_annuel,
        nper=duree_annees,
        pmt=0,
        fv=-valeur_future  # Négatif car entrée future
    )
    
    actualisation = valeur_future - valeur_actuelle
    
    return (
        f"Valeur actuelle: {valeur_actuelle:,.2f}€\n"
        f"Valeur future: {valeur_future:,.2f}€\n"
        f"Actualisation: {actualisation:,.2f}€\n"
        f"Taux d'actualisation: {taux_annuel*100:.2f}%\n"
        f"Durée: {duree_annees} ans"
    )


def calculer_taux_interet(
    capital_initial: float,
    valeur_future: float,
    duree_annees: float
) -> str:
    """Calcule le taux d'intérêt nécessaire pour atteindre un objectif.
    
    Utilise numpy-financial.rate() pour un calcul précis.
    
    Args:
        capital_initial: Montant initial en euros
        valeur_future: Valeur future souhaitée en euros
        duree_annees: Durée en années
    
    Returns:
        Taux d'intérêt calculé
    """
    # npf.rate(nper, pmt, pv, fv)
    taux = npf.rate(
        nper=duree_annees,
        pmt=0,
        pv=-capital_initial,
        fv=valeur_future
    )
    
    return (
        f"Taux d'intérêt requis: {taux*100:.4f}% par an\n"
        f"Capital initial: {capital_initial:,.2f}€\n"
        f"Valeur future souhaitée: {valeur_future:,.2f}€\n"
        f"Durée: {duree_annees} ans"
    )


# Agent avec outils améliorés
finance_calculator_agent = Agent(
    finance_model,
    model_settings=ModelSettings(max_output_tokens=2000),
    system_prompt=(
        "Vous êtes un conseiller financier expert avec accès à des outils de calcul financier précis.\n\n"
        "RÈGLES CRITIQUES:\n"
        "1. VOUS DEVEZ TOUJOURS utiliser les outils disponibles pour TOUS les calculs financiers\n"
        "2. NE CALCULEZ JAMAIS manuellement - utilisez TOUJOURS les outils\n"
        "3. Pour calculer une valeur future → utilisez calculer_valeur_future\n"
        "4. Pour calculer un versement mensuel → utilisez calculer_versement_mensuel\n"
        "5. Pour calculer une valeur actuelle → utilisez calculer_valeur_actuelle\n"
        "6. Pour calculer un taux requis → utilisez calculer_taux_interet\n"
        "7. Pour analyser une performance → utilisez calculer_performance_portfolio\n\n"
        "N'expliquez pas comment calculer - UTILISEZ LES OUTILS directement.\n"
        "Répondez en français de manière claire et structurée après avoir utilisé les outils."
    ),
    tools=[
        calculer_valeur_future,
        calculer_versement_mensuel,
        calculer_performance_portfolio,
        calculer_valeur_actuelle,
        calculer_taux_interet,
    ],
)


async def exemple_agent_avec_outils():
    """Exemple d'utilisation d'un agent avec outils améliorés."""
    print("\n🔧 Agent 2 (Improved): Agent avec outils financiers (numpy-financial)")
    print("=" * 60)
    
    question = (
        "J'ai un capital de 50 000€ que je veux placer à 4% par an pendant 10 ans. "
        "Combien aurai-je à la fin ? Et si j'emprunte 200 000€ sur 20 ans à 3.5% "
        "pour acheter un appartement, combien paierai-je par mois ?"
    )
    
    print(f"Question:\n{question}\n")
    
    result = await finance_calculator_agent.run(question)
    
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
    result1 = await finance_calculator_agent.run(question1)
    print(f"Question: {question1}")
    print(f"Réponse: {result1.output[:300]}...")
    
    # Exemple 2: Taux requis
    print("\n2. Calcul de taux requis:")
    question2 = "J'ai 25 000€ aujourd'hui et je veux avoir 50 000€ dans 8 ans. Quel taux d'intérêt me faut-il?"
    result2 = await finance_calculator_agent.run(question2)
    print(f"Question: {question2}")
    print(f"Réponse: {result2.output[:300]}...")


if __name__ == "__main__":
    asyncio.run(exemple_agent_avec_outils())
    asyncio.run(exemple_calculs_avances())

