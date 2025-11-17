"""
Agent 2: Agent avec outils (Tools) pour calculs financiers

Cet agent démontre l'utilisation d'outils Python que l'agent peut appeler
pour effectuer des calculs financiers complexes.

Monitoring avec Logfire activé pour tracer les exécutions et tool calls.
"""

import asyncio
from typing import Annotated
from pydantic import BaseModel
from pydantic_ai import Agent, ModelSettings, Tool, InstrumentationSettings
import logfire

from app.models import finance_model
from app.logfire_config import configure_logfire

# Configurer Logfire pour le monitoring
# Projet: open-finance dans l'organisation deal-ex-machina (UE)
# Note: Pour la première utilisation, exécutez: logfire auth
# Le token sera automatiquement associé au projet via l'organisation
# 'if-token-present' = n'envoie que si authentifié, sinon mode local
configure_logfire(send_to_logfire='if-token-present')


# Outils que l'agent peut utiliser
def calculer_valeur_future(
    capital_initial: float,
    taux_annuel: float,
    duree_annees: float
) -> str:
    """Calcule la valeur future avec intérêts composés.
    
    Args:
        capital_initial: Montant initial en euros
        taux_annuel: Taux d'intérêt annuel (ex: 0.05 pour 5%)
        duree_annees: Durée en années
    
    Returns:
        Valeur future calculée
    """
    valeur_future = capital_initial * (1 + taux_annuel) ** duree_annees
    interets = valeur_future - capital_initial
    return (
        f"Valeur future: {valeur_future:,.2f}€\n"
        f"Intérêts générés: {interets:,.2f}€\n"
        f"Capital initial: {capital_initial:,.2f}€"
    )


def calculer_versement_mensuel(
    capital_emprunte: float,
    taux_annuel: float,
    duree_mois: int
) -> str:
    """Calcule le versement mensuel pour un prêt.
    
    Args:
        capital_emprunte: Montant emprunté en euros
        taux_annuel: Taux d'intérêt annuel (ex: 0.04 pour 4%)
        duree_mois: Durée du prêt en mois
    
    Returns:
        Versement mensuel calculé
    """
    taux_mensuel = taux_annuel / 12
    versement = capital_emprunte * (
        taux_mensuel * (1 + taux_mensuel) ** duree_mois
    ) / ((1 + taux_mensuel) ** duree_mois - 1)
    
    total_rembourse = versement * duree_mois
    cout_total = total_rembourse - capital_emprunte
    
    return (
        f"Versement mensuel: {versement:,.2f}€\n"
        f"Total remboursé: {total_rembourse:,.2f}€\n"
        f"Coût total du crédit: {cout_total:,.2f}€"
    )


def calculer_performance_portfolio(
    valeur_initiale: float,
    valeur_actuelle: float,
    duree_jours: int
) -> str:
    """Calcule la performance d'un portfolio.
    
    Args:
        valeur_initiale: Valeur initiale en euros
        valeur_actuelle: Valeur actuelle en euros
        duree_jours: Durée en jours
    
    Returns:
        Performance calculée
    """
    gain_absolu = valeur_actuelle - valeur_initiale
    gain_pourcentage = (gain_absolu / valeur_initiale) * 100
    rendement_annuelise = ((valeur_actuelle / valeur_initiale) ** (365 / duree_jours) - 1) * 100
    
    return (
        f"Gain absolu: {gain_absolu:+,.2f}€ ({gain_pourcentage:+.2f}%)\n"
        f"Rendement annualisé: {rendement_annuelise:+.2f}%\n"
        f"Durée: {duree_jours} jours"
    )


# Agent avec outils et monitoring Logfire
finance_calculator_agent = Agent(
    finance_model,
    model_settings=ModelSettings(max_output_tokens=2000),  # Increased for tool usage explanations
    system_prompt=(
        "Vous êtes un conseiller financier expert avec accès à des outils de calcul financier précis.\n\n"
        "RÈGLES IMPORTANTES:\n"
        "1. TOUJOURS utiliser les outils de calcul disponibles pour TOUS les calculs financiers\n"
        "2. Ne JAMAIS calculer manuellement - utilisez toujours les outils\n"
        "3. Après avoir utilisé un outil, mentionnez explicitement: 'J'ai utilisé l'outil [nom_outil]'\n"
        "4. Présentez les résultats de l'outil dans votre réponse\n"
        "5. Expliquez toujours les résultats dans le contexte de la question du client\n\n"
        "Outils disponibles:\n"
        "- calculer_valeur_future: Pour calculer la valeur future d'un investissement\n"
        "- calculer_versement_mensuel: Pour calculer les mensualités d'un prêt\n"
        "- calculer_performance_portfolio: Pour analyser la performance d'un portfolio\n\n"
        "Répondez toujours en français et indiquez clairement quand vous utilisez un outil."
    ),
    tools=[
        Tool(
            calculer_valeur_future,
            name="calculer_valeur_future",
            description="Calcule la valeur future d'un investissement avec intérêts composés. OBLIGATOIRE pour tous les calculs de valeur future.",
            max_retries=3,
        ),
        Tool(
            calculer_versement_mensuel,
            name="calculer_versement_mensuel",
            description="Calcule le versement mensuel d'un prêt. OBLIGATOIRE pour tous les calculs de prêts.",
            max_retries=3,
        ),
        Tool(
            calculer_performance_portfolio,
            name="calculer_performance_portfolio",
            description="Calcule la performance d'un portfolio d'investissement. OBLIGATOIRE pour toutes les analyses de performance.",
            max_retries=3,
        ),
    ],
    instrument=InstrumentationSettings(),  # Active Logfire monitoring
)


def afficher_statistiques_outils(result):
    """Affiche les statistiques d'utilisation des outils."""
    print("\n" + "=" * 60)
    print("📊 STATISTIQUES D'UTILISATION DES OUTILS")
    print("=" * 60)
    
    # Vérifier les tool calls dans le résultat
    tool_calls_count = 0
    tools_utilises = []
    tool_calls_details = []
    
    # PydanticAI stocke les tool calls dans result.all_messages() -> ModelResponse.tool_calls
    if hasattr(result, 'all_messages'):
        try:
            messages = list(result.all_messages())
            for msg in messages:
                # ModelResponse a un attribut tool_calls
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    for tc in msg.tool_calls:
                        tool_calls_count += 1
                        # Extraire le nom de l'outil depuis tool_calls
                        tool_name = None
                        # Format standard: tc.function.name
                        if hasattr(tc, 'function'):
                            func = tc.function
                            if hasattr(func, 'name'):
                                tool_name = func.name
                            elif isinstance(func, dict):
                                tool_name = func.get('name', 'unknown')
                        # Autres formats possibles
                        elif hasattr(tc, 'tool_name'):
                            tool_name = tc.tool_name
                        elif hasattr(tc, 'name'):
                            tool_name = tc.name
                        elif isinstance(tc, dict):
                            tool_name = tc.get('tool_name') or tc.get('name') or tc.get('function', {}).get('name', 'unknown')
                        else:
                            tool_name = str(tc)
                        
                        tools_utilises.append(tool_name)
                        
                        # Extraire les arguments
                        args = {}
                        if hasattr(tc, 'function') and hasattr(tc.function, 'arguments'):
                            # Format standard: tc.function.arguments
                            args = tc.function.arguments if isinstance(tc.function.arguments, dict) else {}
                        elif hasattr(tc, 'args'):
                            args = tc.args if isinstance(tc.args, dict) else {}
                        elif hasattr(tc, 'arguments'):
                            args = tc.arguments if isinstance(tc.arguments, dict) else {}
                        elif isinstance(tc, dict):
                            args = tc.get('args', tc.get('arguments', {}))
                        
                        tool_calls_details.append({
                            'name': tool_name,
                            'args': args,
                            'result': getattr(tc, 'result', None)
                        })
                
                # Vérifier aussi builtin_tool_calls
                if hasattr(msg, 'builtin_tool_calls') and msg.builtin_tool_calls:
                    for tc in msg.builtin_tool_calls:
                        tool_calls_count += 1
                        tool_name = getattr(tc, 'tool_name', None) or getattr(tc, 'name', None) or str(tc)
                        tools_utilises.append(tool_name)
                        tool_calls_details.append({
                            'name': tool_name,
                            'args': getattr(tc, 'args', {}),
                            'result': getattr(tc, 'result', None)
                        })
        except Exception as e:
            # Si l'accès échoue, essayer une autre méthode
            print(f"  [Debug] Erreur lors de l'inspection: {e}")
    
    # Vérifier dans les attributs directs du résultat
    if hasattr(result, 'tool_calls'):
        tool_calls = result.tool_calls
        if tool_calls:
            tool_calls_count = len(tool_calls)
            for tc in tool_calls:
                tool_name = getattr(tc, 'tool_name', None) or getattr(tc, 'name', None) or str(tc)
                tools_utilises.append(tool_name)
                tool_calls_details.append({
                    'name': tool_name,
                    'args': getattr(tc, 'args', {}),
                    'result': getattr(tc, 'result', None)
                })
    
    # Essayer d'accéder via all_messages_json
    if hasattr(result, 'all_messages_json'):
        try:
            messages_json = result.all_messages_json()
            if isinstance(messages_json, list):
                for msg in messages_json:
                    if isinstance(msg, dict):
                        # Chercher tool_calls dans le message
                        if 'tool_calls' in msg:
                            for tc in msg['tool_calls']:
                                tool_calls_count += 1
                                if isinstance(tc, dict):
                                    tool_name = tc.get('function', {}).get('name', 'unknown')
                                else:
                                    tool_name = str(tc)
                                tools_utilises.append(tool_name)
        except Exception:
            pass
    
    # Vérifier si tool_calls existe mais est vide (simulation)
    tool_calls_exist_but_empty = False
    if hasattr(result, 'all_messages'):
        try:
            messages = list(result.all_messages())
            for msg in messages:
                if hasattr(msg, 'tool_calls'):
                    if msg.tool_calls is not None and len(msg.tool_calls) == 0:
                        tool_calls_exist_but_empty = True
        except Exception:
            pass
    
    # Afficher les résultats
    if tool_calls_count > 0:
        print(f"✅ Outils utilisés: {tool_calls_count} appel(s)")
        print(f"\n📋 Détail des outils appelés:")
        for i, tool_name in enumerate(tools_utilises, 1):
            print(f"  {i}. {tool_name}")
        
        # Compter les occurrences de chaque outil
        from collections import Counter
        compteur = Counter(tools_utilises)
        print(f"\n📈 Répartition:")
        for tool_name, count in compteur.items():
            print(f"  - {tool_name}: {count} fois")
        
        # Afficher les détails si disponibles
        if tool_calls_details:
            print(f"\n🔍 Détails des appels:")
            for i, detail in enumerate(tool_calls_details[:5], 1):  # Limiter à 5 pour la lisibilité
                print(f"  {i}. {detail['name']}")
                if detail.get('args'):
                    args_str = str(detail['args'])[:100]
                    print(f"     Arguments: {args_str}")
    else:
        if tool_calls_exist_but_empty:
            print("⚠️  SIMULATION D'UTILISATION D'OUTILS (pas d'appels réels)")
            print("   Le modèle mentionne les outils dans sa réponse mais ne les appelle pas réellement.")
            print("   Les tool_calls sont présents mais vides [].")
        else:
            print("⚠️  AUCUN OUTIL N'A ÉTÉ UTILISÉ")
            print("   L'agent a effectué les calculs directement sans utiliser les outils disponibles.")
        
        print("\n💡 Analyse:")
        print("   - Le modèle peut simuler l'utilisation des outils dans sa réponse textuelle")
        print("   - Mais ne fait pas d'appels réels aux fonctions Python")
        print("   - Cela peut être dû au fait que le modèle préfère calculer directement")
        print("\n💡 Suggestions pour forcer l'utilisation:")
        print("   - Rendre les calculs plus complexes")
        print("   - Utiliser un prompt plus strict")
        print("   - Vérifier la configuration du modèle (certains modèles ont des limitations)")
    
    # Afficher les statistiques de tokens si disponibles
    print(f"\n💾 Statistiques de tokens:")
    if hasattr(result, 'usage') and result.usage:
        usage = result.usage
        input_tokens = getattr(usage, 'input_tokens', None) or getattr(usage, 'prompt_tokens', None)
        output_tokens = getattr(usage, 'output_tokens', None) or getattr(usage, 'completion_tokens', None)
        total_tokens = getattr(usage, 'total_tokens', None)
        
        print(f"  - Tokens d'entrée: {input_tokens if input_tokens is not None else 'N/A'}")
        print(f"  - Tokens de sortie: {output_tokens if output_tokens is not None else 'N/A'}")
        print(f"  - Total: {total_tokens if total_tokens is not None else (input_tokens + output_tokens if input_tokens and output_tokens else 'N/A')}")
    else:
        print("  - Informations non disponibles")
    
    print("=" * 60)


async def exemple_agent_avec_outils():
    """Exemple d'utilisation d'un agent avec outils et monitoring Logfire."""
    print("\n🔧 Agent 2: Agent avec outils de calcul (Logfire monitoring activé)")
    print("=" * 60)
    
    # Créer un span Logfire pour cette exécution
    with logfire.span('agent_financial_calculation'):
        question = (
            "J'ai un capital de 50 000€ que je veux placer à 4% par an pendant 10 ans. "
            "Combien aurai-je à la fin ? Et si j'emprunte 200 000€ sur 20 ans à 3.5% "
            "pour acheter un appartement, combien paierai-je par mois ?"
        )
        
        print(f"Question:\n{question}\n")
        
        # Logfire trace automatiquement l'exécution de l'agent
        result = await finance_calculator_agent.run(question)
        
        print("✅ Réponse de l'agent avec calculs:")
        print(result.output)
        print()
        
        # Afficher les statistiques détaillées
        afficher_statistiques_outils(result)
        
        # Logger des métriques personnalisées
        logfire.info(
            "Agent execution completed",
            question_length=len(question),
            response_length=len(result.output),
            run_id=result.run_id,
        )


if __name__ == "__main__":
    asyncio.run(exemple_agent_avec_outils())

