"""
Agent 3: Workflow multi-étapes avec agents spécialisés

Cet agent démontre la création d'un workflow où plusieurs agents spécialisés
collaborent pour résoudre un problème financier complexe.

Améliorations:
- Utilisation de structured outputs (output_type)
- Outils financiers pour calculs précis
- Validation compliance des tool calls
- Gestion d'erreurs robuste
- Pas de troncature des sorties
"""

import asyncio
from typing import List
from pydantic import BaseModel, Field
from pydantic_ai import Agent, ModelSettings, Tool
import numpy_financial as npf

from app.models import finance_model


# ============================================================================
# MODÈLES STRUCTURÉS
# ============================================================================

class AnalyseRisque(BaseModel):
    """Analyse de risque structurée."""
    niveau_risque: int = Field(description="Niveau de risque de 1 à 5", ge=1, le=5)
    facteurs_risque: list[str] = Field(description="Liste des facteurs de risque identifiés")
    recommandation: str = Field(description="Recommandation basée sur le niveau de risque")
    justification: str = Field(description="Justification détaillée du niveau de risque")


class AnalyseFiscale(BaseModel):
    """Analyse fiscale structurée."""
    regime_fiscal: str = Field(description="Régime fiscal applicable (PEA, assurance-vie, etc.)")
    implications: list[str] = Field(description="Liste des implications fiscales")
    avantages: list[str] = Field(description="Avantages fiscaux identifiés")
    inconvenients: list[str] = Field(description="Inconvénients fiscaux identifiés")
    recommandation: str = Field(description="Recommandation fiscale")


# ============================================================================
# OUTILS FINANCIERS
# ============================================================================

def calculer_valeur_future_investissement(
    capital_initial: float,
    taux_annuel: float,
    duree_annees: float
) -> str:
    """Calcule la valeur future d'un investissement avec intérêts composés.
    
    Utilisez cet outil pour calculer la valeur future d'un investissement.
    
    Args:
        capital_initial: Montant initial en euros
        taux_annuel: Taux d'intérêt annuel (ex: 0.05 pour 5%)
        duree_annees: Durée en années
    
    Returns:
        Valeur future calculée avec détails
    """
    valeur_future = npf.fv(
        rate=taux_annuel,
        nper=duree_annees,
        pmt=0,
        pv=-capital_initial
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


def calculer_rendement_portfolio(
    allocation_actions: float,
    allocation_obligations: float,
    allocation_immobilier: float,
    allocation_autres: float,
    rendement_actions: float = 0.07,
    rendement_obligations: float = 0.03,
    rendement_immobilier: float = 0.05,
    rendement_autres: float = 0.10
) -> str:
    """Calcule le rendement attendu d'un portfolio basé sur l'allocation.
    
    Utilisez cet outil pour calculer le rendement attendu d'un portfolio.
    
    Args:
        allocation_actions: Pourcentage en actions (ex: 0.40 pour 40%)
        allocation_obligations: Pourcentage en obligations
        allocation_immobilier: Pourcentage en immobilier
        allocation_autres: Pourcentage en autres actifs
        rendement_actions: Rendement attendu actions (défaut: 7%)
        rendement_obligations: Rendement attendu obligations (défaut: 3%)
        rendement_immobilier: Rendement attendu immobilier (défaut: 5%)
        rendement_autres: Rendement attendu autres (défaut: 10%)
    
    Returns:
        Rendement attendu du portfolio avec détails
    """
    total_allocation = allocation_actions + allocation_obligations + allocation_immobilier + allocation_autres
    if abs(total_allocation - 1.0) > 0.01:
        return f"Erreur: L'allocation totale doit être 100% (actuel: {total_allocation*100:.1f}%)"
    
    rendement_portfolio = (
        allocation_actions * rendement_actions +
        allocation_obligations * rendement_obligations +
        allocation_immobilier * rendement_immobilier +
        allocation_autres * rendement_autres
    )
    
    return (
        f"Rendement attendu du portfolio: {rendement_portfolio*100:.2f}%\n"
        f"Allocation:\n"
        f"  - Actions: {allocation_actions*100:.1f}% (rendement: {rendement_actions*100:.1f}%)\n"
        f"  - Obligations: {allocation_obligations*100:.1f}% (rendement: {rendement_obligations*100:.1f}%)\n"
        f"  - Immobilier: {allocation_immobilier*100:.1f}% (rendement: {rendement_immobilier*100:.1f}%)\n"
        f"  - Autres: {allocation_autres*100:.1f}% (rendement: {rendement_autres*100:.1f}%)"
    )


# ============================================================================
# AGENTS SPÉCIALISÉS
# ============================================================================

risk_analyst = Agent(
    finance_model,
    model_settings=ModelSettings(max_output_tokens=1200),
    system_prompt=(
        "Vous êtes un analyste de risque financier. "
        "Vous évaluez les risques associés à différents instruments financiers "
        "et stratégies d'investissement.\n\n"
        "FORMAT DE SORTIE OBLIGATOIRE - JSON STRICT:\n"
        "Vous DEVEZ répondre UNIQUEMENT avec un objet JSON valide correspondant exactement à ce schéma:\n"
        "{\n"
        '  "niveau_risque": <entier entre 1 et 5>,\n'
        '  "facteurs_risque": ["facteur1", "facteur2", ...],\n'
        '  "recommandation": "<texte de recommandation>",\n'
        '  "justification": "<texte de justification détaillée>"\n'
        "}\n\n"
        "EXEMPLE CORRECT:\n"
        "{\n"
        '  "niveau_risque": 3,\n'
        '  "facteurs_risque": ["Volatilité élevée des actions", "Concentration en cryptomonnaies", "Manque de diversification"],\n'
        '  "recommandation": "Réduire l\'exposition aux cryptomonnaies et diversifier davantage le portfolio",\n'
        '  "justification": "Le portfolio présente un niveau de risque modéré-élevé (3/5) en raison de la volatilité des actions (40%) et de l\'exposition significative aux cryptomonnaies (10%), actifs très volatils. La diversification est limitée avec seulement 4 classes d\'actifs."\n'
        "}\n\n"
        "RÈGLES CRITIQUES:\n"
        "1. Répondez UNIQUEMENT avec du JSON valide, rien d'autre\n"
        "2. niveau_risque doit être un ENTIER entre 1 et 5 (pas de décimales)\n"
        "3. facteurs_risque doit être un TABLEAU de chaînes (au moins 2 éléments)\n"
        "4. recommandation et justification doivent être des CHAÎNES non vides\n"
        "5. Utilisez les outils disponibles pour calculer les rendements attendus avant d'analyser\n"
        "6. Analysez les facteurs de risque de manière structurée\n"
        "7. Fournissez des recommandations claires et justifiées\n\n"
        "NIVEAUX DE RISQUE:\n"
        "1 = Très faible (obligations d'État, épargne)\n"
        "2 = Faible (obligations corporate, immobilier locatif)\n"
        "3 = Modéré (actions diversifiées, ETF)\n"
        "4 = Élevé (actions individuelles, cryptomonnaies)\n"
        "5 = Très élevé (dérivés, leverage, cryptomonnaies volatiles)"
    ),
    tools=[
        Tool(
            calculer_rendement_portfolio,
            name="calculer_rendement_portfolio",
            description="OBLIGATOIRE pour calculer le rendement attendu d'un portfolio. Utilisez cet outil pour analyser les rendements basés sur l'allocation d'actifs.",
        ),
    ],
    output_type=AnalyseRisque,  # Utilisation du structured output
)

tax_advisor = Agent(
    finance_model,
    model_settings=ModelSettings(max_output_tokens=1500),
    system_prompt=(
        "Vous êtes un conseiller fiscal français. "
        "Vous expliquez les implications fiscales des investissements "
        "selon la réglementation française (PEA, assurance-vie, compte-titres, etc.).\n\n"
        "FORMAT DE SORTIE OBLIGATOIRE - JSON STRICT:\n"
        "Vous DEVEZ répondre UNIQUEMENT avec un objet JSON valide correspondant exactement à ce schéma:\n"
        "{\n"
        '  "regime_fiscal": "<nom du régime>",\n'
        '  "implications": ["implication1", "implication2", ...],\n'
        '  "avantages": ["avantage1", "avantage2", ...],\n'
        '  "inconvenients": ["inconvénient1", "inconvénient2", ...],\n'
        '  "recommandation": "<texte de recommandation>"\n'
        "}\n\n"
        "EXEMPLE CORRECT - Portfolio mixte:\n"
        "{\n"
        '  "regime_fiscal": "Mixte (PEA + Compte-titres + Assurance-vie)",\n'
        '  "implications": ["PEA: Exonération après 5 ans", "Compte-titres: PFU 30% ou barème progressif", "Assurance-vie: Abattement après 8 ans"],\n'
        '  "avantages": ["PEA: Pas d\'impôt sur les plus-values après 5 ans", "Assurance-vie: Transmission avantageuse", "Diversification fiscale"],\n'
        '  "inconvenients": ["Plafond PEA: 150k€ par personne", "Compte-titres: Fiscalité immédiate", "Complexité de gestion multiple"],\n'
        '  "recommandation": "Privilégier le PEA pour les actions (jusqu\'à 150k€), utiliser l\'assurance-vie pour la diversification et la transmission, et limiter le compte-titres aux montants dépassant les plafonds."\n'
        "}\n\n"
        "EXEMPLE CORRECT - PEA uniquement:\n"
        "{\n"
        '  "regime_fiscal": "PEA (Plan d\'Épargne en Actions)",\n'
        '  "implications": ["Exonération totale après 5 ans de détention", "Prélèvements sociaux: 17.2% avant 5 ans", "Plafond: 150k€ par personne"],\n'
        '  "avantages": ["Exonération complète après 5 ans", "Pas de déclaration annuelle", "Fiscalité avantageuse"],\n'
        '  "inconvenients": ["Plafond limité à 150k€", "Restriction aux actions européennes", "Fermeture du compte en cas de retrait avant 5 ans"],\n'
        '  "recommandation": "Le PEA est optimal pour un investissement actions à long terme. Respecter le plafond de 150k€ et la durée minimale de 5 ans pour bénéficier de l\'exonération."\n'
        "}\n\n"
        "RÈGLES CRITIQUES:\n"
        "1. Répondez UNIQUEMENT avec du JSON valide, rien d'autre\n"
        "2. regime_fiscal doit être une CHAÎNE non vide (ex: 'PEA', 'Assurance-vie', 'Compte-titres', 'Mixte')\n"
        "3. implications, avantages, inconvenients doivent être des TABLEAUX de chaînes (au moins 1 élément chacun)\n"
        "4. recommandation doit être une CHAÎNE non vide\n"
        "5. Mentionnez toujours le régime fiscal applicable\n"
        "6. Listez les avantages et inconvénients fiscaux de manière exhaustive\n"
        "7. Fournissez des recommandations pratiques et actionnables\n\n"
        "RÉGIMES FISCAUX FRANÇAIS:\n"
        "- PEA: Plan d'Épargne en Actions (exonération après 5 ans, plafond 150k€)\n"
        "- Assurance-vie: Abattement après 8 ans, transmission avantageuse\n"
        "- Compte-titres: PFU 30% ou barème progressif, fiscalité immédiate\n"
        "- SCPI: Revenus fonciers, ISF/IFI selon le cas\n"
        "- Cryptomonnaies: Plus-values imposables, déclaration obligatoire"
    ),
    output_type=AnalyseFiscale,  # Utilisation du structured output
)

portfolio_optimizer = Agent(
    finance_model,
    model_settings=ModelSettings(max_output_tokens=2000),
    system_prompt=(
        "Vous êtes un optimiseur de portfolio. "
        "Vous proposez des allocations d'actifs optimisées "
        "en fonction des objectifs, de l'horizon temporel et du profil de risque.\n\n"
        "RÈGLES:\n"
        "1. Utilisez les outils pour calculer les rendements attendus\n"
        "2. Tenez compte des analyses de risque et fiscales fournies\n"
        "3. Proposez des allocations concrètes avec justifications\n"
        "Répondez toujours en français."
    ),
    tools=[
        Tool(
            calculer_rendement_portfolio,
            name="calculer_rendement_portfolio",
            description="OBLIGATOIRE pour calculer le rendement attendu d'un portfolio. Utilisez cet outil pour comparer différentes allocations.",
        ),
        Tool(
            calculer_valeur_future_investissement,
            name="calculer_valeur_future_investissement",
            description="OBLIGATOIRE pour calculer la valeur future d'un investissement. Utilisez cet outil pour projeter les résultats à long terme.",
        ),
    ],
)


# ============================================================================
# VALIDATION COMPLIANCE
# ============================================================================

def extract_tool_calls(result) -> List[str]:
    """Extrait les appels d'outils d'un résultat d'agent."""
    tool_calls: List[str] = []
    for msg in result.all_messages():
        msg_calls = getattr(msg, "tool_calls", None) or []
        for call in msg_calls:
            name = None
            args = None
            if hasattr(call, "function"):
                func = call.function
                name = getattr(func, "name", None)
                args = getattr(func, "arguments", None)
            elif hasattr(call, "tool_name"):
                name = call.tool_name
                args = getattr(call, "args", None)
            if name is None and hasattr(call, "name"):
                name = call.name
            if name is None:
                continue

            normalized_args = args
            if normalized_args is not None and not isinstance(normalized_args, str):
                normalized_args = str(normalized_args)

            tool_calls.append(f"{name}: {normalized_args}")
    return tool_calls


compliance_checker = Agent(
    finance_model,
    model_settings=ModelSettings(max_output_tokens=600),
    system_prompt=(
        "Tu es un contrôleur compliance pour workflows multi-agents.\n"
        "On te fournit: l'étape du workflow, la question, la réponse, et les appels d'outils.\n"
        "Règles:\n"
        "1. Si l'agent devait utiliser des outils mais qu'aucun n'a été appelé → Non conforme\n"
        "2. Si les outils ont été utilisés correctement → Conforme\n"
        "3. Si la réponse mentionne des calculs non vérifiés par outils → Flag potential issue\n"
        "Réponds en français, format court: 'Conforme' ou 'Non conforme' + justification."
    ),
)


async def check_compliance(step_name: str, question: str, result, expected_tools: bool = True) -> str:
    """Vérifie la compliance d'une étape du workflow."""
    tool_calls = extract_tool_calls(result)
    
    compliance_input = (
        f"ÉTAPE: {step_name}\n"
        f"QUESTION: {question}\n\n"
        f"RÉPONSE: {result.output}\n\n"
        f"APPELS D'OUTILS: {chr(10).join(tool_calls) if tool_calls else 'Aucun'}\n\n"
        f"OUTILS ATTENDUS: {'Oui' if expected_tools else 'Non'}"
    )
    
    compliance = await compliance_checker.run(compliance_input)
    return compliance.output, tool_calls


# ============================================================================
# WORKFLOW MULTI-ÉTAPES
# ============================================================================

async def workflow_analyse_investissement():
    """Workflow multi-étapes pour analyser un investissement."""
    print("\n🔄 Agent 3: Workflow multi-étapes (amélioré)")
    print("=" * 70)
    
    scenario = """
    Un investisseur de 35 ans avec un profil modéré souhaite investir 100 000€.
    Objectif: Préparer la retraite dans 30 ans.
    Il envisage:
    - 40% en actions françaises (CAC 40)
    - 30% en obligations d'État
    - 20% en immobiler via SCPI
    - 10% en cryptomonnaies
    
    Analysez ce portfolio du point de vue:
    1. Risque
    2. Fiscalité
    3. Optimisation
    """
    
    print("Scénario:\n", scenario, "\n")
    
    try:
        # Étape 1: Analyse de risque (avec outils)
        print("📊 Étape 1: Analyse de risque...")
        risk_result = await risk_analyst.run(
            f"Analyse le niveau de risque (1-5) de cette stratégie:\n{scenario}\n\n"
            "Fournis: niveau de risque (1-5), facteurs de risque principaux, et recommandation. "
            "Utilise les outils pour calculer les rendements attendus."
        )
        
        # Utilisation du structured output
        if hasattr(risk_result, 'data') and risk_result.data:
            risk_analysis = risk_result.data
            print(f"  ✅ Analyse structurée:")
            print(f"     Niveau de risque: {risk_analysis.niveau_risque}/5")
            print(f"     Facteurs: {', '.join(risk_analysis.facteurs_risque[:3])}...")
            print(f"     Recommandation: {risk_analysis.recommandation[:100]}...")
        else:
            print(f"  Analyse:\n  {risk_result.output}\n")
        
        # Compliance check
        compliance_risk, tool_calls_risk = await check_compliance(
            "Analyse de risque",
            scenario,
            risk_result,
            expected_tools=True
        )
        print(f"  🔍 Compliance: {compliance_risk}")
        if tool_calls_risk:
            print(f"  🔧 Outils utilisés: {len(tool_calls_risk)}")
            for tc in tool_calls_risk[:2]:  # Afficher les 2 premiers
                print(f"     - {tc[:80]}...")
        print()
        
        # Étape 2: Conseil fiscal (sans outils requis)
        print("💰 Étape 2: Analyse fiscale...")
        tax_result = await tax_advisor.run(
            f"Quelles sont les implications fiscales de cette stratégie d'investissement "
            f"en France?\n{scenario}"
        )
        
        # Utilisation du structured output
        if hasattr(tax_result, 'data') and tax_result.data:
            tax_analysis = tax_result.data
            print(f"  ✅ Analyse structurée:")
            print(f"     Régime fiscal: {tax_analysis.regime_fiscal}")
            print(f"     Avantages: {len(tax_analysis.avantages)} identifiés")
            print(f"     Inconvénients: {len(tax_analysis.inconvenients)} identifiés")
        else:
            print(f"  Conseil fiscal:\n  {tax_result.output}\n")
        
        # Compliance check
        compliance_tax, tool_calls_tax = await check_compliance(
            "Analyse fiscale",
            scenario,
            tax_result,
            expected_tools=False
        )
        print(f"  🔍 Compliance: {compliance_tax}\n")
        
        # Étape 3: Optimisation avec contexte complet (avec outils)
        print("🎯 Étape 3: Optimisation du portfolio...")
        
        # Préparer le contexte complet (sans troncature)
        risk_context = risk_result.output if not hasattr(risk_result, 'data') else str(risk_result.data)
        tax_context = tax_result.output if not hasattr(tax_result, 'data') else str(tax_result.data)
        
        optimization_result = await portfolio_optimizer.run(
            f"""
            Scénario: {scenario}
            
            Analyses précédentes:
            - Analyse de risque: {risk_context}
            - Analyse fiscale: {tax_context}
            
            Propose une allocation optimisée en tenant compte de ces analyses.
            Utilise les outils pour calculer et comparer les rendements attendus.
            """
        )
        print(f"  Recommandation d'optimisation:\n  {optimization_result.output}\n")
        
        # Compliance check
        compliance_opt, tool_calls_opt = await check_compliance(
            "Optimisation portfolio",
            scenario,
            optimization_result,
            expected_tools=True
        )
        print(f"  🔍 Compliance: {compliance_opt}")
        if tool_calls_opt:
            print(f"  🔧 Outils utilisés: {len(tool_calls_opt)}")
            for tc in tool_calls_opt[:2]:
                print(f"     - {tc[:80]}...")
        print()
        
        # Résumé final
        print("✅ Workflow terminé avec succès!")
        print(f"  - Analyse de risque: Complétée (outils: {len(tool_calls_risk)})")
        print(f"  - Conseils fiscaux: Fournis")
        print(f"  - Optimisation: Recommandation générée (outils: {len(tool_calls_opt)})")
        
    except Exception as e:
        print(f"❌ Erreur dans le workflow: {e}")
        raise


async def exemple_agent_simple():
    """Exemple simplifié d'un agent qui fait tout en une étape."""
    print("\n🚀 Agent 3 (Variante): Agent tout-en-un")
    print("=" * 70)
    
    multi_agent = Agent(
        finance_model,
        model_settings=ModelSettings(max_output_tokens=2000),
        system_prompt=(
            "Vous êtes un conseiller financier complet. "
            "Pour chaque demande d'analyse, fournissez:\n"
            "1. Une évaluation du risque (1-5)\n"
            "2. Les implications fiscales en France\n"
            "3. Une recommandation d'optimisation\n"
            "Répondez toujours en français de manière structurée."
        ),
        tools=[
            Tool(
                calculer_rendement_portfolio,
                name="calculer_rendement_portfolio",
                description="Calcule le rendement attendu d'un portfolio basé sur l'allocation.",
            ),
        ],
    )
    
    question = (
        "J'ai 50 000€ à investir avec un horizon de 15 ans. "
        "Je pense à 60% actions, 30% obligations, 10% immobilier. "
        "Analysez cette stratégie."
    )
    
    try:
        result = await multi_agent.run(question)
        print(f"Question: {question}\n")
        print(f"Analyse complète:\n{result.output}\n")
        
        # Compliance check
        compliance, tool_calls = await check_compliance(
            "Agent tout-en-un",
            question,
            result,
            expected_tools=True
        )
        print(f"🔍 Compliance: {compliance}")
        if tool_calls:
            print(f"🔧 Outils utilisés: {len(tool_calls)}")
    except Exception as e:
        print(f"❌ Erreur: {e}")


# ============================================================================
# TEST DE VALIDATION DES TOOL CALLS
# ============================================================================

async def test_tool_calling():
    """Test pour valider que les agents appellent bien les outils."""
    print("\n🧪 Test: Validation des tool calls")
    print("=" * 70)
    
    test_question = (
        "J'ai 100 000€ à investir avec 40% actions, 30% obligations, "
        "20% immobilier, 10% autres. Calculez le rendement attendu."
    )
    
    print(f"Question test: {test_question}\n")
    
    result = await portfolio_optimizer.run(test_question)
    tool_calls = extract_tool_calls(result)
    
    print(f"✅ Résultat obtenu")
    print(f"📊 Tool calls détectés: {len(tool_calls)}")
    
    if tool_calls:
        print("✅ SUCCÈS: Les outils ont été appelés")
        for i, tc in enumerate(tool_calls, 1):
            print(f"   {i}. {tc[:100]}...")
    else:
        print("❌ ÉCHEC: Aucun outil n'a été appelé")
        print(f"   Réponse: {result.output[:200]}...")
    
    # Compliance check
    compliance, _ = await check_compliance(
        "Test tool calling",
        test_question,
        result,
        expected_tools=True
    )
    print(f"\n🔍 Compliance: {compliance}")
    print("=" * 70)


if __name__ == "__main__":
    # À lancer avec: python -m examples.agent_3_multi_step
    print("Exécution du workflow multi-étapes...")
    asyncio.run(workflow_analyse_investissement())
    
    print("\n\n" + "=" * 70)
    asyncio.run(exemple_agent_simple())
    
    print("\n\n" + "=" * 70)
    asyncio.run(test_tool_calling())
