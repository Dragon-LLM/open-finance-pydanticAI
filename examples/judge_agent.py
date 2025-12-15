"""
Judge Agent: Critical evaluation of agent outputs with improvement suggestions.

This agent reviews the outputs from agents 1-5 and provides:
- Critical analysis of correctness
- Quality assessment
- Improvement suggestions
- Best practices recommendations
"""

import asyncio
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from pydantic_ai import Agent, ModelSettings

from app.models import judge_model


# ============================================================================
# JUDGMENT MODELS
# ============================================================================

class AgentOutputReview(BaseModel):
    """Review of a single agent's output."""
    agent_name: str = Field(description="Name of the agent being reviewed")
    correctness_score: float = Field(description="Correctness score (0.0-1.0)", ge=0.0, le=1.0)
    quality_score: float = Field(description="Quality score (0.0-1.0)", ge=0.0, le=1.0)
    strengths: List[str] = Field(description="List of strengths identified")
    weaknesses: List[str] = Field(description="List of weaknesses identified")
    critical_issues: List[str] = Field(description="Critical issues that need attention")
    improvement_suggestions: List[str] = Field(description="Specific improvement suggestions")


class ComprehensiveJudgment(BaseModel):
    """Comprehensive judgment of all agent outputs."""
    overall_assessment: str = Field(description="Overall assessment of all agents")
    agent_reviews: List[AgentOutputReview] = Field(description="Individual reviews for each agent")
    common_issues: List[str] = Field(description="Issues common across multiple agents")
    best_practices_identified: List[str] = Field(description="Best practices observed")
    priority_improvements: List[str] = Field(description="High-priority improvements recommended")
    overall_score: float = Field(description="Overall score across all agents (0.0-1.0)", ge=0.0, le=1.0)


# ============================================================================
# JUDGE AGENT
# ============================================================================

judge_agent = Agent(
    judge_model,
    model_settings=ModelSettings(max_output_tokens=4000, temperature=0.3),
    system_prompt="""Vous êtes un expert en évaluation critique de systèmes d'IA financiers. 
Votre rôle est d'analyser de manière rigoureuse les sorties des agents financiers et de fournir des jugements critiques constructifs.

CRITÈRES D'ÉVALUATION:
1. **Correctness (Exactitude)**: Les résultats sont-ils mathématiquement corrects? Les calculs sont-ils précis? Vérifiez les calculs arithmétiques (ex: sommes, multiplications).
2. **Quality (Qualité)**: La sortie est-elle bien structurée? Les données sont-elles complètes et cohérentes? Le format de réponse est-il standardisé?
3. **Tool Usage (Utilisation d'outils)**: 
   - Les outils ont-ils été utilisés correctement? Y a-t-il des calculs manuels non autorisés?
   - Y a-t-il des appels d'outils dupliqués? Comptez les appels uniques vs total des appels.
   - Les outils requis ont-ils été appelés (ex: calculer_rendement_portfolio pour Agent 3)?
4. **Input Validation (Validation des entrées)**: Les données d'entrée ont-elles été validées? Y a-t-il des vérifications de format, de plage, ou de champs requis?
5. **Response Format Consistency (Cohérence du format)**: Les formats de réponse sont-ils cohérents entre les agents? Les structures de données sont-elles standardisées?
6. **Compliance (Conformité)**: Les règles de conformité sont-elles respectées?
7. **Best Practices (Meilleures pratiques)**: Les meilleures pratiques sont-elles suivies?

STRUCTURE DE VOTRE ÉVALUATION:
- Identifiez les forces de chaque agent
- Identifiez les faiblesses et problèmes critiques
- Fournissez des suggestions d'amélioration concrètes et actionnables
- Identifiez les problèmes communs à plusieurs agents
- Recommandez des améliorations prioritaires

SOYEZ CRITIQUE MAIS CONSTRUCTIF:
- Ne soyez pas complaisant - identifiez les vrais problèmes
- Soyez spécifique dans vos critiques
- Proposez des solutions pratiques
- Priorisez les améliorations par impact

Répondez avec un objet ComprehensiveJudgment structuré.""",
    output_type=ComprehensiveJudgment,
)


# ============================================================================
# EVALUATION FUNCTIONS
# ============================================================================

async def judge_agent_output(
    agent_name: str,
    agent_output: Any,
    expected_result: Optional[Any] = None,
    tool_calls: Optional[List[Dict[str, Any]]] = None,
    input_prompt: Optional[str] = None,
) -> AgentOutputReview:
    """Judge a single agent's output.
    
    Args:
        agent_name: Name of the agent (e.g., "Agent 1", "Agent 2")
        agent_output: The output from the agent
        expected_result: Expected result for correctness comparison
        tool_calls: List of tool calls made by the agent
        input_prompt: Original input prompt to the agent
        
    Returns:
        AgentOutputReview with critical analysis
    """
    # Build evaluation prompt
    evaluation_prompt = f"""Évaluez la sortie de {agent_name}.

SORTIE DE L'AGENT:
{str(agent_output)}

"""
    
    if input_prompt:
        evaluation_prompt += f"PROMPT ORIGINAL:\n{input_prompt}\n\n"
    
    if expected_result:
        evaluation_prompt += f"RÉSULTAT ATTENDU:\n{expected_result}\n\n"
    
    if tool_calls:
        evaluation_prompt += f"APPELS D'OUTILS:\n"
        for tc in tool_calls:
            evaluation_prompt += f"- {tc}\n"
        evaluation_prompt += "\n"
    
    evaluation_prompt += """Analysez cette sortie de manière critique:
1. Vérifiez l'exactitude des calculs et résultats (vérifiez les calculs arithmétiques)
2. Évaluez la qualité de la structure et des données (format standardisé?)
3. Vérifiez l'utilisation correcte des outils (outils requis appelés? appels dupliqués?)
4. Vérifiez la validation des entrées (données validées avant traitement?)
5. Vérifiez la cohérence du format de réponse (structure standardisée?)
6. Identifiez les forces et faiblesses
7. Proposez des améliorations concrètes"""

    result = await judge_agent.run(evaluation_prompt)
    judgment = result.output
    
    # Extract review for this specific agent
    if judgment.agent_reviews:
        for review in judgment.agent_reviews:
            if review.agent_name == agent_name:
                return review
    
    # Fallback: create review from overall judgment
    return AgentOutputReview(
        agent_name=agent_name,
        correctness_score=judgment.overall_score,
        quality_score=judgment.overall_score,
        strengths=judgment.best_practices_identified,
        weaknesses=judgment.common_issues,
        critical_issues=judgment.priority_improvements[:3],
        improvement_suggestions=judgment.priority_improvements,
    )


async def judge_all_agents(
    agent_outputs: Dict[str, Dict[str, Any]],
) -> ComprehensiveJudgment:
    """Judge outputs from all agents (1-5).
    
    Args:
        agent_outputs: Dictionary mapping agent names to their outputs
        Format: {
            "Agent 1": {
                "output": ...,
                "expected_result": ...,
                "tool_calls": [...],
                "input_prompt": "..."
            },
            ...
        }
        
    Returns:
        ComprehensiveJudgment with reviews of all agents
    """
    # Build comprehensive evaluation prompt
    evaluation_prompt = """Évaluez de manière critique les sorties de tous les agents financiers (Agent 1 à Agent 5).

"""
    
    for agent_name, agent_data in agent_outputs.items():
        evaluation_prompt += f"=== {agent_name} ===\n"
        
        if agent_data.get("input_prompt"):
            evaluation_prompt += f"PROMPT: {agent_data['input_prompt']}\n"
        
        evaluation_prompt += f"SORTIE:\n{str(agent_data.get('output', 'N/A'))}\n"
        
        if agent_data.get("expected_result"):
            evaluation_prompt += f"ATTENDU: {agent_data['expected_result']}\n"
        
        if agent_data.get("tool_calls"):
            evaluation_prompt += f"OUTILS UTILISÉS: {', '.join(str(tc) for tc in agent_data['tool_calls'][:3])}\n"
        
        evaluation_prompt += "\n"
    
    evaluation_prompt += """Analysez TOUS ces agents de manière critique et complète:
1. Évaluez chaque agent individuellement:
   - Correctness: Vérifiez les calculs arithmétiques (ex: sommes, multiplications)
   - Quality: Structure, complétude, format standardisé
   - Tool Usage: Outils requis appelés? Appels dupliqués? (comptez appels uniques vs total)
   - Input Validation: Données validées avant traitement?
   - Response Format: Format cohérent et standardisé?
2. Identifiez les problèmes communs à plusieurs agents (ex: appels dupliqués, absence de validation, formats non standardisés)
3. Identifiez les meilleures pratiques observées
4. Recommandez des améliorations prioritaires pour chaque agent
5. Fournissez un score global et une évaluation d'ensemble

Soyez rigoureux, critique mais constructif. Identifiez les vrais problèmes et proposez des solutions pratiques."""

    result = await judge_agent.run(evaluation_prompt)
    return result.output


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

async def judge_evaluation_results(evaluation_results_file: str = "examples/evaluate_all_agents_results.json"):
    """Judge agent outputs from evaluation results file.
    
    Args:
        evaluation_results_file: Path to JSON file with evaluation results
        
    Returns:
        ComprehensiveJudgment of all agents
    """
    import json
    from pathlib import Path
    
    results_path = Path(evaluation_results_file)
    if not results_path.exists():
        raise FileNotFoundError(f"Evaluation results file not found: {evaluation_results_file}")
    
    with open(results_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Build agent outputs dictionary from evaluation results
    agent_outputs = {}
    for agent_data in data.get("agents", []):
        agent_name = agent_data.get("name", "Unknown Agent")
        agent_outputs[agent_name] = {
            "output": agent_data.get("output_text", ""),
            "expected_result": agent_data.get("expected_result"),
            "actual_result": agent_data.get("actual_result"),
            "input_prompt": agent_data.get("input_prompt", ""),
            "tool_calls": agent_data.get("tool_calls", []),
            "correctness": agent_data.get("correctness", ""),
            "errors": agent_data.get("errors", []),
        }
    
    return await judge_all_agents(agent_outputs)


async def exemple_judgment():
    """Example of using the judge agent."""
    print("⚖️  Judge Agent: Critical Evaluation of Agent Outputs")
    print("=" * 70)
    
    # Example: Judge Agent 1 output
    agent_1_output = {
        "positions": [
            {"symbole": "AIR.PA", "quantite": 50, "prix_achat": 120.0, "date_achat": "2024-01-15"},
            {"symbole": "SAN.PA", "quantite": 30, "prix_achat": 85.0, "date_achat": "2024-02-20"},
        ],
        "valeur_totale": 8550.0,
        "date_evaluation": "2024-03-01"
    }
    
    print("\n1. Evaluating Agent 1 output...")
    review = await judge_agent_output(
        agent_name="Agent 1",
        agent_output=agent_1_output,
        expected_result=8550.0,
        input_prompt="Extrais le portfolio: 50 AIR.PA à 120€, 30 SAN.PA à 85€",
    )
    
    print(f"\n✅ Agent: {review.agent_name}")
    print(f"   Correctness Score: {review.correctness_score:.2f}/1.0")
    print(f"   Quality Score: {review.quality_score:.2f}/1.0")
    print(f"\n   Forces:")
    for strength in review.strengths:
        print(f"     + {strength}")
    print(f"\n   Faiblesses:")
    for weakness in review.weaknesses:
        print(f"     - {weakness}")
    print(f"\n   Problèmes Critiques:")
    for issue in review.critical_issues:
        print(f"     ⚠️  {issue}")
    print(f"\n   Suggestions d'Amélioration:")
    for suggestion in review.improvement_suggestions:
        print(f"     💡 {suggestion}")
    
    # Example: Comprehensive judgment of multiple agents
    print("\n" + "=" * 70)
    print("2. Comprehensive Evaluation of All Agents...")
    
    all_outputs = {
        "Agent 1": {
            "output": agent_1_output,
            "expected_result": 8550.0,
            "input_prompt": "Extrais le portfolio: 50 AIR.PA à 120€, 30 SAN.PA à 85€",
            "tool_calls": [],
        },
        "Agent 2": {
            "output": "Valeur future: 74,012.21€",
            "expected_result": 74012.21,
            "input_prompt": "50,000€ at 4% for 10 years. Future value?",
            "tool_calls": [{"name": "calculer_valeur_future", "args": {"capital_initial": 50000, "taux": 0.04, "duree": 10}}],
        },
    }
    
    comprehensive = await judge_all_agents(all_outputs)
    
    print(f"\n📊 Évaluation Globale:")
    print(f"   Score Global: {comprehensive.overall_score:.2f}/1.0")
    print(f"\n   Évaluation d'Ensemble:")
    print(f"   {comprehensive.overall_assessment}")
    
    print(f"\n   Problèmes Communs:")
    for issue in comprehensive.common_issues:
        print(f"     - {issue}")
    
    print(f"\n   Meilleures Pratiques Identifiées:")
    for practice in comprehensive.best_practices_identified:
        print(f"     ✓ {practice}")
    
    print(f"\n   Améliorations Prioritaires:")
    for improvement in comprehensive.priority_improvements:
        print(f"     🔧 {improvement}")


async def exemple_judge_from_evaluation_file():
    """Example: Judge agents using evaluation results file."""
    print("⚖️  Judge Agent: Evaluating from evaluation_results.json")
    print("=" * 70)
    
    try:
        comprehensive = await judge_evaluation_results()
        
        print(f"\n📊 Évaluation Globale:")
        print(f"   Score Global: {comprehensive.overall_score:.2f}/1.0")
        print(f"\n   Évaluation d'Ensemble:")
        print(f"   {comprehensive.overall_assessment[:500]}...")
        
        print(f"\n   Problèmes Communs:")
        for issue in comprehensive.common_issues[:5]:
            print(f"     - {issue}")
        
        print(f"\n   Meilleures Pratiques Identifiées:")
        for practice in comprehensive.best_practices_identified[:5]:
            print(f"     ✓ {practice}")
        
        print(f"\n   Améliorations Prioritaires:")
        for improvement in comprehensive.priority_improvements[:5]:
            print(f"     🔧 {improvement}")
        
        print(f"\n   Détails par Agent:")
        for review in comprehensive.agent_reviews:
            print(f"\n   {review.agent_name}:")
            print(f"     Correctness: {review.correctness_score:.2f}/1.0")
            print(f"     Quality: {review.quality_score:.2f}/1.0")
            if review.critical_issues:
                print(f"     ⚠️  Problèmes: {', '.join(review.critical_issues[:2])}")
            
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("   Run evaluate_all_agents.py first to generate evaluation results.")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--from-file":
        # Judge from evaluation results file
        asyncio.run(exemple_judge_from_evaluation_file())
    else:
        # Standard example
        asyncio.run(exemple_judgment())

