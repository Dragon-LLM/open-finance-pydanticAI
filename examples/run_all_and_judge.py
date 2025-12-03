"""
Run all agents and then judge their outputs using the Judge Agent.

This script:
1. Runs all 5 agents with sample inputs
2. Collects their outputs
3. Uses the Judge Agent to provide critical evaluation
"""

import asyncio
import sys
from pathlib import Path
from typing import Dict, Any

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from examples.judge_agent import judge_all_agents
from examples.agent_1 import agent_1, Portfolio
from examples.agent_2 import agent_2
from examples.agent_3 import risk_analyst, tax_advisor
from examples.agent_4 import agent_4
from examples.agent_5 import agent_5
from app.mitigation_strategies import ToolCallDetector


async def run_all_agents() -> Dict[str, Dict[str, Any]]:
    """Run all agents and collect their outputs."""
    print("🚀 Running All Agents...")
    print("=" * 70)
    
    agent_outputs = {}
    
    # Agent 1: Portfolio extraction
    print("\n1. Running Agent 1 (Portfolio Extraction)...")
    try:
        prompt_1 = "Extrais le portfolio: 50 AIR.PA à 120€, 30 SAN.PA à 85€, 100 TTE.PA à 55€"
        result_1 = await agent_1.run(prompt_1, output_type=Portfolio)
        tool_calls_1 = ToolCallDetector.extract_tool_calls(result_1) or []
        agent_outputs["Agent 1: Portfolio Extraction"] = {
            "output": result_1.output,
            "expected_result": 14050.0,  # 50*120 + 30*85 + 100*55
            "input_prompt": prompt_1,
            "tool_calls": tool_calls_1,
        }
        print(f"   ✅ Completed - Portfolio value: {result_1.output.valeur_totale}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
        agent_outputs["Agent 1: Portfolio Extraction"] = {
            "output": f"Error: {e}",
            "input_prompt": prompt_1,
            "tool_calls": [],
        }
    
    # Agent 2: Financial calculation
    print("\n2. Running Agent 2 (Financial Tools)...")
    try:
        prompt_2 = "50,000€ at 4% for 10 years. How much will I have?"
        result_2 = await agent_2.run(prompt_2)
        tool_calls_2 = ToolCallDetector.extract_tool_calls(result_2) or []
        agent_outputs["Agent 2: Financial Calculations"] = {
            "output": str(result_2.output),
            "expected_result": "~74,012€",
            "input_prompt": prompt_2,
            "tool_calls": tool_calls_2,
        }
        print(f"   ✅ Completed - Used tools: {len(tool_calls_2)}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
        agent_outputs["Agent 2: Financial Calculations"] = {
            "output": f"Error: {e}",
            "input_prompt": prompt_2,
            "tool_calls": [],
        }
    
    # Agent 3: Risk analysis
    print("\n3. Running Agent 3 (Risk Analysis)...")
    try:
        prompt_3 = "Analyse le niveau de risque (1-5) d'un portfolio: 40% actions, 30% obligations, 20% immobilier, 10% autres. Investissement 100k€, 30 ans."
        result_3 = await risk_analyst.run(prompt_3)
        tool_calls_3 = ToolCallDetector.extract_tool_calls(result_3) or []
        agent_outputs["Agent 3: Risk Analysis"] = {
            "output": result_3.output,
            "expected_result": "Risk level 1-5 with structured output",
            "input_prompt": prompt_3,
            "tool_calls": tool_calls_3,
        }
        print(f"   ✅ Completed - Risk level: {result_3.output.niveau_risque}/5")
    except Exception as e:
        print(f"   ❌ Error: {e}")
        agent_outputs["Agent 3: Risk Analysis"] = {
            "output": f"Error: {e}",
            "input_prompt": prompt_3,
            "tool_calls": [],
        }
    
    # Agent 4: Option pricing
    print("\n4. Running Agent 4 (Option Pricing)...")
    try:
        prompt_4 = """Calcule le prix d'un call européen:
- Spot: 100
- Strike: 105
- Maturité: 0.5 an
- Taux sans risque: 0.02
- Volatilité: 0.25
- Dividende: 0.01"""
        result_4 = await agent_4.run(prompt_4)
        tool_calls_4 = ToolCallDetector.extract_tool_calls(result_4) or []
        agent_outputs["Agent 4: Option Pricing"] = {
            "output": result_4.output,
            "expected_result": "Price ~5.15 with Greeks",
            "input_prompt": prompt_4,
            "tool_calls": tool_calls_4,
        }
        print(f"   ✅ Completed - Option price: {result_4.output.option_price}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
        agent_outputs["Agent 4: Option Pricing"] = {
            "output": f"Error: {e}",
            "input_prompt": prompt_4,
            "tool_calls": [],
        }
    
    # Agent 5: SWIFT conversion
    print("\n5. Running Agent 5 (SWIFT/ISO 20022 Conversion)...")
    try:
        swift_msg = """{1:F01BANKFRPPAXXX1234567890}
{2:O10312002401031200BANKDEFFXXX22221234567890123456789012345678901234567890}
{4:
:20:REF123456789
:32A:240101EUR1000,00
:50A:/FR1420041010050500013M02606
COMPAGNIE ABC
:59:/DE89370400440532013000
COMPAGNIE XYZ
-}"""
        prompt_5 = f"Convertis ce message SWIFT MT103 en ISO 20022:\n\n{swift_msg}"
        result_5 = await agent_5.run(prompt_5)
        tool_calls_5 = ToolCallDetector.extract_tool_calls(result_5) or []
        agent_outputs["Agent 5: SWIFT/ISO 20022"] = {
            "output": str(result_5.output)[:500],  # Truncate for display
            "expected_result": "ISO 20022 XML message",
            "input_prompt": prompt_5[:100] + "...",
            "tool_calls": tool_calls_5,
        }
        print(f"   ✅ Completed - Used tools: {len(tool_calls_5)}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
        agent_outputs["Agent 5: SWIFT/ISO 20022"] = {
            "output": f"Error: {e}",
            "input_prompt": prompt_5[:100] if 'prompt_5' in locals() else "",
            "tool_calls": [],
        }
    
    return agent_outputs


async def main():
    """Run all agents and judge their outputs."""
    print("=" * 70)
    print("COMPREHENSIVE AGENT EVALUATION WITH JUDGE AGENT")
    print("=" * 70)
    
    # Run all agents
    agent_outputs = await run_all_agents()
    
    # Judge all agents
    print("\n" + "=" * 70)
    print("⚖️  Judge Agent: Critical Evaluation")
    print("=" * 70)
    
    try:
        comprehensive = await judge_all_agents(agent_outputs)
        
        print(f"\n📊 ÉVALUATION GLOBALE")
        print(f"   Score Global: {comprehensive.overall_score:.2f}/1.0")
        print(f"\n   Évaluation d'Ensemble:")
        print(f"   {comprehensive.overall_assessment}")
        
        print(f"\n   Problèmes Communs Identifiés:")
        for issue in comprehensive.common_issues:
            print(f"     ⚠️  {issue}")
        
        print(f"\n   Meilleures Pratiques:")
        for practice in comprehensive.best_practices_identified:
            print(f"     ✓ {practice}")
        
        print(f"\n   Améliorations Prioritaires:")
        for improvement in comprehensive.priority_improvements:
            print(f"     🔧 {improvement}")
        
        print(f"\n   DÉTAILS PAR AGENT:")
        print("   " + "-" * 66)
        for review in comprehensive.agent_reviews:
            print(f"\n   📋 {review.agent_name}")
            print(f"      Correctness: {review.correctness_score:.2f}/1.0")
            print(f"      Quality: {review.quality_score:.2f}/1.0")
            
            if review.strengths:
                print(f"      Forces:")
                for strength in review.strengths[:3]:
                    print(f"        + {strength}")
            
            if review.weaknesses:
                print(f"      Faiblesses:")
                for weakness in review.weaknesses[:3]:
                    print(f"        - {weakness}")
            
            if review.critical_issues:
                print(f"      Problèmes Critiques:")
                for issue in review.critical_issues:
                    print(f"        ⚠️  {issue}")
            
            if review.improvement_suggestions:
                print(f"      Suggestions:")
                for suggestion in review.improvement_suggestions[:2]:
                    print(f"        💡 {suggestion}")
        
    except Exception as e:
        print(f"❌ Error in Judge Agent: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())

