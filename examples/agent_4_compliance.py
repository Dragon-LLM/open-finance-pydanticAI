"""
Agent 4 Compliance: Wrapper that executes agent_4 (option pricing) and runs compliance checks.

Optimizations:
- Uses optimized agent_4 with QuantLib tools
- Optimized tool signatures
- Concise compliance agent prompt
- Verifies correct QuantLib tool usage
"""

import asyncio
from typing import List, Tuple
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from pydantic_ai import Agent, ModelSettings
from app.models import finance_model
from app.mitigation_strategies import ToolCallDetector

# Import optimized agent_4
from examples.agent_4 import agent_4


async def run_option_pricing_agent(question: str):
    """Execute the option pricing agent and return (result, tool_calls_log)."""
    result = await agent_4.run(question)
    
    # Use ToolCallDetector for better extraction
    tool_calls = ToolCallDetector.extract_tool_calls(result) or []
    
    # Format tool calls for compliance check
    tool_calls_log: List[str] = []
    for tc in tool_calls:
        name = tc.get('name', 'unknown')
        args = tc.get('args', {})
        args_str = ', '.join(f"{k}={v}" for k, v in args.items()) if isinstance(args, dict) else str(args)
        tool_calls_log.append(f"{name}({args_str})")
    
    return result, tool_calls_log


# ============================================================================
# OPTIMIZED COMPLIANCE AGENT FOR OPTION PRICING
# ============================================================================

compliance_agent = Agent(
    finance_model,
    model_settings=ModelSettings(max_output_tokens=400),
    system_prompt="""Contrôleur compliance pour pricing d'options avec QuantLib.
Règles:
1. Liste d'outils vide → Non conforme (calculs manuels interdits)
2. calculer_prix_call_black_scholes utilisé → Conforme
3. Autre outil ou calcul mentionné sans outil → Non conforme
4. Vérifier que tous paramètres (spot, strike, maturité, taux, volatilité) sont présents
Réponse: 'Conforme' ou 'Non conforme' + justification courte.""",  # 95 tokens - slightly longer due to QuantLib specificity
)


async def run_with_compliance(question: str) -> Tuple[str, List[str], str]:
    """Run option pricing agent with compliance check.
    
    Returns:
        (agent_response_json, tool_calls, compliance_verdict)
    """
    result, tool_calls = await run_option_pricing_agent(question)
    
    # Minimal compliance check to save tokens
    tool_used = any("calculer_prix_call_black_scholes" in tc for tc in tool_calls)
    compliance_verdict = f"✅ Conforme - QuantLib utilisé" if tool_used else "❌ Non conforme - QuantLib requis"
    
    # Return proper JSON with all Greeks
    import json
    if hasattr(result.output, 'model_dump'):
        response_json = json.dumps(result.output.model_dump(), indent=2, ensure_ascii=False)
    else:
        response_json = json.dumps(result.output, indent=2, default=str, ensure_ascii=False)
    
    return response_json, tool_calls, compliance_verdict


async def exemple_compliance_check():
    """Example of compliance checking for option pricing."""
    print("📊 Agent 4 Compliance: Option Pricing with Compliance Check")
    print("=" * 70)
    
    questions = [
        (
            "Calcule le prix d'un call européen:\n"
            "- Spot: 100\n"
            "- Strike: 105\n"
            "- Maturité: 0.5 an\n"
            "- Taux sans risque: 0.02\n"
            "- Volatilité: 0.25\n"
            "- Dividende: 0.01"
        ),
        (
            "Prix d'un call avec:\n"
            "- Spot: 50\n"
            "- Strike: 55\n"
            "- Maturité: 1 an\n"
            "- Taux: 3%\n"
            "- Volatilité: 20%"
        ),
    ]
    
    for i, question in enumerate(questions, 1):
        print(f"\n{'='*70}")
        print(f"Question {i}:")
        print(question)
        print("="*70)
        
        try:
            import time
            start = time.time()
            response, tool_calls, compliance = await run_with_compliance(question)
            elapsed = time.time() - start
            
            print(f"\n✅ Réponse Agent:")
            print(f"{response}\n")
            
            print(f"🔧 Appels d'outils détectés:")
            if tool_calls:
                for tc in tool_calls:
                    print(f"  - {tc}")
            else:
                print("  ⚠️ Aucun (non conforme - calculs manuels interdits)")
            
            # Check if correct tool was used
            correct_tool_used = any(
                "calculer_prix_call_black_scholes" in tc for tc in tool_calls
            )
            if tool_calls and not correct_tool_used:
                print("  ⚠️ Outil incorrect utilisé")
            
            print(f"\n🔍 Avis Compliance:")
            print(f"{compliance}")
            
            print(f"\n⏱️  Temps: {elapsed:.2f}s")
            
        except Exception as e:
            print(f"❌ Erreur: {e}")
            import traceback
            traceback.print_exc()
        
        print("\n" + "-" * 70 + "\n")


async def main():
    """Main function."""
    await exemple_compliance_check()


if __name__ == "__main__":
    asyncio.run(main())



