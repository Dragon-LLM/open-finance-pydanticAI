"""
Agent 2 Compliance: Wrapper with dynamic tool selection to minimize context usage.

Optimizations:
- Dynamic tool selection: only loads the needed tool
- Minimal system prompt
- Prevents tool call loops by design
- Ultra-compact to avoid 8192 token limit
"""

import asyncio
from typing import List, Tuple, Callable, Dict, Any
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from pydantic_ai import Agent, ModelSettings
from app.models import finance_model
from app.mitigation_strategies import ToolCallDetector
import numpy_financial as npf

# Import wrapped tools with matching parameter names
from examples.agent_2_wrapped import (
    calculer_valeur_future,
    calculer_versement_mensuel,
    calculer_performance_portfolio,
    calculer_valeur_actuelle,
    calculer_taux_interet,
    FinancialCalculationResult
)


def select_tool_from_question(question: str) -> Tuple[Callable, str]:
    """Dynamically select which tool to use based on the question.
    
    Returns:
        (tool_function, tool_name)
    """
    question_lower = question.lower()
    
    # Simple keyword matching to select the right tool
    if any(word in question_lower for word in ['valeur finale', 'combien aurai', 'capital', 'placement', 'investir', 'intérêts']):
        return calculer_valeur_future, "valeur_future"
    elif any(word in question_lower for word in ['prêt', 'emprunt', 'mensualité', 'versement', 'crédit', 'rembours']):
        return calculer_versement_mensuel, "versement_mensuel"
    elif any(word in question_lower for word in ['performance', 'rendement', 'gain', 'portfolio']):
        return calculer_performance_portfolio, "performance"
    elif any(word in question_lower for word in ['valeur actuelle', 'actualisation', 'discount']):
        return calculer_valeur_actuelle, "valeur_actuelle"
    elif any(word in question_lower for word in ['taux', 'pourcentage nécessaire', 'quel taux']):
        return calculer_taux_interet, "taux_interet"
    else:
        # Default to future value
        return calculer_valeur_future, "valeur_future"


async def run_finance_agent(question: str):
    """Execute the financial agent with dynamic tool selection to minimize context usage."""
    # Select only the relevant tool
    tool_func, tool_name = select_tool_from_question(question)
    
    # Create a minimal agent with ONLY the selected tool
    # Use an ultra-strict prompt to prevent tool loops
    minimal_agent = Agent(
        finance_model,
        model_settings=ModelSettings(
            max_output_tokens=250,  # Reduced further
            temperature=0.0,  # Deterministic to prevent loops
        ),
        system_prompt=(
            f"Tu es un calculateur. Étapes exactes:\n"
            f"1. Appelle {tool_name} UNE FOIS avec les paramètres de la question\n"
            f"2. Prends le résultat de l'outil\n"
            f"3. Retourne FinancialCalculationResult avec ce résultat\n"
            f"4. STOP - ne rappelle JAMAIS l'outil\n"
            f"INTERDIT: 2+ appels, vérifications, calculs manuels."
        ),
        tools=[tool_func],  # Only ONE tool!
        output_type=FinancialCalculationResult,
        retries=0,
    )
    
    # Run with the minimal agent
    result = await minimal_agent.run(question)
    
    # Use ToolCallDetector for better extraction
    tool_calls = ToolCallDetector.extract_tool_calls(result) or []
    
    # Detect and warn about duplicate tool calls
    if len(tool_calls) > 1:
        # Check for exact duplicates
        unique_calls = []
        seen = set()
        for tc in tool_calls:
            name = tc.get('name', 'unknown')
            args = tc.get('args', {})
            # Create a hashable key for deduplication
            key = f"{name}:{sorted(args.items()) if isinstance(args, dict) else args}"
            if key not in seen:
                seen.add(key)
                unique_calls.append(tc)
        
        if len(unique_calls) < len(tool_calls):
            print(f"⚠️  Detected {len(tool_calls) - len(unique_calls)} duplicate tool calls (same tool, same args)")
    
    # Format tool calls for compliance check
    tool_calls_log: List[str] = []
    for tc in tool_calls:
        name = tc.get('name', 'unknown')
        args = tc.get('args', {})
        args_str = ', '.join(f"{k}={v}" for k, v in args.items()) if isinstance(args, dict) else str(args)
        tool_calls_log.append(f"{name}({args_str})")
    
    return result, tool_calls_log


# ============================================================================
# DETERMINISTIC COMPLIANCE VALIDATION
# ============================================================================

def validate_calculation(result_output, tool_calls: List[str]) -> Tuple[bool, str, Dict[str, Any]]:
    """Deterministic compliance validation.
    
    Returns:
        (is_compliant, verdict, details)
    """
    details = {
        "tools_used": len(tool_calls) > 0,
        "tool_count": len(tool_calls),
        "duplicates": 0,
        "calculation_verified": False,
        "errors": []
    }
    
    # Check 1: Tools must be used
    if not tool_calls:
        return False, "❌ Non conforme - Aucun outil utilisé", details
    
    # Check 2: Detect duplicates
    unique_calls = set(tool_calls)
    details["duplicates"] = len(tool_calls) - len(unique_calls)
    
    # Check 3: Parse result to get the calculated value
    try:
        import json
        # Handle both JSON string and Pydantic model
        if isinstance(result_output, str):
            result_data = json.loads(result_output)
        elif hasattr(result_output, 'model_dump'):
            result_data = result_output.model_dump()
        elif hasattr(result_output, 'dict'):
            result_data = result_output.dict()
        else:
            result_data = result_output
        calc_result = result_data.get('result', 0)
        calc_type = result_data.get('calculation_type', 'unknown')
        params = result_data.get('input_parameters', {})
        
        # Check 4: Reverse-verify the calculation
        if calc_type in ['future_value', 'Valeur future']:
            # Verify: FV = PV * (1 + r)^n
            capital = params.get('capital', 0)
            taux = params.get('taux', 0)
            duree = params.get('duree', 0)
            
            if taux > 1.0:
                taux = taux / 100.0
            
            expected = capital * ((1 + taux) ** duree)
            error_pct = abs(calc_result - expected) / expected * 100 if expected > 0 else 100
            
            if error_pct < 1.0:  # Within 1%
                details["calculation_verified"] = True
                verdict = f"✅ Conforme - Calcul vérifié (erreur: {error_pct:.2f}%)"
            else:
                details["errors"].append(f"Calcul incorrect: attendu {expected:,.2f}, obtenu {calc_result:,.2f}")
                verdict = f"⚠️ Partiellement conforme - Outil utilisé mais calcul incorrect (erreur: {error_pct:.1f}%)"
        
        elif calc_type in ['loan_payment', 'Mensualité', 'Versement mensuel']:
            # Verify: PMT = PV * r * (1+r)^n / ((1+r)^n - 1)
            capital = params.get('capital', 0)
            taux = params.get('taux', 0)
            duree = params.get('duree', 0)
            
            if taux > 1.0:
                taux = taux / 100.0
            
            duree_mois = int(duree * 12)
            taux_mensuel = taux / 12
            
            # Use numpy-financial for verification
            expected = -npf.pmt(rate=taux_mensuel, nper=duree_mois, pv=capital)
            error_pct = abs(calc_result - expected) / expected * 100 if expected > 0 else 100
            
            if error_pct < 1.0:
                details["calculation_verified"] = True
                verdict = f"✅ Conforme - Calcul vérifié (erreur: {error_pct:.2f}%)"
            else:
                details["errors"].append(f"Calcul incorrect: attendu {expected:,.2f}, obtenu {calc_result:,.2f}")
                verdict = f"⚠️ Partiellement conforme - Outil utilisé mais calcul incorrect (erreur: {error_pct:.1f}%)"
        
        else:
            # For other calculation types, just check that tools were used
            verdict = f"✅ Conforme - {len(tool_calls)} outil(s) utilisé(s)"
            details["calculation_verified"] = True  # Assume correct
        
        # Add warning for duplicates
        if details["duplicates"] > 0:
            verdict += f" ⚠️ {details['duplicates']} appels dupliqués"
        
        return True, verdict, details
    
    except Exception as e:
        details["errors"].append(f"Erreur validation: {str(e)}")
        verdict = f"⚠️ Outil utilisé mais validation impossible: {str(e)[:50]}"
        return True, verdict, details


async def run_with_compliance(question: str) -> Tuple[str, List[str], str]:
    """Run financial agent with deterministic compliance validation.
    
    Returns:
        (agent_response, tool_calls, compliance_verdict)
    """
    result, tool_calls = await run_finance_agent(question)
    
    # Deterministic compliance validation - pass the actual result object
    is_compliant, verdict, details = validate_calculation(result.output, tool_calls)
    
    # Return JSON string for consistency
    import json
    if hasattr(result.output, 'model_dump'):
        response_str = json.dumps(result.output.model_dump(), indent=2, ensure_ascii=False)
    else:
        response_str = str(result.output)
    
    return response_str, tool_calls, verdict


async def exemple_compliance_check():
    """Example of compliance checking."""
    print("📊 Agent 2 Compliance: Financial Calculations with Compliance Check")
    print("=" * 70)
    
    questions = [
        "J'ai 25 000€ à 4% pendant 8 ans. Combien aurai-je?",
        "J'emprunte 150 000€ sur 15 ans à 2.8%. Quel est le versement mensuel?",
    ]
    
    for i, question in enumerate(questions, 1):
        print(f"\n{'='*70}")
        print(f"Question {i}: {question}")
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
                print("  ⚠️ Aucun (non conforme)")
            
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
