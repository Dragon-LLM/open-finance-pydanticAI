#!/usr/bin/env python3
"""
Compare both inference endpoints (Koyeb vs HF) with detailed Langfuse tracing.

Metrics tracked:
- Total time (TTS)
- Tokens per second (it/s)
- Input/Output tokens
- Tool usage (yes/no)
- Structured output (ok/nok)
"""

import asyncio
import time
import sys
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from pydantic import BaseModel, Field
from pydantic_ai import Agent, ModelSettings

from app.config import settings, ENDPOINTS
from app.models import get_model_for_endpoint
from app.langfuse_config import get_langfuse_client
from app.mitigation_strategies import ToolCallDetector

# Import agents
from examples.agent_1 import agent_1, Portfolio
from examples.agent_2_wrapped import agent_2_wrapped as agent_2
from examples.agent_4 import agent_4


# ============================================================================
# LANGFUSE TRACING WITH METRICS
# ============================================================================

def create_comparison_trace(langfuse, endpoint: str, agent_name: str, test_name: str) -> Any:
    """Create a Langfuse span for endpoint comparison."""
    if not langfuse:
        return None
    
    span = langfuse.start_span(
        name=f"{agent_name}_{endpoint}",
        metadata={
            "endpoint": endpoint,
            "agent_name": agent_name,
            "test_name": test_name,
            "comparison_run": True,
            "timestamp": datetime.now().isoformat(),
        },
    )
    return span


def record_metrics(langfuse, trace_id: str, metrics: Dict[str, Any]) -> None:
    """Record metrics as scores in Langfuse."""
    if not langfuse or not trace_id:
        return
    
    try:
        # Record numeric scores
        score_mappings = {
            "tokens_per_second": ("tokens_per_second", metrics.get("tokens_per_second", 0)),
            "total_time": ("latency_seconds", metrics.get("total_time", 0)),
            "input_tokens": ("input_tokens", metrics.get("input_tokens", 0)),
            "output_tokens": ("output_tokens", metrics.get("output_tokens", 0)),
            "total_tokens": ("total_tokens", metrics.get("total_tokens", 0)),
            "tools_used": ("tools_used", 1.0 if metrics.get("tools_called") else 0.0),
            "structured_output_ok": ("structured_output_ok", 1.0 if metrics.get("structured_output_ok") else 0.0),
            "correctness": ("correctness", 1.0 if metrics.get("correct") else 0.0),
        }
        
        for key, (score_name, value) in score_mappings.items():
            if value is not None:
                try:
                    langfuse.create_score(
                        trace_id=trace_id,
                        name=score_name,
                        value=float(value),
                    )
                except Exception as e:
                    print(f"[DEBUG] Failed to create score {score_name}: {e}")
        
        langfuse.flush()
    except Exception as e:
        print(f"[WARNING] Error recording metrics: {e}")


# ============================================================================
# TEST FUNCTIONS
# ============================================================================

async def test_agent_1_on_endpoint(endpoint: str, langfuse) -> Dict[str, Any]:
    """Test Agent 1 (Structured Data Extraction) on a specific endpoint."""
    
    print(f"\n  📊 Agent 1: Structured Data Extraction ({endpoint.upper()})")
    
    # Create model for endpoint
    model = get_model_for_endpoint(endpoint)
    
    # Create agent with endpoint-specific model
    agent = Agent(
        model,
        model_settings=ModelSettings(max_output_tokens=600),
        system_prompt="""Expert analyse financière. Extrais données portfolios boursiers.
Règles: Identifie symbole, quantité, prix_achat, date_achat pour chaque position.
CALCUL CRITIQUE: valeur_totale = Σ(quantité × prix_achat).
Répondez avec un objet Portfolio structuré.""",
        output_type=Portfolio,
    )
    
    # Test input
    texte = "50 AIR.PA à 120€, 30 SAN.PA à 85€, 100 TTE.PA à 55€"
    expected_total = 50*120 + 30*85 + 100*55  # 14,050€
    prompt = f"Extrais le portfolio: {texte}"
    
    # Create trace
    span = create_comparison_trace(langfuse, endpoint, "agent_1", "structured_extraction")
    trace_id = span.trace_id if span and hasattr(span, 'trace_id') else None
    
    # Run agent
    start = time.time()
    result = await agent.run(prompt, output_type=Portfolio)
    elapsed = time.time() - start
    
    # Extract metrics
    portfolio = result.output
    usage = result.usage()
    calculated_total = sum(p.quantite * p.prix_achat for p in portfolio.positions)
    
    metrics = {
        "total_time": elapsed,
        "input_tokens": usage.input_tokens if usage else 0,
        "output_tokens": usage.output_tokens if usage else 0,
        "total_tokens": usage.total_tokens if usage else 0,
        "tokens_per_second": (usage.total_tokens / elapsed) if usage and elapsed > 0 else 0,
        "tools_called": False,
        "structured_output_ok": len(portfolio.positions) == 3,
        "correct": abs(calculated_total - expected_total) < 1,
        "actual_value": calculated_total,
        "expected_value": expected_total,
    }
    
    # Update span
    if span:
        span.update(
            output={
                "positions_count": len(portfolio.positions),
                "calculated_total": calculated_total,
                "correct": metrics["correct"],
            },
            metadata=metrics,
        )
        span.end()
    
    # Record scores
    record_metrics(langfuse, trace_id, metrics)
    
    status = "✅" if metrics["correct"] else "❌"
    print(f"     {status} Time: {elapsed:.2f}s | {metrics['tokens_per_second']:.1f} tok/s | Tokens: {metrics['total_tokens']}")
    print(f"        Structured: {'✅' if metrics['structured_output_ok'] else '❌'} | Value: {calculated_total:,.0f}€")
    
    return metrics


async def test_agent_2_on_endpoint(endpoint: str, langfuse) -> Dict[str, Any]:
    """Test Agent 2 (Financial Tools) on a specific endpoint."""
    
    print(f"\n  🧮 Agent 2: Financial Tools ({endpoint.upper()})")
    
    # Create model for endpoint
    model = get_model_for_endpoint(endpoint)
    
    # Import agent_2's tools and recreate with new model
    from examples.agent_2 import calculer_valeur_future, calculer_paiement_periodique
    
    agent = Agent(
        model,
        model_settings=ModelSettings(max_output_tokens=400),
        system_prompt="Expert calculs financiers. Utilise les outils pour les calculs précis.",
        tools=[calculer_valeur_future, calculer_paiement_periodique],
    )
    
    # Test input
    question = "50000€ a 4% sur 5 ans?"
    import numpy_financial as npf
    expected_fv = abs(npf.fv(rate=0.04, nper=5, pmt=0, pv=-50000))
    
    # Create trace
    span = create_comparison_trace(langfuse, endpoint, "agent_2", "financial_calculation")
    trace_id = span.trace_id if span and hasattr(span, 'trace_id') else None
    
    # Run agent
    start = time.time()
    result = await agent.run(question)
    elapsed = time.time() - start
    
    # Extract metrics
    usage = result.usage()
    tool_calls = ToolCallDetector.extract_tool_calls(result) or []
    
    # Try to extract result value
    actual_value = None
    output_text = str(result.output)
    import re
    numbers = re.findall(r'[\d\s,]+\.?\d*', output_text.replace(' ', ''))
    for num_str in numbers:
        try:
            num = float(num_str.replace(',', '').replace(' ', ''))
            if 50000 < num < 100000:  # Reasonable range for FV
                actual_value = num
                break
        except:
            continue
    
    correct = actual_value is not None and abs(actual_value - expected_fv) < expected_fv * 0.05
    
    metrics = {
        "total_time": elapsed,
        "input_tokens": usage.input_tokens if usage else 0,
        "output_tokens": usage.output_tokens if usage else 0,
        "total_tokens": usage.total_tokens if usage else 0,
        "tokens_per_second": (usage.total_tokens / elapsed) if usage and elapsed > 0 else 0,
        "tools_called": len(tool_calls) > 0,
        "tool_names": [tc.get('name', 'unknown') for tc in tool_calls],
        "structured_output_ok": True,  # Text output
        "correct": correct,
        "actual_value": actual_value,
        "expected_value": expected_fv,
    }
    
    # Update span
    if span:
        span.update(
            output={
                "tools_used": metrics["tool_names"],
                "actual_value": actual_value,
                "correct": correct,
            },
            metadata=metrics,
        )
        span.end()
    
    # Record scores
    record_metrics(langfuse, trace_id, metrics)
    
    status = "✅" if correct else "❌"
    tools_str = "✅ " + ", ".join(metrics["tool_names"]) if metrics["tools_called"] else "❌ No tools"
    print(f"     {status} Time: {elapsed:.2f}s | {metrics['tokens_per_second']:.1f} tok/s | Tokens: {metrics['total_tokens']}")
    print(f"        Tools: {tools_str}")
    
    return metrics


async def test_agent_4_on_endpoint(endpoint: str, langfuse) -> Dict[str, Any]:
    """Test Agent 4 (Option Pricing) on a specific endpoint."""
    
    print(f"\n  📈 Agent 4: Option Pricing ({endpoint.upper()})")
    
    # Create model for endpoint
    model = get_model_for_endpoint(endpoint)
    
    # Import agent_4's tools and recreate with new model
    from examples.agent_4 import calculer_prix_call_black_scholes, calculer_prix_put_black_scholes
    
    agent = Agent(
        model,
        model_settings=ModelSettings(max_output_tokens=600),
        system_prompt="Expert pricing options financières avec QuantLib. Utilise les outils Black-Scholes.",
        tools=[calculer_prix_call_black_scholes, calculer_prix_put_black_scholes],
    )
    
    # Test input
    question = "Prix call: Spot=100, Strike=105, Maturité=0.5an, Taux=0.02, Vol=0.25, Div=0.01"
    expected_price_range = (2.0, 6.0)
    
    # Create trace
    span = create_comparison_trace(langfuse, endpoint, "agent_4", "option_pricing")
    trace_id = span.trace_id if span and hasattr(span, 'trace_id') else None
    
    # Run agent
    start = time.time()
    result = await agent.run(question)
    elapsed = time.time() - start
    
    # Extract metrics
    usage = result.usage()
    tool_calls = ToolCallDetector.extract_tool_calls(result) or []
    
    # Try to extract price
    actual_price = None
    output_text = str(result.output)
    import re
    numbers = re.findall(r'\d+\.?\d*', output_text)
    for num_str in numbers:
        try:
            num = float(num_str)
            if 1.0 < num < 20.0:  # Reasonable range for option price
                actual_price = num
                break
        except:
            continue
    
    correct = actual_price is not None and expected_price_range[0] <= actual_price <= expected_price_range[1]
    
    metrics = {
        "total_time": elapsed,
        "input_tokens": usage.input_tokens if usage else 0,
        "output_tokens": usage.output_tokens if usage else 0,
        "total_tokens": usage.total_tokens if usage else 0,
        "tokens_per_second": (usage.total_tokens / elapsed) if usage and elapsed > 0 else 0,
        "tools_called": len(tool_calls) > 0,
        "tool_names": [tc.get('name', 'unknown') for tc in tool_calls],
        "structured_output_ok": True,
        "correct": correct,
        "actual_value": actual_price,
        "expected_range": expected_price_range,
    }
    
    # Update span
    if span:
        span.update(
            output={
                "tools_used": metrics["tool_names"],
                "actual_price": actual_price,
                "correct": correct,
            },
            metadata=metrics,
        )
        span.end()
    
    # Record scores
    record_metrics(langfuse, trace_id, metrics)
    
    status = "✅" if correct else "❌"
    tools_str = "✅ " + ", ".join(metrics["tool_names"]) if metrics["tools_called"] else "❌ No tools"
    print(f"     {status} Time: {elapsed:.2f}s | {metrics['tokens_per_second']:.1f} tok/s | Tokens: {metrics['total_tokens']}")
    print(f"        Tools: {tools_str} | Price: {actual_price}")
    
    return metrics


# ============================================================================
# MAIN COMPARISON
# ============================================================================

async def run_comparison():
    """Run full comparison on both endpoints."""
    
    print("=" * 70)
    print("🔬 ENDPOINT COMPARISON: KOYEB vs HF SPACE")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Langfuse: {'✅ Enabled' if settings.enable_langfuse else '❌ Disabled'}")
    
    # Initialize Langfuse
    langfuse = get_langfuse_client()
    if langfuse:
        print(f"Traces will be available at: https://cloud.langfuse.com")
    
    endpoints = ["koyeb", "hf"]
    results = {ep: {} for ep in endpoints}
    
    for endpoint in endpoints:
        print(f"\n{'='*70}")
        print(f"📍 TESTING ENDPOINT: {endpoint.upper()}")
        print(f"   URL: {ENDPOINTS[endpoint]['url']}")
        print(f"   Model: {ENDPOINTS[endpoint]['model']}")
        print(f"{'='*70}")
        
        try:
            # Test Agent 1
            results[endpoint]["agent_1"] = await test_agent_1_on_endpoint(endpoint, langfuse)
            await asyncio.sleep(1)
            
            # Test Agent 2
            results[endpoint]["agent_2"] = await test_agent_2_on_endpoint(endpoint, langfuse)
            await asyncio.sleep(1)
            
            # Test Agent 4
            results[endpoint]["agent_4"] = await test_agent_4_on_endpoint(endpoint, langfuse)
            
        except Exception as e:
            print(f"   ❌ Error testing {endpoint}: {e}")
            import traceback
            traceback.print_exc()
    
    # Flush Langfuse
    if langfuse:
        langfuse.flush()
    
    # Print comparison summary
    print("\n" + "=" * 70)
    print("📊 COMPARISON SUMMARY")
    print("=" * 70)
    
    print(f"\n{'Agent':<25} | {'Metric':<15} | {'KOYEB':<15} | {'HF':<15} | {'Winner':<10}")
    print("-" * 85)
    
    for agent_name in ["agent_1", "agent_2", "agent_4"]:
        koyeb_data = results.get("koyeb", {}).get(agent_name, {})
        hf_data = results.get("hf", {}).get(agent_name, {})
        
        if not koyeb_data or not hf_data:
            continue
        
        # Time comparison
        k_time = koyeb_data.get("total_time", 0)
        h_time = hf_data.get("total_time", 0)
        winner_time = "KOYEB" if k_time < h_time else "HF" if h_time < k_time else "TIE"
        print(f"{agent_name:<25} | {'Time (s)':<15} | {k_time:<15.2f} | {h_time:<15.2f} | {winner_time:<10}")
        
        # Speed comparison
        k_speed = koyeb_data.get("tokens_per_second", 0)
        h_speed = hf_data.get("tokens_per_second", 0)
        winner_speed = "KOYEB" if k_speed > h_speed else "HF" if h_speed > k_speed else "TIE"
        print(f"{'':<25} | {'tok/s':<15} | {k_speed:<15.1f} | {h_speed:<15.1f} | {winner_speed:<10}")
        
        # Correctness
        k_correct = "✅" if koyeb_data.get("correct") else "❌"
        h_correct = "✅" if hf_data.get("correct") else "❌"
        print(f"{'':<25} | {'Correct':<15} | {k_correct:<15} | {h_correct:<15} |")
        
        # Tools
        k_tools = "✅" if koyeb_data.get("tools_called") else "❌"
        h_tools = "✅" if hf_data.get("tools_called") else "❌"
        print(f"{'':<25} | {'Tools Used':<15} | {k_tools:<15} | {h_tools:<15} |")
        
        print("-" * 85)
    
    # Overall summary
    print("\n📈 OVERALL METRICS:")
    for endpoint in endpoints:
        total_time = sum(r.get("total_time", 0) for r in results[endpoint].values())
        avg_speed = sum(r.get("tokens_per_second", 0) for r in results[endpoint].values()) / len(results[endpoint]) if results[endpoint] else 0
        correct_count = sum(1 for r in results[endpoint].values() if r.get("correct"))
        
        print(f"   {endpoint.upper()}: Total time: {total_time:.2f}s | Avg speed: {avg_speed:.1f} tok/s | Correct: {correct_count}/{len(results[endpoint])}")
    
    print("\n" + "=" * 70)
    print("🔗 View detailed traces in Langfuse: https://cloud.langfuse.com")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(run_comparison())
