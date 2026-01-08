#!/usr/bin/env python3
"""
Comprehensive Langfuse Evaluation for Open Finance Agents.

This script:
1. Creates/uses datasets in Langfuse
2. Runs experiments on both Koyeb and HF endpoints
3. Records comprehensive scores for comparison:
   - latency_seconds (numeric)
   - tokens_per_second (numeric) 
   - input_tokens (numeric)
   - output_tokens (numeric)
   - tools_used (boolean: 0/1)
   - structured_output_ok (boolean: 0/1)
   - correctness (boolean: 0/1)

View results in Langfuse: https://cloud.langfuse.com
"""

import asyncio
import time
import sys
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from pydantic import BaseModel, Field
from pydantic_ai import Agent, ModelSettings
from langfuse import Langfuse

from app.config import settings, ENDPOINTS
from app.models import get_model_for_endpoint
from app.langfuse_datasets import AGENT_1_DATASET, AGENT_2_DATASET, AGENT_4_DATASET
from app.mitigation_strategies import ToolCallDetector

# Import agent output types
from examples.agent_1 import Portfolio


# ============================================================================
# LANGFUSE CLIENT
# ============================================================================

def get_langfuse() -> Langfuse:
    """Get configured Langfuse client."""
    return Langfuse(
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key,
        host=settings.langfuse_host_resolved,
    )


# ============================================================================
# DATASET MANAGEMENT
# ============================================================================

def ensure_dataset(langfuse: Langfuse, name: str, items: List[Dict]) -> str:
    """Ensure dataset exists in Langfuse, create if needed."""
    try:
        dataset = langfuse.get_dataset(name)
        print(f"  📂 Using existing dataset: {name}")
        return name
    except Exception:
        # Create new dataset
        print(f"  📂 Creating dataset: {name}")
        langfuse.create_dataset(name=name)
        
        # Add items
        for i, item in enumerate(items):
            langfuse.create_dataset_item(
                dataset_name=name,
                input=item["prompt"],
                expected_output=item.get("expected_output"),
                metadata={
                    "difficulty": item.get("difficulty", "unknown"),
                    "category": item.get("category", "unknown"),
                    "item_index": i,
                }
            )
        
        langfuse.flush()
        return name


# ============================================================================
# SCORING FUNCTIONS
# ============================================================================

def record_scores(langfuse: Langfuse, trace_id: str, metrics: Dict[str, Any]) -> None:
    """Record all metrics as scores on a trace."""
    
    # Numeric scores
    numeric_scores = [
        ("latency_seconds", metrics.get("total_time", 0)),
        ("tokens_per_second", metrics.get("tokens_per_second", 0)),
        ("input_tokens", metrics.get("input_tokens", 0)),
        ("output_tokens", metrics.get("output_tokens", 0)),
        ("total_tokens", metrics.get("total_tokens", 0)),
    ]
    
    for name, value in numeric_scores:
        if value is not None:
            langfuse.create_score(
                trace_id=trace_id,
                name=name,
                value=float(value),
                data_type="NUMERIC",
            )
    
    # Boolean scores (0 or 1)
    boolean_scores = [
        ("tools_used", 1 if metrics.get("tools_called") else 0),
        ("structured_output_ok", 1 if metrics.get("structured_output_ok") else 0),
        ("correctness", 1 if metrics.get("correct") else 0),
    ]
    
    for name, value in boolean_scores:
        langfuse.create_score(
            trace_id=trace_id,
            name=name,
            value=value,
            data_type="BOOLEAN",
        )
    
    # Categorical scores
    if metrics.get("difficulty"):
        langfuse.create_score(
            trace_id=trace_id,
            name="difficulty",
            value=metrics["difficulty"],
            data_type="CATEGORICAL",
        )


# ============================================================================
# AGENT RUNNERS
# ============================================================================

async def run_agent_1_item(
    endpoint: str,
    item: Dict,
    langfuse: Langfuse,
    run_name: str,
) -> Dict[str, Any]:
    """Run Agent 1 (Portfolio Extraction) on a single item."""
    
    model = get_model_for_endpoint(endpoint)
    agent = Agent(
        model,
        model_settings=ModelSettings(max_output_tokens=600),
        system_prompt="""Expert analyse financière. Extrais données portfolios boursiers.
Règles: Identifie symbole, quantité, prix_achat, date_achat pour chaque position.
CALCUL CRITIQUE: valeur_totale = Σ(quantité × prix_achat).
Répondez avec un objet Portfolio structuré.""",
        output_type=Portfolio,
    )
    
    prompt = item["prompt"]
    expected = item.get("expected_output", {})
    expected_total = expected.get("total_value") if isinstance(expected, dict) else None
    expected_count = expected.get("positions_count") if isinstance(expected, dict) else None
    
    # Create trace with span
    with langfuse.start_as_current_span(
        name=f"agent_1_{endpoint}",
        input={"prompt": prompt[:500]},
        metadata={
            "endpoint": endpoint,
            "agent": "agent_1",
            "run_name": run_name,
            "difficulty": item.get("difficulty"),
            "category": item.get("category"),
        }
    ) as span:
        trace_id = span.trace_id
        
        start = time.time()
        try:
            result = await agent.run(prompt, output_type=Portfolio)
            elapsed = time.time() - start
            
            portfolio = result.output
            usage = result.usage()
            calculated_total = sum(p.quantite * p.prix_achat for p in portfolio.positions)
            
            # Validate structured output quality
            structured_ok = (
                len(portfolio.positions) > 0 and
                all(p.symbole and p.quantite > 0 and p.prix_achat > 0 for p in portfolio.positions) and
                portfolio.valeur_totale > 0
            )
            
            # Check correctness with tolerance
            total_ok = expected_total is None or abs(calculated_total - expected_total) < max(1, expected_total * 0.01)
            count_ok = expected_count is None or len(portfolio.positions) == expected_count
            
            metrics = {
                "total_time": elapsed,
                "input_tokens": usage.input_tokens if usage else 0,
                "output_tokens": usage.output_tokens if usage else 0,
                "total_tokens": usage.total_tokens if usage else 0,
                "tokens_per_second": (usage.total_tokens / elapsed) if usage and elapsed > 0 else 0,
                "tools_called": False,  # Agent 1 doesn't use tools
                "structured_output_ok": structured_ok,
                "correct": total_ok and count_ok,
                "difficulty": item.get("difficulty"),
                "actual_value": calculated_total,
                "expected_value": expected_total,
                "positions_count": len(portfolio.positions),
            }
            
            span.update(
                output={
                    "calculated_total": calculated_total,
                    "positions_count": len(portfolio.positions),
                    "correct": metrics["correct"],
                },
            )
            
        except Exception as e:
            elapsed = time.time() - start
            metrics = {
                "total_time": elapsed,
                "correct": False,
                "structured_output_ok": False,
                "tools_called": False,
                "error": str(e),
                "difficulty": item.get("difficulty"),
            }
            span.update(output={"error": str(e)})
        
        # Record scores
        record_scores(langfuse, trace_id, metrics)
        
        return metrics


async def run_agent_2_item(
    endpoint: str,
    item: Dict,
    langfuse: Langfuse,
    run_name: str,
) -> Dict[str, Any]:
    """Run Agent 2 (Financial Tools) on a single item."""
    
    model = get_model_for_endpoint(endpoint)
    
    # Import tools
    from examples.agent_2 import calculer_valeur_future, calculer_versement_mensuel
    
    agent = Agent(
        model,
        model_settings=ModelSettings(max_output_tokens=400),
        system_prompt="Expert calculs financiers. Utilise les outils pour les calculs précis.",
        tools=[calculer_valeur_future, calculer_versement_mensuel],
    )
    
    prompt = item["prompt"]
    expected = item.get("expected_output")
    expected_value = expected if isinstance(expected, (int, float)) else None
    
    with langfuse.start_as_current_span(
        name=f"agent_2_{endpoint}",
        input={"prompt": prompt[:500]},
        metadata={
            "endpoint": endpoint,
            "agent": "agent_2",
            "run_name": run_name,
            "difficulty": item.get("difficulty"),
            "category": item.get("category"),
        }
    ) as span:
        trace_id = span.trace_id
        
        start = time.time()
        try:
            result = await agent.run(prompt)
            elapsed = time.time() - start
            
            usage = result.usage()
            tool_calls = ToolCallDetector.extract_tool_calls(result) or []
            
            # Extract result value from tool output or text
            import re
            actual_value = None
            
            # Method 1: Look in tool call results from message parts
            for msg in result.all_messages():
                if hasattr(msg, 'parts'):
                    for part in msg.parts:
                        # ToolReturnPart contains the result
                        if hasattr(part, 'content') and part.content:
                            content = str(part.content)
                            # Look for "FV: 60,833.00€" pattern from calculer_valeur_future
                            fv_match = re.search(r'FV:\s*([\d,\s]+\.?\d*)\s*€', content)
                            if fv_match:
                                actual_value = float(fv_match.group(1).replace(',', '').replace(' ', ''))
                                break
            
            # Method 2: Fallback - extract from output text
            if actual_value is None:
                output_text = str(result.output)
                # Look for large numbers (financial results > 1000)
                numbers = re.findall(r'([\d,\s]+\.?\d*)\s*€?', output_text)
                for num_str in numbers:
                    try:
                        num = float(num_str.replace(',', '').replace(' ', ''))
                        if num > 1000:
                            actual_value = num
                            break
                    except:
                        continue
            
            # Check correctness (within 2% tolerance for financial calculations)
            correct = False
            if expected_value and actual_value:
                tolerance = max(1, expected_value * 0.02)  # 2% or at least 1€
                correct = abs(actual_value - expected_value) < tolerance
            
            metrics = {
                "total_time": elapsed,
                "input_tokens": usage.input_tokens if usage else 0,
                "output_tokens": usage.output_tokens if usage else 0,
                "total_tokens": usage.total_tokens if usage else 0,
                "tokens_per_second": (usage.total_tokens / elapsed) if usage and elapsed > 0 else 0,
                "tools_called": len(tool_calls) > 0,
                "tool_names": [tc.get("name") for tc in tool_calls],
                "structured_output_ok": len(tool_calls) > 0,  # For agent_2, success = tool was called
                "correct": correct,
                "difficulty": item.get("difficulty"),
                "actual_value": actual_value,
                "expected_value": expected_value,
            }
            
            span.update(
                output={
                    "tools_used": metrics["tool_names"],
                    "actual_value": actual_value,
                    "correct": correct,
                },
            )
            
        except Exception as e:
            elapsed = time.time() - start
            metrics = {
                "total_time": elapsed,
                "correct": False,
                "structured_output_ok": False,
                "tools_called": False,
                "error": str(e),
                "difficulty": item.get("difficulty"),
            }
            span.update(output={"error": str(e)})
        
        record_scores(langfuse, trace_id, metrics)
        return metrics


# ============================================================================
# EVALUATION RUNNER
# ============================================================================

async def run_evaluation(
    endpoints: List[str] = ["koyeb", "hf"],
    agents: List[str] = ["agent_1", "agent_2"],
    max_items_per_agent: int = 4,
):
    """Run comprehensive evaluation on both endpoints."""
    
    print("=" * 70)
    print("🔬 LANGFUSE EVALUATION: KOYEB vs HF SPACE")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Endpoints: {endpoints}")
    print(f"Agents: {agents}")
    print(f"Max items per agent: {max_items_per_agent}")
    
    langfuse = get_langfuse()
    
    # Datasets configuration
    datasets_config = {
        "agent_1": {
            "name": "open_finance_agent_1_portfolio",
            "items": AGENT_1_DATASET[:max_items_per_agent],
            "runner": run_agent_1_item,
        },
        "agent_2": {
            "name": "open_finance_agent_2_financial",
            "items": AGENT_2_DATASET[:max_items_per_agent],
            "runner": run_agent_2_item,
        },
    }
    
    results = {ep: {} for ep in endpoints}
    
    for agent_name in agents:
        if agent_name not in datasets_config:
            print(f"\n⚠️  Agent {agent_name} not configured, skipping")
            continue
        
        config = datasets_config[agent_name]
        
        print(f"\n{'='*70}")
        print(f"📊 EVALUATING: {agent_name.upper()}")
        print(f"{'='*70}")
        
        # Ensure dataset exists
        ensure_dataset(langfuse, config["name"], config["items"])
        
        for endpoint in endpoints:
            print(f"\n  🎯 Endpoint: {endpoint.upper()}")
            print(f"     URL: {ENDPOINTS[endpoint]['url']}")
            
            run_name = f"{agent_name}_{endpoint}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            endpoint_results = []
            
            for i, item in enumerate(config["items"]):
                print(f"     [{i+1}/{len(config['items'])}] {item.get('difficulty', 'unknown')} - ", end="", flush=True)
                
                try:
                    metrics = await config["runner"](endpoint, item, langfuse, run_name)
                    endpoint_results.append(metrics)
                    
                    status = "✅" if metrics.get("correct") else "❌"
                    tools = "🔧" if metrics.get("tools_called") else ""
                    print(f"{status} {metrics.get('total_time', 0):.2f}s {tools}")
                    
                except Exception as e:
                    print(f"❌ Error: {e}")
                    endpoint_results.append({"correct": False, "error": str(e)})
                
                await asyncio.sleep(0.5)  # Rate limiting
            
            results[endpoint][agent_name] = endpoint_results
            langfuse.flush()
    
    # Print summary
    print("\n" + "=" * 70)
    print("📊 EVALUATION SUMMARY")
    print("=" * 70)
    
    for agent_name in agents:
        if agent_name not in datasets_config:
            continue
        
        print(f"\n{agent_name.upper()}:")
        print(f"{'Endpoint':<12} | {'Correct':<10} | {'Avg Time':<12} | {'Avg Tok/s':<12} | {'Tools':<8}")
        print("-" * 60)
        
        for endpoint in endpoints:
            agent_results = results.get(endpoint, {}).get(agent_name, [])
            if not agent_results:
                continue
            
            correct = sum(1 for r in agent_results if r.get("correct"))
            total = len(agent_results)
            avg_time = sum(r.get("total_time", 0) for r in agent_results) / total if total else 0
            avg_tps = sum(r.get("tokens_per_second", 0) for r in agent_results) / total if total else 0
            tools_used = sum(1 for r in agent_results if r.get("tools_called"))
            
            print(f"{endpoint:<12} | {correct}/{total:<8} | {avg_time:<12.2f} | {avg_tps:<12.1f} | {tools_used}/{total}")
    
    print("\n" + "=" * 70)
    print("🔗 View detailed results: https://cloud.langfuse.com")
    print("   - Compare runs side-by-side in Experiments view")
    print("   - Filter by scores: correctness, latency, tools_used")
    print("=" * 70)
    
    langfuse.shutdown()
    return results


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Langfuse evaluation")
    parser.add_argument("--endpoints", nargs="+", default=["koyeb", "hf"], help="Endpoints to test")
    parser.add_argument("--agents", nargs="+", default=["agent_1", "agent_2"], help="Agents to test")
    parser.add_argument("--max-items", type=int, default=4, help="Max items per agent")
    
    args = parser.parse_args()
    
    asyncio.run(run_evaluation(
        endpoints=args.endpoints,
        agents=args.agents,
        max_items_per_agent=args.max_items,
    ))
