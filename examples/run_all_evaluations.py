#!/usr/bin/env python3
"""
Comprehensive Langfuse Evaluation for ALL Open Finance Agents.

Runs all 5 agents with specific evaluators:
- Agent 1: Portfolio extraction (structured output quality)
- Agent 2: Financial calculator (tool usage + calculation accuracy)
- Agent 3: Multi-step workflow (risk/tax analysis completeness)
- Agent 4: Option pricing (QuantLib tool usage + Greeks)
- Agent 5: SWIFT/ISO20022 conversion (message parsing + validation)

Usage:
    python examples/run_all_evaluations.py --endpoint hf --max-items 2
    python examples/run_all_evaluations.py --endpoint koyeb --agents agent_1 agent_2
"""

import asyncio
import time
import sys
import re
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass, field

sys.path.insert(0, str(Path(__file__).parent.parent))

from pydantic import BaseModel
from pydantic_ai import Agent, ModelSettings
from langfuse import Langfuse

from app.config import settings, ENDPOINTS
from app.models import get_model_for_endpoint
from app.mitigation_strategies import ToolCallDetector
from app.langfuse_datasets import (
    AGENT_1_DATASET, AGENT_2_DATASET, AGENT_3_DATASET,
    AGENT_4_DATASET, AGENT_5_DATASET
)


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class EvalResult:
    """Evaluation result for a single item."""
    agent: str
    endpoint: str
    difficulty: str
    latency: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    tokens_per_second: float = 0.0
    tools_called: bool = False
    tool_names: List[str] = field(default_factory=list)
    structured_output_ok: bool = False
    correct: bool = False
    error: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# LANGFUSE HELPERS
# ============================================================================

def get_langfuse() -> Langfuse:
    """Get configured Langfuse client from environment."""
    return Langfuse(
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key,
        host=settings.langfuse_host_resolved,
    )


def record_scores(langfuse: Langfuse, trace_id: str, result: EvalResult) -> None:
    """Record all metrics as Langfuse scores."""
    # Numeric scores
    for name, value in [
        ("latency_seconds", result.latency),
        ("tokens_per_second", result.tokens_per_second),
        ("input_tokens", result.input_tokens),
        ("output_tokens", result.output_tokens),
    ]:
        if value is not None:
            langfuse.create_score(trace_id=trace_id, name=name, value=float(value), data_type="NUMERIC")
    
    # Boolean scores
    for name, value in [
        ("tools_used", 1 if result.tools_called else 0),
        ("structured_output_ok", 1 if result.structured_output_ok else 0),
        ("correctness", 1 if result.correct else 0),
    ]:
        langfuse.create_score(trace_id=trace_id, name=name, value=value, data_type="BOOLEAN")
    
    # Categorical
    langfuse.create_score(trace_id=trace_id, name="difficulty", value=result.difficulty, data_type="CATEGORICAL")


# ============================================================================
# EVALUATORS
# ============================================================================

class Agent1Evaluator:
    """Evaluator for Agent 1: Portfolio extraction."""
    
    @staticmethod
    async def run(endpoint: str, item: Dict, langfuse: Langfuse, run_name: str) -> EvalResult:
        from examples.agent_1 import agent_1, Portfolio
        
        result = EvalResult(agent="agent_1", endpoint=endpoint, difficulty=item.get("difficulty", "unknown"))
        expected = item.get("expected_output", {})
        
        with langfuse.start_as_current_span(
            name=f"agent_1_{endpoint}",
            input={"prompt": item["prompt"][:500]},
            metadata={"endpoint": endpoint, "agent": "agent_1", "run_name": run_name, "difficulty": result.difficulty}
        ) as span:
            start = time.time()
            try:
                agent_result = await agent_1.run(item["prompt"], output_type=Portfolio)
                result.latency = time.time() - start
                
                portfolio = agent_result.output
                usage = agent_result.usage()
                
                if usage:
                    result.input_tokens = usage.input_tokens
                    result.output_tokens = usage.output_tokens
                    result.tokens_per_second = usage.total_tokens / result.latency if result.latency > 0 else 0
                
                # Evaluate structured output
                calculated_total = sum(p.quantite * p.prix_achat for p in portfolio.positions)
                result.structured_output_ok = (
                    len(portfolio.positions) > 0 and
                    all(p.symbole and p.quantite > 0 and p.prix_achat > 0 for p in portfolio.positions)
                )
                
                # Evaluate correctness
                expected_total = expected.get("total_value") if isinstance(expected, dict) else None
                expected_count = expected.get("positions_count") if isinstance(expected, dict) else None
                
                total_ok = expected_total is None or abs(calculated_total - expected_total) < max(1, expected_total * 0.01)
                count_ok = expected_count is None or len(portfolio.positions) == expected_count
                result.correct = total_ok and count_ok and result.structured_output_ok
                
                result.details = {
                    "calculated_total": calculated_total,
                    "expected_total": expected_total,
                    "positions_count": len(portfolio.positions),
                }
                
                span.update(output=result.details)
                
            except Exception as e:
                result.latency = time.time() - start
                result.error = str(e)
                span.update(output={"error": str(e)})
            
            record_scores(langfuse, span.trace_id, result)
        
        return result


class Agent2Evaluator:
    """Evaluator for Agent 2: Financial calculator with tools."""
    
    @staticmethod
    async def run(endpoint: str, item: Dict, langfuse: Langfuse, run_name: str) -> EvalResult:
        from examples.agent_2 import agent_2
        
        result = EvalResult(agent="agent_2", endpoint=endpoint, difficulty=item.get("difficulty", "unknown"))
        expected_value = item.get("expected_output")
        if isinstance(expected_value, dict):
            expected_value = expected_value.get("expected_return")
        
        with langfuse.start_as_current_span(
            name=f"agent_2_{endpoint}",
            input={"prompt": item["prompt"][:500]},
            metadata={"endpoint": endpoint, "agent": "agent_2", "run_name": run_name, "difficulty": result.difficulty}
        ) as span:
            start = time.time()
            try:
                agent_result = await agent_2.run(item["prompt"])
                result.latency = time.time() - start
                
                usage = agent_result.usage()
                if usage:
                    result.input_tokens = usage.input_tokens
                    result.output_tokens = usage.output_tokens
                    result.tokens_per_second = usage.total_tokens / result.latency if result.latency > 0 else 0
                
                # Check tool calls
                tool_calls = ToolCallDetector.extract_tool_calls(agent_result) or []
                result.tools_called = len(tool_calls) > 0
                result.tool_names = [tc.get("name", "unknown") for tc in tool_calls]
                result.structured_output_ok = result.tools_called
                
                # Extract result value from tool output
                actual_value = None
                for msg in agent_result.all_messages():
                    if hasattr(msg, 'parts'):
                        for part in msg.parts:
                            if hasattr(part, 'content') and part.content:
                                fv_match = re.search(r'FV:\s*([\d,\s]+\.?\d*)\s*€', str(part.content))
                                if fv_match:
                                    actual_value = float(fv_match.group(1).replace(',', '').replace(' ', ''))
                                    break
                
                # Check correctness
                if expected_value and actual_value:
                    tolerance = max(1, abs(expected_value) * 0.02)
                    result.correct = abs(actual_value - expected_value) < tolerance
                elif result.tools_called:
                    result.correct = True  # Tool was called successfully
                
                result.details = {
                    "tools_used": result.tool_names,
                    "actual_value": actual_value,
                    "expected_value": expected_value,
                }
                span.update(output=result.details)
                
            except Exception as e:
                result.latency = time.time() - start
                result.error = str(e)
                span.update(output={"error": str(e)})
            
            record_scores(langfuse, span.trace_id, result)
        
        return result


class Agent3Evaluator:
    """Evaluator for Agent 3: Multi-step workflow (risk/tax analysis)."""
    
    @staticmethod
    async def run(endpoint: str, item: Dict, langfuse: Langfuse, run_name: str) -> EvalResult:
        from examples.agent_3 import risk_analyst, AnalyseRisque, calculer_rendement_portfolio
        
        result = EvalResult(agent="agent_3", endpoint=endpoint, difficulty=item.get("difficulty", "unknown"))
        expected = item.get("expected_output", {})
        
        with langfuse.start_as_current_span(
            name=f"agent_3_{endpoint}",
            input={"prompt": item["prompt"][:500]},
            metadata={"endpoint": endpoint, "agent": "agent_3", "run_name": run_name, "difficulty": result.difficulty}
        ) as span:
            start = time.time()
            try:
                # Run risk analysis agent
                agent_result = await risk_analyst.run(item["prompt"], output_type=AnalyseRisque)
                result.latency = time.time() - start
                
                usage = agent_result.usage()
                if usage:
                    result.input_tokens = usage.input_tokens
                    result.output_tokens = usage.output_tokens
                    result.tokens_per_second = usage.total_tokens / result.latency if result.latency > 0 else 0
                
                # Check tool calls
                tool_calls = ToolCallDetector.extract_tool_calls(agent_result) or []
                result.tools_called = len(tool_calls) > 0
                result.tool_names = [tc.get("name", "unknown") for tc in tool_calls]
                
                # Evaluate structured output
                risk_analysis = agent_result.output
                result.structured_output_ok = (
                    hasattr(risk_analysis, 'niveau_risque') and 1 <= risk_analysis.niveau_risque <= 5 and
                    hasattr(risk_analysis, 'facteurs_risque') and len(risk_analysis.facteurs_risque) > 0
                )
                
                # Check expected outputs
                has_risk = expected.get("has_risk_analysis", False)
                result.correct = result.structured_output_ok if has_risk else True
                
                result.details = {
                    "niveau_risque": getattr(risk_analysis, 'niveau_risque', None),
                    "facteurs_count": len(getattr(risk_analysis, 'facteurs_risque', [])),
                    "tools_used": result.tool_names,
                }
                span.update(output=result.details)
                
            except Exception as e:
                result.latency = time.time() - start
                result.error = str(e)
                span.update(output={"error": str(e)})
            
            record_scores(langfuse, span.trace_id, result)
        
        return result


class Agent4Evaluator:
    """Evaluator for Agent 4: Option pricing with QuantLib."""
    
    @staticmethod
    async def run(endpoint: str, item: Dict, langfuse: Langfuse, run_name: str) -> EvalResult:
        from examples.agent_4 import agent_4, OptionPricingResult
        
        result = EvalResult(agent="agent_4", endpoint=endpoint, difficulty=item.get("difficulty", "unknown"))
        expected = item.get("expected_output", {})
        
        with langfuse.start_as_current_span(
            name=f"agent_4_{endpoint}",
            input={"prompt": item["prompt"][:500]},
            metadata={"endpoint": endpoint, "agent": "agent_4", "run_name": run_name, "difficulty": result.difficulty}
        ) as span:
            start = time.time()
            try:
                agent_result = await agent_4.run(item["prompt"])
                result.latency = time.time() - start
                
                usage = agent_result.usage()
                if usage:
                    result.input_tokens = usage.input_tokens
                    result.output_tokens = usage.output_tokens
                    result.tokens_per_second = usage.total_tokens / result.latency if result.latency > 0 else 0
                
                # Check tool calls
                tool_calls = ToolCallDetector.extract_tool_calls(agent_result) or []
                result.tools_called = len(tool_calls) > 0
                result.tool_names = [tc.get("name", "unknown") for tc in tool_calls]
                
                # Evaluate structured output
                pricing = agent_result.output
                result.structured_output_ok = (
                    hasattr(pricing, 'option_price') and pricing.option_price > 0
                )
                
                # Check Greeks if expected
                has_greeks = expected.get("has_greeks", False)
                if has_greeks:
                    result.correct = (
                        result.structured_output_ok and
                        hasattr(pricing, 'delta') and
                        hasattr(pricing, 'gamma')
                    )
                else:
                    result.correct = result.structured_output_ok and result.tools_called
                
                result.details = {
                    "option_price": getattr(pricing, 'option_price', None),
                    "delta": getattr(pricing, 'delta', None),
                    "tools_used": result.tool_names,
                }
                span.update(output=result.details)
                
            except Exception as e:
                result.latency = time.time() - start
                result.error = str(e)
                span.update(output={"error": str(e)})
            
            record_scores(langfuse, span.trace_id, result)
        
        return result


class Agent5Evaluator:
    """Evaluator for Agent 5: SWIFT/ISO20022 conversion."""
    
    @staticmethod
    async def run(endpoint: str, item: Dict, langfuse: Langfuse, run_name: str) -> EvalResult:
        from examples.agent_5 import agent_5
        
        result = EvalResult(agent="agent_5", endpoint=endpoint, difficulty=item.get("difficulty", "unknown"))
        expected = item.get("expected_output", {})
        
        with langfuse.start_as_current_span(
            name=f"agent_5_{endpoint}",
            input={"prompt": item["prompt"][:500]},
            metadata={"endpoint": endpoint, "agent": "agent_5", "run_name": run_name, "difficulty": result.difficulty}
        ) as span:
            start = time.time()
            try:
                agent_result = await agent_5.run(item["prompt"])
                result.latency = time.time() - start
                
                usage = agent_result.usage()
                if usage:
                    result.input_tokens = usage.input_tokens
                    result.output_tokens = usage.output_tokens
                    result.tokens_per_second = usage.total_tokens / result.latency if result.latency > 0 else 0
                
                # Check tool calls
                tool_calls = ToolCallDetector.extract_tool_calls(agent_result) or []
                result.tools_called = len(tool_calls) > 0
                result.tool_names = [tc.get("name", "unknown") for tc in tool_calls]
                
                # Check for ISO 20022 markers in output
                output_text = str(agent_result.output)
                has_iso = "pacs" in output_text.lower() or "Document" in output_text or "xml" in output_text.lower()
                
                result.structured_output_ok = result.tools_called or has_iso
                result.correct = expected.get("has_iso20022", False) and result.structured_output_ok
                
                result.details = {
                    "has_iso_markers": has_iso,
                    "tools_used": result.tool_names,
                }
                span.update(output=result.details)
                
            except Exception as e:
                result.latency = time.time() - start
                result.error = str(e)
                span.update(output={"error": str(e)})
            
            record_scores(langfuse, span.trace_id, result)
        
        return result


# ============================================================================
# MAIN EVALUATION RUNNER
# ============================================================================

EVALUATORS = {
    "agent_1": (Agent1Evaluator.run, AGENT_1_DATASET, "Portfolio Extraction"),
    "agent_2": (Agent2Evaluator.run, AGENT_2_DATASET, "Financial Calculator"),
    "agent_3": (Agent3Evaluator.run, AGENT_3_DATASET, "Multi-Step Workflow"),
    "agent_4": (Agent4Evaluator.run, AGENT_4_DATASET, "Option Pricing"),
    "agent_5": (Agent5Evaluator.run, AGENT_5_DATASET, "SWIFT/ISO20022"),
}


async def run_all_evaluations(
    endpoint: str = "hf",
    agents: List[str] = None,
    max_items: int = 3,
):
    """Run evaluations for all agents."""
    
    if agents is None:
        agents = list(EVALUATORS.keys())
    
    print("=" * 80)
    print(f"🔬 LANGFUSE EVALUATION - {endpoint.upper()}")
    print("=" * 80)
    print(f"Endpoint: {ENDPOINTS[endpoint]['url']}")
    print(f"Agents: {', '.join(agents)}")
    print(f"Max items per agent: {max_items}")
    print(f"Timestamp: {datetime.now().isoformat()}")
    
    langfuse = get_langfuse()
    all_results: Dict[str, List[EvalResult]] = {}
    
    for agent_name in agents:
        if agent_name not in EVALUATORS:
            print(f"\n⚠️  Unknown agent: {agent_name}")
            continue
        
        evaluator, dataset, description = EVALUATORS[agent_name]
        items = dataset[:max_items]
        
        print(f"\n{'─'*80}")
        print(f"🤖 {agent_name.upper()}: {description}")
        print(f"{'─'*80}")
        
        run_name = f"{agent_name}_{endpoint}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        results = []
        
        for i, item in enumerate(items):
            diff = item.get("difficulty", "?")
            print(f"  [{i+1}/{len(items)}] {diff:<8} ", end="", flush=True)
            
            try:
                result = await evaluator(endpoint, item, langfuse, run_name)
                results.append(result)
                
                status = "✅" if result.correct else "❌"
                tools = f"🔧{len(result.tool_names)}" if result.tools_called else "  "
                struct = "📊" if result.structured_output_ok else "  "
                print(f"{status} {tools} {struct} {result.latency:.1f}s")
                
                if result.error:
                    print(f"           └─ Error: {result.error[:60]}...")
                    
            except Exception as e:
                print(f"❌ Exception: {e}")
                results.append(EvalResult(agent=agent_name, endpoint=endpoint, difficulty=diff, error=str(e)))
            
            await asyncio.sleep(0.3)
        
        all_results[agent_name] = results
        langfuse.flush()
    
    # Print summary
    print("\n" + "=" * 80)
    print("📊 EVALUATION SUMMARY")
    print("=" * 80)
    print(f"\n{'Agent':<12} │ {'N':>3} │ {'Correct':>8} │ {'Tools':>6} │ {'Struct':>6} │ {'Avg Lat':>8} │ {'Avg Tok/s':>10}")
    print("─" * 80)
    
    for agent_name, results in all_results.items():
        if not results:
            continue
        n = len(results)
        correct = sum(1 for r in results if r.correct)
        tools = sum(1 for r in results if r.tools_called)
        struct = sum(1 for r in results if r.structured_output_ok)
        avg_lat = sum(r.latency for r in results) / n
        avg_tps = sum(r.tokens_per_second for r in results) / n
        
        print(f"{agent_name:<12} │ {n:>3} │ {correct:>3}/{n:<3}  │ {tools:>3}/{n:<2} │ {struct:>3}/{n:<2} │ {avg_lat:>7.1f}s │ {avg_tps:>10.1f}")
    
    print("\n" + "=" * 80)
    print(f"🔗 View in Langfuse: https://cloud.langfuse.com")
    print("=" * 80)
    
    langfuse.shutdown()
    return all_results


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Langfuse evaluations for all agents")
    parser.add_argument("--endpoint", default="hf", choices=["hf", "koyeb"], help="Inference endpoint")
    parser.add_argument("--agents", nargs="+", default=None, help="Agents to evaluate (default: all)")
    parser.add_argument("--max-items", type=int, default=3, help="Max items per agent")
    
    args = parser.parse_args()
    
    asyncio.run(run_all_evaluations(
        endpoint=args.endpoint,
        agents=args.agents,
        max_items=args.max_items,
    ))
