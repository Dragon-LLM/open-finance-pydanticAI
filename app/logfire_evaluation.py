"""Logfire evaluation helpers for scoring and comparing agent runs.

Provides evaluation capabilities similar to Langfuse, using Logfire's
structured logging and span attributes for tracking scores and metrics.
"""

import logging
import time
from typing import Any, Dict, List, Optional

import logfire
from app.logfire_config import configure_logfire, is_logfire_enabled
from app.logfire_metrics import (
    record_agent_run,
    record_context_overflow,
    extract_tool_call_stats,
    record_tool_call_stats,
    detect_tool_call_anomaly,
    AlertLevel,
)

logger = logging.getLogger(__name__)


def score_span(
    span_id: str,
    scores: Dict[str, float],
    agent_name: str = "unknown",
    metadata: Optional[Dict[str, Any]] = None,
) -> bool:
    """
    Record evaluation scores for a span/trace in Logfire.
    
    Unlike Langfuse which has explicit score API, Logfire uses
    structured logging with attributes. Scores are recorded as
    a separate evaluation span linked to the original.
    
    Args:
        span_id: Identifier for the span being scored
        scores: Dictionary of scores (correctness, tool_usage, latency, etc.)
                All scores should be between 0.0 and 1.0
        agent_name: Name of the agent being evaluated
        metadata: Additional metadata to attach
        
    Returns:
        True if successful
    """
    if not is_logfire_enabled():
        logger.debug("Logfire not enabled, skipping score recording")
        return False
    
    try:
        # Record scores as a structured log with attributes
        with logfire.span(
            "evaluation_score",
            _span_id=span_id,
            agent_name=agent_name,
            **scores,
            **(metadata or {}),
        ):
            logfire.info(
                "Recorded evaluation scores",
                span_id=span_id,
                agent_name=agent_name,
                scores=scores,
            )
        
        logger.info(f"Recorded scores for span {span_id}: {scores}")
        return True
        
    except Exception as e:
        logger.error(f"Error recording scores: {e}", exc_info=True)
        return False


def create_evaluation_run(
    dataset_name: str,
    agent_name: str,
    run_id: Optional[str] = None,
) -> Optional[str]:
    """
    Create an evaluation run in Logfire.
    
    Creates a parent span for grouping evaluation results.
    
    Args:
        dataset_name: Name of the dataset being evaluated
        agent_name: Name of the agent being evaluated
        run_id: Optional custom run ID
        
    Returns:
        Run ID if successful, None otherwise
    """
    if not is_logfire_enabled():
        logger.debug("Logfire not enabled, skipping evaluation run creation")
        return None
    
    try:
        import uuid
        run_id = run_id or f"eval_{agent_name}_{uuid.uuid4().hex[:8]}"
        
        # Create evaluation run span
        with logfire.span(
            f"evaluation_run_{agent_name}",
            run_id=run_id,
            dataset_name=dataset_name,
            agent_name=agent_name,
            evaluation_type="dataset_evaluation",
        ):
            logfire.info(
                "Started evaluation run",
                run_id=run_id,
                dataset_name=dataset_name,
                agent_name=agent_name,
            )
        
        logger.info(f"Created evaluation run for {agent_name} on {dataset_name}: {run_id}")
        return run_id
        
    except Exception as e:
        logger.error(f"Error creating evaluation run: {e}", exc_info=True)
        return None


class LogfireEvaluator:
    """
    Evaluator class for running agent evaluations with Logfire tracing.
    
    Similar pattern to Langfuse evaluation but using Logfire's
    span-based approach.
    """
    
    def __init__(
        self,
        agent_name: str,
        dataset_name: str,
        run_id: Optional[str] = None,
    ):
        """
        Initialize the evaluator.
        
        Args:
            agent_name: Name of the agent to evaluate
            dataset_name: Name of the evaluation dataset
            run_id: Optional custom run ID
        """
        self.agent_name = agent_name
        self.dataset_name = dataset_name
        self.run_id = run_id or create_evaluation_run(dataset_name, agent_name)
        self.results: List[Dict[str, Any]] = []
        self._start_time = time.time()
    
    async def evaluate_item(
        self,
        agent,
        prompt: str,
        expected_output: Any,
        item_id: str,
        difficulty: str = "unknown",
        category: str = "unknown",
        endpoint: str = "koyeb",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate a single dataset item.
        
        Args:
            agent: PydanticAI Agent instance
            prompt: Input prompt
            expected_output: Expected output for comparison
            item_id: Unique identifier for this item
            difficulty: Difficulty level (easy, medium, hard)
            category: Category of the test case
            endpoint: Inference endpoint being used
            metadata: Additional metadata
            
        Returns:
            Evaluation result dictionary
        """
        result = {
            "item_id": item_id,
            "difficulty": difficulty,
            "category": category,
            "success": False,
            "scores": {},
            "error": None,
            "tool_stats": None,
        }
        
        start_time = time.time()
        
        try:
            with logfire.span(
                f"eval_item_{item_id}",
                run_id=self.run_id,
                agent_name=self.agent_name,
                endpoint=endpoint,
                item_id=item_id,
                difficulty=difficulty,
                category=category,
            ):
                # Run agent
                agent_result = await agent.run(prompt)
                elapsed = time.time() - start_time
                
                # Extract output
                output = agent_result.output if hasattr(agent_result, 'output') else agent_result
                
                # Extract usage
                usage = None
                input_tokens = 0
                output_tokens = 0
                try:
                    usage_func = getattr(agent_result, 'usage', None)
                    if callable(usage_func):
                        usage = usage_func()
                        input_tokens = getattr(usage, 'input_tokens', 0)
                        output_tokens = getattr(usage, 'output_tokens', 0)
                except Exception:
                    pass
                
                # Extract tool call statistics
                tool_stats = extract_tool_call_stats(agent_result, self.agent_name)
                result["tool_stats"] = {
                    "total_calls": tool_stats.total_calls,
                    "unique_tools": tool_stats.unique_tools,
                    "tool_names": tool_stats.tool_names,
                    "is_anomaly": tool_stats.is_anomaly,
                    "anomaly_level": tool_stats.anomaly_level.value,
                }
                
                # Record tool call stats
                record_tool_call_stats(self.agent_name, endpoint, tool_stats)
                
                # Calculate scores
                scores = self._calculate_scores(
                    output=output,
                    expected=expected_output,
                    elapsed=elapsed,
                    usage=usage,
                    agent_result=agent_result,
                    tool_stats=tool_stats,
                )
                
                result["success"] = True
                result["output"] = str(output)[:500]
                result["elapsed"] = elapsed
                result["scores"] = scores
                result["usage"] = {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                }
                
                # Record comprehensive agent run metrics
                record_agent_run(
                    agent_name=self.agent_name,
                    endpoint=endpoint,
                    success=True,
                    elapsed_seconds=elapsed,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    tool_calls=tool_stats.total_calls,
                    tool_names=tool_stats.tool_names,
                    metadata={
                        "item_id": item_id,
                        "difficulty": difficulty,
                        "category": category,
                        "run_id": self.run_id,
                    },
                )
                
                # Log evaluation result
                logfire.info(
                    "Evaluation item completed",
                    item_id=item_id,
                    success=True,
                    elapsed=elapsed,
                    tool_calls=tool_stats.total_calls,
                    tool_anomaly=tool_stats.is_anomaly,
                    **scores,
                )
                
        except Exception as e:
            elapsed = time.time() - start_time
            error_str = str(e)
            result["error"] = error_str
            
            # Check for context overflow
            if "maximum context length" in error_str.lower() or "8192 tokens" in error_str:
                # Extract token count from error if possible
                import re
                token_match = re.search(r'(\d+)\s*(?:input\s*)?tokens', error_str)
                input_tokens = int(token_match.group(1)) if token_match else 8500
                
                record_context_overflow(
                    agent_name=self.agent_name,
                    endpoint=endpoint,
                    input_tokens=input_tokens,
                    error_message=error_str,
                )
            
            # Record failed run
            record_agent_run(
                agent_name=self.agent_name,
                endpoint=endpoint,
                success=False,
                elapsed_seconds=elapsed,
                error=error_str[:500],
                metadata={
                    "item_id": item_id,
                    "difficulty": difficulty,
                    "run_id": self.run_id,
                },
            )
            
            logfire.error(
                "Evaluation item failed",
                item_id=item_id,
                error=error_str[:200],
            )
        
        self.results.append(result)
        return result
    
    def _calculate_scores(
        self,
        output: Any,
        expected: Any,
        elapsed: float,
        usage: Any,
        agent_result: Any,
        tool_stats: Optional[Any] = None,
    ) -> Dict[str, float]:
        """Calculate evaluation scores."""
        scores = {}
        
        # Latency score (1.0 = fast, 0.0 = slow)
        # Assume <2s is excellent, >10s is poor
        if elapsed < 2:
            scores["latency_score"] = 1.0
        elif elapsed > 10:
            scores["latency_score"] = 0.0
        else:
            scores["latency_score"] = 1.0 - (elapsed - 2) / 8
        
        # Token efficiency score
        if usage:
            total_tokens = getattr(usage, 'total_tokens', 0)
            # Assume <500 tokens is efficient, >2000 is inefficient
            if total_tokens < 500:
                scores["token_efficiency"] = 1.0
            elif total_tokens > 2000:
                scores["token_efficiency"] = 0.0
            else:
                scores["token_efficiency"] = 1.0 - (total_tokens - 500) / 1500
        
        # Tool usage from stats or extraction
        if tool_stats:
            tool_count = tool_stats.total_calls
            scores["tools_used"] = 1.0 if tool_count > 0 else 0.0
            scores["tool_call_count"] = float(tool_count)
            # Tool behavior score (1.0 = normal, 0.0 = anomaly)
            if tool_stats.anomaly_level == AlertLevel.CRITICAL:
                scores["tool_behavior_score"] = 0.0
            elif tool_stats.anomaly_level == AlertLevel.WARNING:
                scores["tool_behavior_score"] = 0.5
            else:
                scores["tool_behavior_score"] = 1.0
        else:
            # Fallback extraction
            tool_calls = []
            try:
                if hasattr(agent_result, 'all_messages'):
                    for msg in agent_result.all_messages():
                        if hasattr(msg, 'tool_calls') and msg.tool_calls:
                            tool_calls.extend(msg.tool_calls)
            except Exception:
                pass
            scores["tools_used"] = 1.0 if tool_calls else 0.0
            scores["tool_call_count"] = float(len(tool_calls))
            scores["tool_behavior_score"] = 1.0  # Assume normal without stats
        
        # Structured output validation
        scores["structured_output_ok"] = 1.0 if output is not None else 0.0
        
        # Basic correctness check (can be overridden per-agent)
        if expected is not None:
            if isinstance(expected, dict):
                # Check if expected keys are present in output
                if hasattr(output, '__dict__'):
                    output_dict = output.__dict__ if hasattr(output, '__dict__') else {}
                    matches = sum(1 for k in expected.keys() if k in str(output))
                    scores["correctness"] = matches / len(expected) if expected else 1.0
                else:
                    scores["correctness"] = 0.5  # Partial credit
            else:
                scores["correctness"] = 1.0 if str(expected) in str(output) else 0.0
        
        return scores
    
    def get_summary(self) -> Dict[str, Any]:
        """Get evaluation summary with aggregated metrics."""
        if not self.results:
            return {"error": "No results"}
        
        total = len(self.results)
        successful = sum(1 for r in self.results if r["success"])
        
        # Aggregate scores
        all_scores: Dict[str, List[float]] = {}
        for result in self.results:
            for score_name, score_value in result.get("scores", {}).items():
                if score_name not in all_scores:
                    all_scores[score_name] = []
                all_scores[score_name].append(score_value)
        
        avg_scores = {
            name: sum(values) / len(values) if values else 0.0
            for name, values in all_scores.items()
        }
        
        # By difficulty
        by_difficulty: Dict[str, Dict[str, Any]] = {}
        for result in self.results:
            diff = result.get("difficulty", "unknown")
            if diff not in by_difficulty:
                by_difficulty[diff] = {"total": 0, "successful": 0}
            by_difficulty[diff]["total"] += 1
            if result["success"]:
                by_difficulty[diff]["successful"] += 1
        
        summary = {
            "run_id": self.run_id,
            "agent_name": self.agent_name,
            "dataset_name": self.dataset_name,
            "total_items": total,
            "successful": successful,
            "success_rate": successful / total if total > 0 else 0.0,
            "average_scores": avg_scores,
            "by_difficulty": by_difficulty,
            "total_elapsed": time.time() - self._start_time,
        }
        
        # Log summary to Logfire
        if is_logfire_enabled():
            logfire.info(
                "Evaluation run completed",
                run_id=self.run_id,
                agent_name=self.agent_name,
                total_items=total,
                successful=successful,
                success_rate=summary["success_rate"],
                **avg_scores,
            )
        
        return summary


def compare_runs(run_ids: List[str]) -> Dict[str, Any]:
    """
    Compare different evaluation runs.
    
    Note: Logfire comparison is best done in the Logfire UI.
    This provides a summary with links.
    
    Args:
        run_ids: List of run IDs to compare
        
    Returns:
        Comparison summary with Logfire dashboard links
    """
    return {
        "run_ids": run_ids,
        "dashboard_url": "https://logfire-eu.pydantic.dev/deal-ex-machina/open-finance",
        "note": "Use Logfire dashboard for detailed trace comparison",
        "filter_hint": f"Filter by run_id in: {', '.join(run_ids)}",
    }


def export_evaluation_results(run_id: str) -> Dict[str, Any]:
    """
    Export evaluation results.
    
    Args:
        run_id: Evaluation run ID
        
    Returns:
        Export information with Logfire dashboard link
    """
    return {
        "run_id": run_id,
        "dashboard_url": f"https://logfire-eu.pydantic.dev/deal-ex-machina/open-finance",
        "filter": f"run_id={run_id}",
        "note": "Use Logfire dashboard to export traces as JSON",
    }
