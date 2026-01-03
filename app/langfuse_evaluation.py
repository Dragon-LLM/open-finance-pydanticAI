"""Langfuse evaluation helpers for scoring and comparing agent runs."""

import logging
from typing import Any, Dict, List, Optional

from app.langfuse_config import get_langfuse_client

logger = logging.getLogger(__name__)


def score_trace(trace_id: str, scores: Dict[str, float]) -> bool:
    """
    Add scores to a Langfuse trace.
    
    Args:
        trace_id: Langfuse trace ID
        scores: Dictionary of scores (correctness, tool_usage_score, latency_score, overall_score)
                All scores should be between 0.0 and 1.0
        
    Returns:
        True if successful
    """
    langfuse = get_langfuse_client()
    if not langfuse:
        logger.warning("Langfuse not configured")
        return False
    
    try:
        # Get trace and add scores
        trace = langfuse.trace(id=trace_id)
        
        # Add scores as observations or metadata
        for score_name, score_value in scores.items():
            try:
                trace.score(
                    name=score_name,
                    value=score_value,
                )
            except Exception as e:
                logger.debug(f"Failed to add score {score_name}: {e}")
                # Fallback: add to metadata
                trace.update(metadata={f"score_{score_name}": score_value})
        
        logger.info(f"Added scores to trace {trace_id}: {scores}")
        return True
        
    except Exception as e:
        logger.error(f"Error scoring trace {trace_id}: {e}", exc_info=True)
        return False


def create_evaluation_run(dataset_name: str, agent_name: str) -> Optional[str]:
    """
    Create an evaluation run in Langfuse.
    
    Args:
        dataset_name: Name of the dataset being evaluated
        agent_name: Name of the agent being evaluated
        
    Returns:
        Run ID if successful, None otherwise
    """
    langfuse = get_langfuse_client()
    if not langfuse:
        logger.warning("Langfuse not configured")
        return None
    
    try:
        # Create a trace for the evaluation run
        trace = langfuse.trace(
            name=f"evaluation_run_{agent_name}",
            metadata={
                "dataset_name": dataset_name,
                "agent_name": agent_name,
                "evaluation_type": "dataset_evaluation",
            },
        )
        
        logger.info(f"Created evaluation run for {agent_name} on {dataset_name}: {trace.id}")
        return trace.id
        
    except Exception as e:
        logger.error(f"Error creating evaluation run: {e}", exc_info=True)
        return None


def compare_runs(run_ids: List[str]) -> Dict[str, Any]:
    """
    Compare different agent runs/versions.
    
    Args:
        run_ids: List of trace/run IDs to compare
        
    Returns:
        Comparison results with metrics
    """
    langfuse = get_langfuse_client()
    if not langfuse:
        logger.warning("Langfuse not configured")
        return {}
    
    try:
        comparison = {
            "run_ids": run_ids,
            "metrics": {},
        }
        
        # Fetch traces and extract metrics
        traces = []
        for run_id in run_ids:
            try:
                trace = langfuse.trace(id=run_id)
                traces.append({
                    "id": run_id,
                    "metadata": trace.metadata if hasattr(trace, 'metadata') else {},
                    "scores": {},  # Would extract from trace scores
                })
            except Exception as e:
                logger.debug(f"Failed to fetch trace {run_id}: {e}")
        
        comparison["traces"] = traces
        
        # Calculate aggregate metrics
        all_scores = {}
        for trace in traces:
            for score_name, score_value in trace.get("scores", {}).items():
                if score_name not in all_scores:
                    all_scores[score_name] = []
                all_scores[score_name].append(score_value)
        
        for score_name, values in all_scores.items():
            if values:
                comparison["metrics"][score_name] = {
                    "mean": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values),
                }
        
        logger.info(f"Compared {len(run_ids)} runs")
        return comparison
        
    except Exception as e:
        logger.error(f"Error comparing runs: {e}", exc_info=True)
        return {}


def export_evaluation_results(run_id: str) -> Dict[str, Any]:
    """
    Export evaluation results for analysis.
    
    Args:
        run_id: Evaluation run ID
        
    Returns:
        Dictionary with evaluation results
    """
    langfuse = get_langfuse_client()
    if not langfuse:
        logger.warning("Langfuse not configured")
        return {}
    
    try:
        trace = langfuse.trace(id=run_id)
        
        results = {
            "run_id": run_id,
            "metadata": trace.metadata if hasattr(trace, 'metadata') else {},
            "scores": {},  # Would extract from trace
            "spans": [],  # Would extract child spans
        }
        
        logger.info(f"Exported evaluation results for {run_id}")
        return results
        
    except Exception as e:
        logger.error(f"Error exporting evaluation results: {e}", exc_info=True)
        return {}

