"""Logfire metrics and alerting helpers.

Provides structured metrics for:
- Agent performance tracking
- Inference server monitoring
- Tool call anomaly detection
- Context overflow alerts
"""

import logging
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from enum import Enum

import logfire
from app.logfire_config import is_logfire_enabled

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS FOR ANOMALY DETECTION
# =============================================================================

class AlertLevel(Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


# Tool call thresholds per agent
# Note: PydanticAI structured output (output_type) may count as internal "tool call"
TOOL_CALL_THRESHOLDS = {
    "agent_1": {"normal": 1, "warning": 3, "critical": 5},   # Structured output (Portfolio)
    "agent_2": {"normal": 1, "warning": 5, "critical": 15},  # Calculator tools (may retry)
    "agent_3": {"normal": 3, "warning": 8, "critical": 20},  # Multi-agent workflow
    "agent_4": {"normal": 2, "warning": 5, "critical": 10},  # Pricing tool + structured
    "agent_5": {"normal": 2, "warning": 5, "critical": 10},  # Conversion tool + structured
    "default": {"normal": 2, "warning": 5, "critical": 15},
}

# Context length thresholds
CONTEXT_WARNING_THRESHOLD = 6000  # Warn at 6000 tokens
CONTEXT_CRITICAL_THRESHOLD = 7500  # Critical at 7500 tokens (8192 max)


# =============================================================================
# METRICS RECORDING
# =============================================================================

def record_agent_run(
    agent_name: str,
    endpoint: str,
    success: bool,
    elapsed_seconds: float,
    input_tokens: int = 0,
    output_tokens: int = 0,
    tool_calls: int = 0,
    tool_names: Optional[List[str]] = None,
    error: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Record metrics for an agent run.
    
    This creates a structured span with all metrics as attributes,
    enabling filtering and aggregation in Logfire dashboards.
    
    Args:
        agent_name: Name of the agent (e.g., "agent_1", "agent_2")
        endpoint: Inference endpoint (e.g., "koyeb", "hf", "ollama")
        success: Whether the run completed successfully
        elapsed_seconds: Total execution time
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        tool_calls: Number of tool calls made
        tool_names: List of tool names called
        error: Error message if failed
        metadata: Additional metadata
    """
    if not is_logfire_enabled():
        return
    
    total_tokens = input_tokens + output_tokens
    tokens_per_second = total_tokens / elapsed_seconds if elapsed_seconds > 0 else 0
    
    # Check for anomalies
    tool_anomaly = detect_tool_call_anomaly(agent_name, tool_calls)
    context_alert = check_context_usage(total_tokens)
    
    # Determine overall status
    status = "success" if success else "error"
    if tool_anomaly["level"] in [AlertLevel.WARNING, AlertLevel.CRITICAL]:
        status = "warning" if success else "error"
    
    # Record as structured span
    with logfire.span(
        "agent_run_metrics",
        # Core identifiers (for filtering)
        agent_name=agent_name,
        endpoint=endpoint,
        status=status,
        # Performance metrics
        elapsed_seconds=elapsed_seconds,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        tokens_per_second=round(tokens_per_second, 2),
        # Tool metrics
        tool_calls=tool_calls,
        tool_names=tool_names or [],
        tool_anomaly_level=tool_anomaly["level"].value,
        tool_anomaly_message=tool_anomaly.get("message", ""),
        # Context metrics
        context_usage_percent=round((total_tokens / 8192) * 100, 1),
        context_alert_level=context_alert["level"].value,
        # Error info
        error=error,
        success=success,
        # Additional metadata
        **(metadata or {}),
    ):
        # Log based on status
        if not success:
            logfire.error(
                f"Agent run failed: {agent_name}",
                agent_name=agent_name,
                endpoint=endpoint,
                error=error,
            )
        elif tool_anomaly["level"] == AlertLevel.CRITICAL:
            logfire.error(
                f"Tool call anomaly detected: {tool_anomaly['message']}",
                agent_name=agent_name,
                tool_calls=tool_calls,
            )
        elif tool_anomaly["level"] == AlertLevel.WARNING:
            logfire.warn(
                f"Elevated tool calls: {tool_anomaly['message']}",
                agent_name=agent_name,
                tool_calls=tool_calls,
            )
        else:
            logfire.info(
                f"Agent run completed: {agent_name}",
                agent_name=agent_name,
                endpoint=endpoint,
                elapsed_seconds=elapsed_seconds,
                total_tokens=total_tokens,
            )


def record_context_overflow(
    agent_name: str,
    endpoint: str,
    input_tokens: int,
    max_tokens: int = 8192,
    error_message: str = "",
) -> None:
    """
    Record a context overflow error.
    
    This creates a specific span for context overflow alerts,
    enabling targeted alerting in Logfire.
    
    Args:
        agent_name: Name of the agent
        endpoint: Inference endpoint
        input_tokens: Number of tokens that caused overflow
        max_tokens: Maximum context length
        error_message: Full error message
    """
    if not is_logfire_enabled():
        return
    
    overflow_amount = input_tokens - max_tokens
    
    # Log as error with structured data
    logfire.error(
        "context_overflow",
        agent_name=agent_name,
        endpoint=endpoint,
        input_tokens=input_tokens,
        max_tokens=max_tokens,
        overflow_amount=overflow_amount,
        overflow_percent=round((overflow_amount / max_tokens) * 100, 1),
        error_message=error_message[:500],
        alert_type="context_overflow",
        severity="critical",
    )
    
    logger.error(
        f"Context overflow: {agent_name} on {endpoint} - "
        f"{input_tokens}/{max_tokens} tokens ({overflow_amount} over)"
    )


def record_inference_server_metrics(
    endpoint: str,
    response_time_ms: float,
    status_code: int,
    model_name: str,
    success: bool,
    error: Optional[str] = None,
) -> None:
    """
    Record inference server performance metrics.
    
    Args:
        endpoint: Server endpoint name
        response_time_ms: Response time in milliseconds
        status_code: HTTP status code
        model_name: Model being served
        success: Whether request succeeded
        error: Error message if failed
    """
    if not is_logfire_enabled():
        return
    
    # Determine latency category
    if response_time_ms < 1000:
        latency_category = "fast"
    elif response_time_ms < 3000:
        latency_category = "normal"
    elif response_time_ms < 10000:
        latency_category = "slow"
    else:
        latency_category = "very_slow"
    
    with logfire.span(
        "inference_server_metrics",
        endpoint=endpoint,
        model_name=model_name,
        response_time_ms=round(response_time_ms, 2),
        latency_category=latency_category,
        status_code=status_code,
        success=success,
        error=error,
    ):
        if not success:
            logfire.warn(
                f"Inference server error: {endpoint}",
                endpoint=endpoint,
                status_code=status_code,
                error=error,
            )
        elif latency_category in ["slow", "very_slow"]:
            logfire.warn(
                f"Slow inference: {endpoint}",
                endpoint=endpoint,
                response_time_ms=response_time_ms,
            )


# =============================================================================
# ANOMALY DETECTION
# =============================================================================

def detect_tool_call_anomaly(
    agent_name: str,
    tool_calls: int,
) -> Dict[str, Any]:
    """
    Detect if tool call count is anomalous for the agent.
    
    Returns:
        Dictionary with level (AlertLevel) and message
    """
    thresholds = TOOL_CALL_THRESHOLDS.get(
        agent_name,
        TOOL_CALL_THRESHOLDS["default"]
    )
    
    if tool_calls >= thresholds["critical"]:
        return {
            "level": AlertLevel.CRITICAL,
            "message": f"Critical: {tool_calls} tool calls (threshold: {thresholds['critical']})",
            "expected": thresholds["normal"],
            "actual": tool_calls,
        }
    elif tool_calls >= thresholds["warning"]:
        return {
            "level": AlertLevel.WARNING,
            "message": f"Warning: {tool_calls} tool calls (threshold: {thresholds['warning']})",
            "expected": thresholds["normal"],
            "actual": tool_calls,
        }
    else:
        return {
            "level": AlertLevel.INFO,
            "message": "Normal tool call count",
            "expected": thresholds["normal"],
            "actual": tool_calls,
        }


def check_context_usage(total_tokens: int) -> Dict[str, Any]:
    """
    Check if context usage is approaching limits.
    
    Returns:
        Dictionary with level and details
    """
    if total_tokens >= CONTEXT_CRITICAL_THRESHOLD:
        return {
            "level": AlertLevel.CRITICAL,
            "message": f"Critical context usage: {total_tokens}/8192 tokens",
            "usage_percent": round((total_tokens / 8192) * 100, 1),
        }
    elif total_tokens >= CONTEXT_WARNING_THRESHOLD:
        return {
            "level": AlertLevel.WARNING,
            "message": f"High context usage: {total_tokens}/8192 tokens",
            "usage_percent": round((total_tokens / 8192) * 100, 1),
        }
    else:
        return {
            "level": AlertLevel.INFO,
            "message": "Normal context usage",
            "usage_percent": round((total_tokens / 8192) * 100, 1),
        }


# =============================================================================
# TOOL CALL TRACKING
# =============================================================================

@dataclass
class ToolCallStats:
    """Statistics for tool calls in an agent run."""
    total_calls: int = 0
    unique_tools: int = 0
    tool_names: List[str] = None
    calls_per_tool: Dict[str, int] = None
    is_anomaly: bool = False
    anomaly_level: AlertLevel = AlertLevel.INFO
    anomaly_message: str = ""
    
    def __post_init__(self):
        if self.tool_names is None:
            self.tool_names = []
        if self.calls_per_tool is None:
            self.calls_per_tool = {}


def extract_tool_call_stats(agent_result, agent_name: str) -> ToolCallStats:
    """
    Extract tool call statistics from an agent result.
    
    Args:
        agent_result: PydanticAI agent run result
        agent_name: Name of the agent for threshold lookup
        
    Returns:
        ToolCallStats with extracted metrics
    """
    stats = ToolCallStats()
    
    try:
        if hasattr(agent_result, 'all_messages'):
            for msg in agent_result.all_messages():
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    for tool_call in msg.tool_calls:
                        tool_name = getattr(tool_call, 'name', 'unknown')
                        stats.total_calls += 1
                        stats.tool_names.append(tool_name)
                        stats.calls_per_tool[tool_name] = stats.calls_per_tool.get(tool_name, 0) + 1
        
        stats.unique_tools = len(stats.calls_per_tool)
        
        # Check for anomaly
        anomaly = detect_tool_call_anomaly(agent_name, stats.total_calls)
        stats.is_anomaly = anomaly["level"] in [AlertLevel.WARNING, AlertLevel.CRITICAL]
        stats.anomaly_level = anomaly["level"]
        stats.anomaly_message = anomaly["message"]
        
    except Exception as e:
        logger.debug(f"Error extracting tool stats: {e}")
    
    return stats


def record_tool_call_stats(
    agent_name: str,
    endpoint: str,
    stats: ToolCallStats,
) -> None:
    """Record tool call statistics to Logfire."""
    if not is_logfire_enabled():
        return
    
    with logfire.span(
        "tool_call_stats",
        agent_name=agent_name,
        endpoint=endpoint,
        total_calls=stats.total_calls,
        unique_tools=stats.unique_tools,
        tool_names=stats.tool_names,
        calls_per_tool=stats.calls_per_tool,
        is_anomaly=stats.is_anomaly,
        anomaly_level=stats.anomaly_level.value,
        anomaly_message=stats.anomaly_message,
    ):
        if stats.is_anomaly:
            logfire.warn(
                f"Tool call anomaly: {stats.anomaly_message}",
                agent_name=agent_name,
                total_calls=stats.total_calls,
            )


# =============================================================================
# DASHBOARD QUERIES (for reference in Logfire UI)
# =============================================================================

DASHBOARD_QUERIES = {
    "agent_runs_by_endpoint": """
        -- Agent runs grouped by endpoint
        SELECT 
            agent_name,
            endpoint,
            COUNT(*) as total_runs,
            AVG(elapsed_seconds) as avg_latency,
            SUM(CASE WHEN success THEN 1 ELSE 0 END) as successful,
            AVG(total_tokens) as avg_tokens
        FROM spans
        WHERE span_name = 'agent_run_metrics'
        GROUP BY agent_name, endpoint
        ORDER BY total_runs DESC
    """,
    
    "tool_call_anomalies": """
        -- Tool call anomalies
        SELECT 
            timestamp,
            agent_name,
            endpoint,
            tool_calls,
            tool_anomaly_level,
            tool_anomaly_message
        FROM spans
        WHERE span_name = 'agent_run_metrics'
          AND tool_anomaly_level IN ('warning', 'critical')
        ORDER BY timestamp DESC
        LIMIT 100
    """,
    
    "context_overflows": """
        -- Context overflow errors
        SELECT 
            timestamp,
            agent_name,
            endpoint,
            input_tokens,
            overflow_amount,
            error_message
        FROM spans
        WHERE span_name = 'context_overflow'
        ORDER BY timestamp DESC
        LIMIT 50
    """,
    
    "inference_server_performance": """
        -- Inference server performance by endpoint
        SELECT 
            endpoint,
            model_name,
            COUNT(*) as total_requests,
            AVG(response_time_ms) as avg_latency_ms,
            PERCENTILE(response_time_ms, 0.95) as p95_latency_ms,
            SUM(CASE WHEN success THEN 1 ELSE 0 END) / COUNT(*) * 100 as success_rate
        FROM spans
        WHERE span_name = 'inference_server_metrics'
        GROUP BY endpoint, model_name
    """,
    
    "token_usage_by_agent": """
        -- Token usage breakdown by agent
        SELECT 
            agent_name,
            SUM(input_tokens) as total_input_tokens,
            SUM(output_tokens) as total_output_tokens,
            AVG(tokens_per_second) as avg_throughput,
            COUNT(*) as total_runs
        FROM spans
        WHERE span_name = 'agent_run_metrics'
        GROUP BY agent_name
    """,
}


def print_dashboard_queries():
    """Print dashboard queries for copy-paste into Logfire UI."""
    print("\n" + "="*60)
    print("LOGFIRE DASHBOARD QUERIES")
    print("Copy these into Logfire Dashboard SQL editor")
    print("="*60)
    
    for name, query in DASHBOARD_QUERIES.items():
        print(f"\n--- {name} ---")
        print(query.strip())
    
    print("\n" + "="*60)
