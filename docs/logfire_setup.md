# Logfire Setup: Alerts, Dashboards, and Metrics

This guide explains how to configure Logfire for monitoring Open Finance agents.

## Dashboard URL

https://logfire-eu.pydantic.dev/deal-ex-machina/open-finance

## Setting Up Alerts

### 1. Context Overflow Alert

**Purpose**: Get notified when an agent hits the context length limit.

1. Go to **Alerts** in Logfire sidebar
2. Click **Create Alert**
3. Configure:
   - **Name**: `Context Overflow Alert`
   - **Query**:
     ```sql
     SELECT COUNT(*) as overflow_count
     FROM records
     WHERE message = 'context_overflow'
       AND end_timestamp > NOW() - INTERVAL '5 minutes'
     ```
   - **Condition**: `overflow_count > 0`
   - **Notification**: Configure webhook (Slack, Discord, etc.)

### 2. Tool Call Anomaly Alert

**Purpose**: Detect agents making too many tool calls (potential loops).

1. Create alert with:
   - **Name**: `Tool Call Anomaly`
   - **Query**:
     ```sql
     SELECT 
       attributes->>'agent_name' as agent_name, 
       attributes->>'tool_calls' as tool_calls, 
       attributes->>'tool_anomaly_message' as message
     FROM spans
     WHERE span_name = 'agent_run_metrics'
       AND attributes->>'tool_anomaly_level' IN ('warning', 'critical')
       AND end_timestamp > NOW() - INTERVAL '10 minutes'
     LIMIT 10
     ```
   - **Condition**: Results > 0

### 3. High Latency Alert

**Purpose**: Alert when inference is slow.

```sql
SELECT 
  attributes->>'endpoint' as endpoint, 
  AVG((attributes->>'response_time_ms')::float) as avg_latency
FROM spans
WHERE span_name = 'inference_server_metrics'
  AND end_timestamp > NOW() - INTERVAL '5 minutes'
GROUP BY attributes->>'endpoint'
HAVING AVG((attributes->>'response_time_ms')::float) > 5000
```

## Creating Dashboards

### 1. Agent Performance Dashboard

Go to **Dashboards** → **Create Dashboard** → Add panels:

#### Panel: Runs by Agent and Endpoint
```sql
SELECT 
    attributes->>'agent_name' as agent_name,
    attributes->>'endpoint' as endpoint,
    COUNT(*) as total_runs,
    ROUND(AVG((attributes->>'elapsed_seconds')::float), 2) as avg_latency,
    SUM(CASE WHEN (attributes->>'success')::boolean THEN 1 ELSE 0 END) as successful,
    ROUND(AVG((attributes->>'total_tokens')::float), 0) as avg_tokens
FROM spans
WHERE span_name = 'agent_run_metrics'
  AND end_timestamp > NOW() - INTERVAL '24 hours'
GROUP BY attributes->>'agent_name', attributes->>'endpoint'
ORDER BY total_runs DESC
```

#### Panel: Success Rate Over Time
```sql
SELECT 
    DATE_TRUNC('hour', timestamp) as time_bucket,
    agent_name,
    ROUND(SUM(CASE WHEN success THEN 1 ELSE 0 END)::float / COUNT(*) * 100, 1) as success_rate
FROM spans
WHERE span_name = 'agent_run_metrics'
  AND timestamp > NOW() - INTERVAL '24 hours'
GROUP BY time_bucket, agent_name
ORDER BY time_bucket
```

### 2. Token Usage Dashboard

#### Panel: Token Usage by Agent
```sql
SELECT 
    agent_name,
    SUM(input_tokens) as total_input_tokens,
    SUM(output_tokens) as total_output_tokens,
    ROUND(AVG(tokens_per_second), 1) as avg_throughput,
    COUNT(*) as total_runs
FROM spans
WHERE span_name = 'agent_run_metrics'
  AND timestamp > NOW() - INTERVAL '24 hours'
GROUP BY agent_name
ORDER BY total_input_tokens DESC
```

#### Panel: Token Cost Estimation
```sql
SELECT 
    agent_name,
    SUM(input_tokens) / 1000 * 0.0001 as input_cost_usd,
    SUM(output_tokens) / 1000 * 0.0003 as output_cost_usd,
    (SUM(input_tokens) / 1000 * 0.0001 + SUM(output_tokens) / 1000 * 0.0003) as total_cost_usd
FROM spans
WHERE span_name = 'agent_run_metrics'
  AND timestamp > NOW() - INTERVAL '24 hours'
GROUP BY agent_name
```

### 3. Tool Call Monitoring Dashboard

#### Panel: Tool Calls by Agent
```sql
SELECT 
    agent_name,
    SUM(total_calls) as total_tool_calls,
    AVG(total_calls) as avg_calls_per_run,
    SUM(CASE WHEN is_anomaly THEN 1 ELSE 0 END) as anomaly_count
FROM spans
WHERE span_name = 'tool_call_stats'
  AND timestamp > NOW() - INTERVAL '24 hours'
GROUP BY agent_name
```

#### Panel: Tool Call Anomalies (Last 24h)
```sql
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
  AND timestamp > NOW() - INTERVAL '24 hours'
ORDER BY timestamp DESC
LIMIT 50
```

### 4. Inference Server Dashboard

#### Panel: Server Performance
```sql
SELECT 
    endpoint,
    COUNT(*) as total_requests,
    ROUND(AVG(response_time_ms), 0) as avg_latency_ms,
    ROUND(PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY response_time_ms), 0) as p95_latency_ms,
    ROUND(SUM(CASE WHEN success THEN 1 ELSE 0 END)::float / COUNT(*) * 100, 1) as success_rate
FROM spans
WHERE span_name = 'inference_server_metrics'
  AND timestamp > NOW() - INTERVAL '24 hours'
GROUP BY endpoint
```

#### Panel: Error Rate Over Time
```sql
SELECT 
    DATE_TRUNC('hour', timestamp) as time_bucket,
    endpoint,
    COUNT(*) as total,
    SUM(CASE WHEN NOT success THEN 1 ELSE 0 END) as errors,
    ROUND(SUM(CASE WHEN NOT success THEN 1 ELSE 0 END)::float / COUNT(*) * 100, 1) as error_rate
FROM spans
WHERE span_name = 'inference_server_metrics'
  AND timestamp > NOW() - INTERVAL '24 hours'
GROUP BY time_bucket, endpoint
ORDER BY time_bucket
```

## Using Standard Dashboards

Logfire provides built-in dashboards:

1. **LLM Tokens and Costs**: Automatic token tracking from PydanticAI instrumentation
2. **Exceptions**: View all errors grouped by type
3. **Web Server Metrics**: If running Gradio/FastAPI

Access via: Dashboard → Standard Dashboards

## Webhook Configuration

For alerts, configure webhooks:

### Slack
1. Create Slack App with Incoming Webhook
2. Copy webhook URL
3. In Logfire Alert → Add Action → Webhook
4. Paste Slack webhook URL

### Discord
1. Server Settings → Integrations → Webhooks
2. Create Webhook, copy URL
3. Use in Logfire Alert

## Programmatic Access

Use Logfire MCP Server or API for programmatic queries:

```python
# Example: Query via API
import httpx

response = httpx.post(
    "https://logfire-eu.pydantic.dev/api/v1/query",
    headers={"Authorization": f"Bearer {LOGFIRE_TOKEN}"},
    json={
        "query": "SELECT * FROM spans WHERE span_name = 'agent_run_metrics' LIMIT 10"
    }
)
```

## Key Span Names for Filtering

| Span Name | Description |
|-----------|-------------|
| `agent_run_metrics` | Comprehensive agent run stats |
| `tool_call_stats` | Tool call details |
| `inference_server_metrics` | Server performance |
| `context_overflow` | Context limit errors |
| `evaluation_run_*` | Evaluation sessions |
| `eval_item_*` | Individual eval items |

## Attributes for Filtering

Common attributes available on spans:

- `agent_name`: agent_1, agent_2, etc.
- `endpoint`: koyeb, hf, ollama
- `success`: true/false
- `tool_calls`: number of calls
- `tool_anomaly_level`: info, warning, critical
- `elapsed_seconds`: execution time
- `total_tokens`: input + output tokens
