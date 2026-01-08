# Langfuse API Update Required

## Issue

Our current integration code (`app/langfuse_integration.py`) uses the **Langfuse SDK v2 API**, but we have **Langfuse SDK v3** installed (3.11.2), which uses a completely different API based on OpenTelemetry.

## Current Code (v2 API - WRONG)

```python
# ❌ This doesn't exist in v3
trace = self.langfuse.trace(
    name=f"agent_{self.agent_name}",
    metadata={...}
)

# ❌ This doesn't exist in v3
span = trace.span(
    name="agent_execution",
    metadata={...}
)

# ❌ This doesn't exist in v3
trace.update(...)
```

## Correct v3 API

According to the [Langfuse documentation](https://langfuse.com/docs/observability/sdk/instrumentation):

1. **Traces are implicitly created** by the first (root) span or generation
2. **Use context managers** with `start_as_current_observation()`:
   ```python
   with langfuse.start_as_current_observation(
       as_type="span",
       name="agent_execution",
       input={"prompt": prompt}
   ) as span:
       # Run agent
       result = await agent.run(prompt)
       
       # Update span
       span.update(
           output={"result": str(result.output)},
           metadata={...}
       )
       
       # Update trace-level attributes
       span.update_trace(
           metadata={"agent_name": self.agent_name}
       )
   ```

3. **For manual observations** (when you need more control):
   ```python
   span = langfuse.start_observation(
       name="agent_execution",
       as_type="span"
   )
   # ... do work ...
   span.update(output={...})
   span.end()  # Must manually end
   ```

4. **For nested observations** (tool calls, LLM generations):
   ```python
   # Create child observation
   tool_span = span.start_observation(
       name="tool_call",
       as_type="tool"
   )
   tool_span.update(output={...})
   tool_span.end()
   ```

## Key Changes Needed

1. Replace `self.langfuse.trace()` → Use `start_as_current_observation()` or `start_observation()`
2. Replace `trace.span()` → Use `span.start_observation()` for children
3. Replace `trace.update()` → Use `span.update_trace()` for trace-level updates
4. Replace `span.end(output=...)` → Use `span.update(output=...)` then `span.end()` or use context manager

## Migration Strategy

Since we're using async code, we should use:
- `start_as_current_observation()` with async context managers (if supported)
- Or `start_observation()` with manual `.end()` calls

Let me update the integration code now.


