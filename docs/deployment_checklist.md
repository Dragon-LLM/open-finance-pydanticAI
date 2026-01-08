# Deployment Checklist for Langfuse v3 API Fix

## Current Status

✅ **Critical Fix Applied**: Updated `app/langfuse_integration.py` from Langfuse SDK v2 API to v3 API

## Changes Made (Not Yet Committed)

1. **`app/langfuse_integration.py`** - Updated to use correct v3 API:
   - ❌ Old: `self.langfuse.trace()` → ✅ New: `self.langfuse.start_observation()`
   - ❌ Old: `trace.span()` → ✅ New: `span.start_observation()`
   - ❌ Old: `trace.update()` → ✅ New: `span.update_trace()`

2. **`app/langfuse_config.py`** - Minor updates

3. **`app/config.py`** - Langfuse configuration additions

4. **Documentation files** - Added compatibility and migration docs

## Deployment Steps

### 1. Commit Changes

```bash
git add app/langfuse_integration.py app/langfuse_config.py app/config.py
git commit -m "fix: Update Langfuse integration to use SDK v3 API

- Replace deprecated v2 API (trace(), trace.span()) with v3 API
- Use start_observation() for root span creation
- Use span.start_observation() for child observations
- Use span.update_trace() for trace-level updates
- Fixes compatibility with Langfuse SDK 3.11.2"
```

### 2. Merge to Main Branch

```bash
git checkout main
git merge prep-langfuse
git push origin main
```

### 3. Automatic Deployments

**Koyeb:**
- ✅ Automatically deploys via GitHub Actions when `main` branch is updated
- Workflow: `.github/workflows/deploy.yml`
- Triggers on push to `main` branch

**Hugging Face Spaces:**
- ✅ Automatically rebuilds and deploys on push to `main` branch
- Uses `Dockerfile` for deployment

### 4. Verify Deployment

After deployment, verify:
1. ✅ Application starts without errors
2. ✅ Langfuse traces are created correctly
3. ✅ Check Langfuse UI for traces
4. ✅ Verify no errors in logs related to Langfuse

## Important Notes

⚠️ **Before Deploying:**
- Ensure Langfuse environment variables are set in both platforms:
  - `LANGFUSE_PUBLIC_KEY`
  - `LANGFUSE_SECRET_KEY`
  - `LANGFUSE_BASE_URL` (or `LANGFUSE_HOST`)
  - `ENABLE_LANGFUSE=true` (optional, defaults to True)

⚠️ **Testing:**
- The old v2 API code would have failed at runtime
- The new v3 API code should work correctly
- Test with a simple agent run to verify traces appear in Langfuse

## Rollback Plan

If deployment fails:
1. Revert the commit
2. Push to main
3. Deployments will automatically rollback


