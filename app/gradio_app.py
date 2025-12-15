"""
Gradio GUI for Open Finance PydanticAI Agents.

Clean tabbed interface with server health checks and tool usage tracking.
"""

import asyncio
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import gradio as gr
import httpx

# Add project root to Python path for imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from app.config import Settings, ENDPOINTS


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_endpoint_from_model(model) -> str:
    """Determine which endpoint a model is using by comparing base_url or model type.
    
    Args:
        model: PydanticAI model instance
        
    Returns:
        Endpoint name ("koyeb", "hf", "llm_pro_finance", "ollama") or "unknown"
    """
    try:
        # Check if it's an OllamaModel from pydanticai-ollama
        model_type_name = type(model).__name__
        if model_type_name == "OllamaModel":
            return "ollama"
        
        # For OpenAI-compatible models, check provider base_url
        if hasattr(model, 'provider') and hasattr(model.provider, 'base_url'):
            base_url = model.provider.base_url
            
            # Remove /v1 or /api suffix for comparison
            base_url_clean = base_url.rstrip('/v1').rstrip('/api').rstrip('/')
            
            # Compare with each endpoint's URL
            for endpoint_name, endpoint_config in ENDPOINTS.items():
                endpoint_url = endpoint_config.get("url", "").rstrip('/')
                api_path = endpoint_config.get("api_path", "/v1").lstrip('/')
                
                # Check if base_url matches this endpoint
                if endpoint_name == "llm_pro_finance":
                    # LLM Pro Finance uses /api path
                    if base_url_clean == endpoint_url or base_url == f"{endpoint_url}/api":
                        return endpoint_name
                elif endpoint_name == "ollama":
                    # Ollama uses /v1 path (for OpenAI-compatible mode)
                    if base_url_clean == endpoint_url or base_url == f"{endpoint_url}/v1":
                        return endpoint_name
                else:
                    # Koyeb and HF use /v1 path
                    if base_url_clean == endpoint_url or base_url == f"{endpoint_url}/v1":
                        return endpoint_name
    except Exception as e:
        print(f"[WARNING] Could not determine endpoint from model: {e}")
    
    return "unknown"


def get_local_ollama_models(max_items: int = 4) -> Tuple[List[str], int]:
    """List local Ollama models from the default manifests directory."""
    base_dir = Path.home() / ".ollama" / "models" / "manifests" / "registry.ollama.ai" / "library"
    try:
        names = [
            d.name
            for d in base_dir.iterdir()
            if d.is_dir()
        ]
        names = sorted(names)
        return names[:max_items], len(names)
    except Exception:
        return [], 0


# ============================================================================
# GLOBAL STATE
# ============================================================================

results_store: Dict[str, Any] = {}

# Agent descriptions
AGENT_INFO = {
    "Agent 1": {
        "title": "Portfolio Extractor",
        "description": "Extracts structured portfolio data from natural language text. Identifies stock symbols, quantities, prices, and calculates total values.",
        "default_input": "Extrais le portfolio: 50 AIR.PA a 120 euros, 30 SAN.PA a 85 euros, 100 TTE.PA a 55 euros",
    },
    "Agent 2": {
        "title": "Financial Calculator", 
        "description": "Performs precise financial calculations using numpy-financial. Computes future values, loan payments, portfolio performance, and interest rates. Ask ONE question at a time.",
        "default_input": "J'ai 50000 euros a placer a 4% par an pendant 10 ans. Quelle sera la valeur finale?",
    },
    "Agent 3": {
        "title": "Risk and Tax Advisor",
        "description": "Multi-agent workflow combining risk analysis and tax optimization. Evaluates portfolio risk levels and provides tax-efficient recommendations.",
        "default_input": "Analyse le risque d'un portfolio: 40% actions, 30% obligations, 20% immobilier, 10% autres. Investissement 100k euros, horizon 30 ans.",
    },
    "Agent 4": {
        "title": "Option Pricing",
        "description": "Prices European options using QuantLib Black-Scholes model. Computes option prices and Greeks (delta, gamma, theta, vega, rho).",
        "default_input": "Prix d'un call europeen: Spot=100, Strike=105, Maturite=0.5 an, Taux=0.02, Volatilite=0.25, Dividende=0.01",
    },
    "Agent 5 - Convert": {
        "title": "Message Conversion",
        "description": "Bidirectional: SWIFT MT103 ↔ ISO 20022 pacs.008. Use convertir_swift_vers_iso20022 or convertir_iso20022_vers_swift",
        "default_input": """Convertis ce SWIFT MT103 vers ISO 20022:

{1:F01BANKFRPPAXXX1234567890}
{4:
:20:REF123
:32A:240101EUR1000,00
:50A:/FR1420041010050500013M02606
COMPAGNIE ABC
:59:/DE89370400440532013000
COMPAGNIE XYZ
-}

Pour la direction inverse (ISO→SWIFT), fournis un XML ISO 20022 et demande la conversion vers SWIFT.""",
    },
    "Agent 5 - Validate": {
        "title": "Message Validation",
        "description": "Validates SWIFT MT and ISO 20022 message structure, format, and required fields",
        "default_input": """{1:F01BANKFRPPAXXX1234567890}
{4:
:20:REF123
:32A:240101EUR1000,00
-}""",
    },
    "Agent 5 - Risk": {
        "title": "Risk Assessment",
        "description": "AML/KYC risk scoring for financial messages. Evaluates transaction patterns and risk indicators.",
        "default_input": """{1:F01BANKUSAAXXX1234567890}
{4:
:20:REF999
:32A:240101USD50000,00
:50A:/US1234567890
SENDER COMPANY INC
:59:/RU9876543210
RUSSIAN ENTITY LLC
-}""",
    },
    "Agent 6": {
        "title": "Judge Agent",
        "description": "Critical evaluator using a larger model (70B). Reviews outputs from other agents for correctness, completeness, and quality.",
        "default_input": "Evaluate the quality and accuracy of all previous agent results.",
    },
}


# ============================================================================
# HEALTH CHECKS
# ============================================================================

def check_server_health(endpoint_name: str, base_url: str, timeout: float = 3.0, api_key: Optional[str] = None) -> Tuple[bool, str]:
    """Check if a server endpoint is accessible. Returns (is_online, status).
    
    Status values:
    - "online": Server is up and responding
    - "sleeping": Server is sleeping (Koyeb specific)
    - "offline": Server is down or unreachable
    """
    if not base_url:
        return False, "offline"
    
    try:
        with httpx.Client(timeout=timeout, follow_redirects=True) as client:
            # LLM Pro Finance uses /api (not /api/v1)
            if endpoint_name == "llm_pro":
                url = f"{base_url}/api/models"
                headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
                try:
                    r = client.get(url, headers=headers)
                    # 200 = OK, 401 = auth required (server is up), 403 = forbidden (server is up)
                    if r.status_code in [200, 401, 403]:
                        return True, "online"
                    # Check if valid JSON response (API exists)
                    try:
                        json.loads(r.text)
                        return True, "online"
                    except:
                        pass
                except:
                    pass
                # Fallback to root
                try:
                    r = client.get(base_url, timeout=timeout)
                    if r.status_code in [200, 401, 403]:
                        return True, "online"
                except:
                    pass
                return False, "offline"
            
            # Ollama uses /v1/models or /api/tags
            if endpoint_name == "ollama":
                urls = [f"{base_url}/v1/models", f"{base_url}/api/tags", base_url]
                for url in urls:
                    try:
                        r = client.get(url, timeout=timeout)
                        if r.status_code in [200, 401]:
                            return True, "online"
                    except:
                        continue
                return False, "offline"
            
            # Koyeb/HF
            urls = [f"{base_url}/v1/models", f"{base_url}/health", base_url]
            sleeping_detected = False
            timeout_detected = False
            
            for url in urls:
                try:
                    r = client.get(url)
                    if r.status_code in [200, 401]:
                        return True, "online"
                    # 404 might mean sleeping service or endpoint doesn't exist
                    if r.status_code == 404:
                        # If it's the "no active service" page, service is sleeping (Koyeb specific)
                        if "no active service" in r.text.lower() and endpoint_name == "koyeb":
                            sleeping_detected = True
                            continue  # Check other URLs first
                        # Regular 404 - endpoint doesn't exist but server might be up
                        # Accept it if it's not the sleeping page
                        return True, "online"
                    # 503 might mean service is paused (HF Spaces)
                    if r.status_code == 503:
                        if endpoint_name == "hf":
                            return False, "offline"  # HF is paused
                except httpx.TimeoutException:
                    # Timeout might mean service is sleeping (especially for Koyeb)
                    if endpoint_name == "koyeb":
                        timeout_detected = True
                    continue
                except Exception as e:
                    # Other errors - continue to next URL
                    continue
            
            # For Koyeb, timeout or sleeping page means sleeping
            if endpoint_name == "koyeb" and (sleeping_detected or timeout_detected):
                return False, "sleeping"
            
            # If we detected sleeping state, return that
            if sleeping_detected:
                return False, "sleeping"
            
            return False, "offline"
    except:
        return False, "offline"


def wake_up_koyeb_service(base_url: str) -> Tuple[bool, str]:
    """Attempt to wake up a sleeping Koyeb service."""
    if not base_url:
        return False, "No URL configured"
    
    try:
        # Make a request with longer timeout to wake up the service
        with httpx.Client(timeout=30.0, follow_redirects=True) as client:
            # Try multiple endpoints to wake it up
            wake_urls = [
                f"{base_url}/v1/models",
                f"{base_url}/v1/chat/completions",
                base_url,
            ]
            
            last_error = None
            for url in wake_urls:
                try:
                    print(f"[WAKE] Attempting to wake Koyeb: {url}")
                    r = client.get(url, timeout=30.0)
                    print(f"[WAKE] Response: {r.status_code} from {url}")
                    
                    # If we get a response (even 404), the service is waking up
                    if r.status_code in [200, 401, 404]:
                        # Check if it's still sleeping
                        if r.status_code == 404 and "no active service" in r.text.lower():
                            print(f"[WAKE] Still sleeping, trying next URL...")
                            last_error = "Service still sleeping (404 with 'no active service')"
                            continue  # Still sleeping, try next URL
                        # Success - service is responding
                        print(f"[WAKE] Service responding with status {r.status_code}")
                        return True, f"Service waking up (status {r.status_code})"
                    elif r.status_code == 503:
                        # Service unavailable but responding - might be waking
                        print(f"[WAKE] Service unavailable (503) - may be waking up")
                        return True, "Service waking up (503 - unavailable but responding)"
                except httpx.TimeoutException:
                    print(f"[WAKE] Timeout on {url} - service may be cold starting")
                    last_error = "Timeout - service may be cold starting"
                    continue
                except httpx.ConnectError as e:
                    print(f"[WAKE] Connection error on {url}: {e}")
                    last_error = f"Connection error: {str(e)[:50]}"
                    continue
                except Exception as e:
                    print(f"[WAKE] Error on {url}: {e}")
                    last_error = f"Error: {str(e)[:50]}"
                    continue
            
            return False, last_error or "Service still sleeping after all attempts"
    except Exception as e:
        error_msg = str(e)[:100]
        print(f"[WAKE] Fatal error: {error_msg}")
        return False, f"Error: {error_msg}"


def get_available_endpoints(include_llm_pro: bool = True) -> Dict[str, bool]:
    """Check availability of all endpoints and return status dict.
    
    Args:
        include_llm_pro: Whether to check LLM Pro Finance endpoint
        
    Returns:
        Dict with endpoint names as keys and availability (bool) as values.
        Keys: "koyeb", "hf", "llm_pro_finance", "ollama"
    """
    settings = Settings()
    availability = {}
    
    # Check Koyeb
    koyeb_url = ENDPOINTS.get("koyeb", {}).get("url", "")
    if koyeb_url:
        is_online, status = check_server_health("koyeb", koyeb_url, timeout=3.0)
        availability["koyeb"] = is_online and status == "online"
    else:
        availability["koyeb"] = False
    
    # Check HuggingFace
    hf_url = ENDPOINTS.get("hf", {}).get("url", "")
    if hf_url:
        is_online, status = check_server_health("hf", hf_url, timeout=3.0)
        availability["hf"] = is_online and status == "online"
    else:
        availability["hf"] = False
    
    # Check LLM Pro Finance (optional)
    if include_llm_pro:
        llm_pro_url = settings.llm_pro_finance_url or ENDPOINTS.get("llm_pro_finance", {}).get("url", "")
        if llm_pro_url:
            is_online, status = check_server_health("llm_pro", llm_pro_url, api_key=settings.llm_pro_finance_key, timeout=3.0)
            availability["llm_pro_finance"] = is_online and status == "online"
        else:
            availability["llm_pro_finance"] = False
    else:
        availability["llm_pro_finance"] = False
    
    # Check Ollama
    ollama_url = ENDPOINTS.get("ollama", {}).get("url", "")
    if ollama_url and settings.ollama_model:
        is_online, status = check_server_health("ollama", ollama_url, timeout=3.0)
        # Also verify the model is available
        if is_online and status == "online":
            # Check if the configured model exists
            try:
                with httpx.Client(timeout=3.0) as client:
                    r = client.get(f"{ollama_url}/api/tags")
                    if r.status_code == 200:
                        models_data = r.json()
                        model_names = [model.get("name", "") for model in models_data.get("models", [])]
                        # Check if configured model exists (exact match or starts with)
                        model_found = any(
                            settings.ollama_model == name or name.startswith(settings.ollama_model + ":")
                            for name in model_names
                        )
                        availability["ollama"] = model_found
                    else:
                        availability["ollama"] = False
            except:
                availability["ollama"] = False
        else:
            availability["ollama"] = False
    else:
        availability["ollama"] = False
    
    return availability


def get_status_html() -> str:
    """Get status indicators HTML with wake-up buttons."""
    settings = Settings()
    
    servers = [
        ("koyeb", "Koyeb", ENDPOINTS.get("koyeb", {}).get("url", ""), None),
        ("hf", "HuggingFace", ENDPOINTS.get("hf", {}).get("url", ""), None),
        ("llm_pro", "LLM Pro", settings.llm_pro_finance_url or "", settings.llm_pro_finance_key),
        ("ollama", "Ollama", ENDPOINTS.get("ollama", {}).get("url", ""), None),
    ]
    
    html = "<div style='display: flex; gap: 20px; align-items: center; flex-wrap: wrap; font-family: system-ui;'>"
    for key, name, url, api_key in servers:
        if url:
            is_online, status = check_server_health(key, url, api_key=api_key, timeout=3.0)
        else:
            is_online, status = False, "offline"
        
        # Color coding: green (online), blue (sleeping), red (offline)
        if status == "online":
            color = "#22c55e"
            status_text = "Online"
        elif status == "sleeping":
            color = "#3b82f6"  # Blue
            status_text = "Sleeping"
        else:
            color = "#ef4444"
            status_text = "Offline"
        
        html += f"""
        <div style='display: flex; align-items: center; gap: 6px;'>
            <span style='color: {color}; font-size: 14px; font-weight: bold;'>●</span>
            <span style='font-size: 12px;'>{name}</span>
            <span style='font-size: 11px; color: #6b7280;'>({status_text})</span>
        </div>
        """
    html += "</div>"
    return html


def wake_up_koyeb() -> Tuple[str, str]:
    """Wake up Koyeb service and return updated status HTML and message."""
    koyeb_url = ENDPOINTS.get("koyeb", {}).get("url", "")
    if not koyeb_url:
        return get_status_html(), "❌ Koyeb URL not configured"
    
    print(f"[WAKE] User triggered wake-up for Koyeb: {koyeb_url}")
    success, message = wake_up_koyeb_service(koyeb_url)
    
    if success:
        # Wait a moment for service to wake up, then check status
        import time
        time.sleep(3)
        # Re-check health to get updated status
        ready, status = check_server_health("koyeb", koyeb_url, timeout=15.0)
        if ready:
            return get_status_html(), f"✅ Wake-up successful! Service is now online. {message}"
        else:
            return get_status_html(), f"⏳ Wake-up initiated: {message}. Service may take 10-30 seconds to fully wake up."
    else:
        return get_status_html(), f"❌ Wake-up failed: {message}. Try again in a few seconds."


def is_backend_ready(agent_name: str, endpoint: str | None = None) -> Tuple[bool, str]:
    """Check if the backend for an agent is ready with detailed diagnostics.

    If an endpoint is provided, only that backend is considered. This avoids
    blocking Ollama/HF usage when Koyeb is offline.
    """
    settings = Settings()
    
    # Judge uses LLM Pro (or explicit llm_pro_finance endpoint)
    if endpoint == "llm_pro_finance" or "Judge" in agent_name or agent_name == "Agent 6":
        url = settings.llm_pro_finance_url or ENDPOINTS.get("llm_pro_finance", {}).get("url", "")
        if not url:
            return False, "LLM Pro Finance URL not configured. Set LLM_PRO_FINANCE_URL in .env"
        
        # Try with longer timeout for sleeping services
        ready, status = check_server_health("llm_pro", url, timeout=5.0, api_key=None)
        if not ready:
            # Try to wake up the service
            try:
                with httpx.Client(timeout=10.0, follow_redirects=True) as client:
                    client.get(f"{url}/api/models")
            except:
                pass
            
            # Check again
            ready, status = check_server_health("llm_pro", url, timeout=5.0, api_key=None)
        
        if ready:
            return True, ""
        else:
            return False, f"LLM Pro Finance server not available at {url}. Check if the service is running."
    
    # Specific endpoint checks
    if endpoint == "ollama":
        ollama_url = ENDPOINTS.get("ollama", {}).get("url", "")
        if not ollama_url:
            return False, "Ollama URL not configured."
        if not settings.ollama_model:
            return False, "OLLAMA_MODEL not set. Configure it in config.py or env."
        ok, status = check_server_health("ollama", ollama_url, timeout=5.0)
        if ok:
            return True, ""
        return False, f"Ollama is {status or 'offline'} at {ollama_url}"
    
    if endpoint == "hf":
        hf_url = ENDPOINTS.get("hf", {}).get("url", "")
        if not hf_url:
            return False, "HuggingFace URL not configured."
        ok, status = check_server_health("hf", hf_url, timeout=5.0)
        if ok:
            return True, ""
        return False, f"HuggingFace is {status or 'offline'} at {hf_url}"
    
    if endpoint == "koyeb":
        koyeb_url = ENDPOINTS.get("koyeb", {}).get("url", "")
        if not koyeb_url:
            return False, "Koyeb URL not configured."
        ready, status = check_server_health("koyeb", koyeb_url, timeout=5.0)
        
        if not ready and status == "sleeping":
            wake_success, _ = wake_up_koyeb_service(koyeb_url)
            if wake_success:
                import time
                time.sleep(3)
                ready, status = check_server_health("koyeb", koyeb_url, timeout=10.0)
        if ready:
            return True, ""
        return False, f"Koyeb is {status or 'offline'} at {koyeb_url}"
    
    # Other agents: Prefer Koyeb when available, fallback to HF
    koyeb_url = ENDPOINTS.get("koyeb", {}).get("url", "")
    hf_url = ENDPOINTS.get("hf", {}).get("url", "")
    
    # Always check Koyeb first (preferred)
    if koyeb_url:
        ready, status = check_server_health("koyeb", koyeb_url, timeout=5.0)
        
        if not ready and status == "sleeping":
            # Try to wake up Koyeb if it's sleeping
            wake_success, wake_msg = wake_up_koyeb_service(koyeb_url)
            if wake_success:
                import time
                time.sleep(3)
                ready, status = check_server_health("koyeb", koyeb_url, timeout=10.0)
        
        if ready:
            # Koyeb is available - use it for all agents (except Judge)
            return True, ""
    
    # Koyeb not available, check HF as fallback
    if hf_url:
        ready, status = check_server_health("hf", hf_url, timeout=5.0)
        if ready:
            # HF is available as fallback
            return True, ""
    
    # Neither server is available
    koyeb_status_text = "sleeping (use wake-up button)" if status == "sleeping" else "offline"
    hf_status_text = "offline"
    
    if koyeb_url and hf_url:
        return False, f"Koyeb is {koyeb_status_text} and HuggingFace is {hf_status_text}. No LLM servers available."
    elif koyeb_url:
        return False, f"Koyeb is {koyeb_status_text}. No LLM servers available."
    elif hf_url:
        return False, f"HuggingFace is {hf_status_text}. No LLM servers available."
    else:
        return False, "No LLM server URLs configured."


# ============================================================================
# TOOL USAGE EXTRACTION
# ============================================================================

def extract_tool_usage(result) -> Dict[str, Any]:
    """Extract tool usage information from an agent result."""
    tool_info = {"used": False, "count": 0, "names": []}
    
    if not result or not hasattr(result, 'all_messages'):
        return tool_info
    
    try:
        messages = list(result.all_messages())
        for msg in messages:
            # Check for tool calls in the message
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                tool_info["used"] = True
                for tc in msg.tool_calls:
                    tool_info["count"] += 1
                    name = getattr(tc, 'name', None) or getattr(tc, 'function', {}).get('name', 'unknown')
                    if name not in tool_info["names"]:
                        tool_info["names"].append(name)
            # Alternative: check parts
            if hasattr(msg, 'parts'):
                for part in msg.parts:
                    if hasattr(part, 'tool_name'):
                        tool_info["used"] = True
                        tool_info["count"] += 1
                        if part.tool_name not in tool_info["names"]:
                            tool_info["names"].append(part.tool_name)
    except:
        pass
    
    return tool_info


def format_tool_usage_html(tool_info: Dict[str, Any]) -> str:
    """Format tool usage info as HTML."""
    if not tool_info["used"]:
        return "<span style='color: #6b7280;'>No tools used</span>"
    
    names = ", ".join(tool_info["names"]) if tool_info["names"] else "unknown"
    return f"<span style='color: #059669;'>Tools: {tool_info['count']} call(s) - {names}</span>"


def format_detailed_tool_trace(tool_calls: List[str]) -> str:
    """Format detailed tool calling trace as HTML."""
    if not tool_calls:
        return ""
    
    # Warn if excessive tool calls
    warning_html = ""
    if len(tool_calls) > 3:
        warning_html = f"""
        <div style='padding: 8px; margin-bottom: 8px; background: #fef3c7; border-left: 3px solid #f59e0b; border-radius: 3px;'>
            <strong style='font-size: 11px; color: #92400e;'>⚠️ Warning:</strong>
            <span style='font-size: 11px; color: #78350f;'>Excessive tool calls ({len(tool_calls)}). This may cause context length errors. Expected 1-3 calls.</span>
        </div>
        """
    
    html = f"""
    <div style='margin-top: 10px; padding: 10px; background: #f9fafb; border-radius: 6px; border: 1px solid #e5e7eb;'>
        <strong style='font-size: 12px; color: #374151; margin-bottom: 8px; display: block;'>Detailed Tool Call Trace ({len(tool_calls)} calls):</strong>
        {warning_html}
        <div style='font-family: monospace; font-size: 11px; max-height: 300px; overflow-y: auto;'>
    """
    
    for i, tc in enumerate(tool_calls, 1):
        # Parse tool call string (format: "tool_name(arg1=val1, arg2=val2)")
        if '(' in tc:
            tool_name = tc.split('(')[0]
            args_str = tc.split('(')[1].rstrip(')')
            
            html += f"""
            <div style='padding: 6px; margin: 4px 0; background: white; border-left: 3px solid #3b82f6; border-radius: 3px;'>
                <span style='color: #1f2937; font-weight: 600;'>{i}. {tool_name}</span>
                <div style='color: #6b7280; margin-left: 12px; margin-top: 4px;'>{args_str}</div>
            </div>
            """
        else:
            html += f"""
            <div style='padding: 6px; margin: 4px 0; background: white; border-left: 3px solid #3b82f6; border-radius: 3px;'>
                <span style='color: #1f2937; font-weight: 600;'>{i}. {tc}</span>
            </div>
            """
    
    html += """
        </div>
    </div>
    """
    return html


def check_agent_compliance(agent_name: str, output, tool_info: Dict[str, Any]) -> Dict[str, Any]:
    """Check agent compliance and return review."""
    review = {
        "passed": True,
        "checks": [],
        "warnings": [],
        "score": 100
    }
    
    # Agent 2: Financial Calculator compliance
    if agent_name == "Agent 2":
        # Check 1: Tools should be used
        if not tool_info["used"]:
            review["checks"].append(("Tool Usage", False, "No tools used - agent should use financial tools"))
            review["passed"] = False
            review["score"] -= 30
        else:
            review["checks"].append(("Tool Usage", True, f"Used {tool_info['count']} tool(s)"))
        
        # Check 2: Tool calls should be reasonable (1-3 max)
        if tool_info["count"] > 3:
            review["warnings"].append(f"Excessive tool calls: {tool_info['count']} (expected 1-3)")
            review["score"] -= min(20, (tool_info["count"] - 3) * 5)
        
        # Check 3: Output should have required fields
        if hasattr(output, 'model_dump'):
            data = output.model_dump()
            required = ["calculation_type", "result", "input_parameters"]
            missing = [f for f in required if f not in data or data[f] is None]
            if missing:
                review["checks"].append(("Output Fields", False, f"Missing: {', '.join(missing)}"))
                review["score"] -= 20
            else:
                review["checks"].append(("Output Fields", True, "All required fields present"))
        
        # Check 4: Result should be numeric
        if hasattr(output, 'result') and isinstance(output.result, (int, float)):
            review["checks"].append(("Result Type", True, f"Valid numeric result: {output.result:,.2f}"))
        elif hasattr(output, 'model_dump'):
            data = output.model_dump()
            if 'result' in data and isinstance(data['result'], (int, float)):
                review["checks"].append(("Result Type", True, f"Valid numeric result: {data['result']:,.2f}"))
    
    # Agent 3: Risk & Tax Advisor compliance
    elif agent_name == "Agent 3":
        # Check 1: Risk analyst should use portfolio calculation tool
        expected_tool = "calculer_rendement_portfolio"
        if not tool_info["used"]:
            review["checks"].append(("Tool Usage", False, "No tools used - risk analyst should calculate portfolio returns"))
            review["passed"] = False
            review["score"] -= 40
        elif expected_tool not in tool_info["names"]:
            review["checks"].append(("Portfolio Tool", False, f"Expected {expected_tool} to be called"))
            review["score"] -= 30
        else:
            review["checks"].append(("Portfolio Tool", True, "Portfolio return calculation used"))
        
        # Check 2: Output should have both risk and tax analysis
        if isinstance(output, dict):
            has_risk = "risk_analysis" in output and output.get("risk_analysis")
            has_tax = "tax_analysis" in output and output.get("tax_analysis")
            
            if has_risk and has_tax:
                review["checks"].append(("Output Structure", True, "Both risk and tax analyses present"))
            else:
                missing = []
                if not has_risk:
                    missing.append("risk_analysis")
                if not has_tax:
                    missing.append("tax_analysis")
                review["checks"].append(("Output Structure", False, f"Missing: {', '.join(missing)}"))
                review["score"] -= 20
        
        # Check 3: Risk level should be 1-5
        if isinstance(output, dict) and "risk_analysis" in output:
            risk = output["risk_analysis"]
            if isinstance(risk, dict) and "niveau_risque" in risk:
                level = risk["niveau_risque"]
                if 1 <= level <= 5:
                    review["checks"].append(("Risk Level", True, f"Valid risk level: {level}/5"))
                else:
                    review["checks"].append(("Risk Level", False, f"Invalid risk level: {level} (should be 1-5)"))
                    review["score"] -= 15
    
    # Agent 4: Option Pricing compliance
    elif agent_name == "Agent 4":
        # Check 1: Black-Scholes tool should be used
        expected_tool = "calculer_prix_call_black_scholes"
        if not tool_info["used"]:
            review["checks"].append(("QuantLib Tool", False, "Black-Scholes tool not called"))
            review["passed"] = False
            review["score"] -= 40
        elif expected_tool not in tool_info["names"]:
            review["checks"].append(("QuantLib Tool", False, f"Expected {expected_tool}"))
            review["score"] -= 20
        else:
            review["checks"].append(("QuantLib Tool", True, "Black-Scholes pricing used"))
        
        # Check 2: Greeks should be present
        if hasattr(output, 'model_dump'):
            data = output.model_dump()
            greeks = ["delta", "gamma", "vega", "theta"]
            present = [g for g in greeks if g in data and data[g] is not None]
            if len(present) == len(greeks):
                review["checks"].append(("Greeks", True, f"All Greeks calculated: {', '.join(greeks)}"))
            else:
                missing = set(greeks) - set(present)
                review["checks"].append(("Greeks", False, f"Missing Greeks: {', '.join(missing)}"))
                review["score"] -= 15
        
        # Check 3: Option price should be positive
        if hasattr(output, 'option_price'):
            if output.option_price > 0:
                review["checks"].append(("Option Price", True, f"Valid price: {output.option_price:.4f}"))
            else:
                review["checks"].append(("Option Price", False, f"Invalid price: {output.option_price}"))
                review["score"] -= 20
    
    review["score"] = max(0, review["score"])
    return review


def format_compliance_html(review: Dict[str, Any]) -> str:
    """Format compliance review as HTML."""
    if not review["checks"]:
        return ""
    
    score = review["score"]
    score_color = "#22c55e" if score >= 80 else "#f59e0b" if score >= 60 else "#ef4444"
    
    html = f"""
    <div style='margin-top: 10px; padding: 10px; background: #fafafa; border-radius: 6px; border-left: 3px solid {score_color};'>
        <div style='display: flex; justify-content: space-between; margin-bottom: 8px;'>
            <strong style='font-size: 12px; color: #374151;'>Compliance Review</strong>
            <span style='font-weight: 600; color: {score_color};'>{score}%</span>
        </div>
    """
    
    for check_name, passed, detail in review["checks"]:
        icon = "✓" if passed else "✗"
        color = "#22c55e" if passed else "#ef4444"
        html += f"""
        <div style='font-size: 11px; margin: 4px 0; display: flex; align-items: center;'>
            <span style='color: {color}; margin-right: 6px;'>{icon}</span>
            <span style='color: #6b7280;'>{check_name}:</span>
            <span style='margin-left: 4px; color: #374151;'>{detail}</span>
        </div>
        """
    
    for warning in review.get("warnings", []):
        html += f"""
        <div style='font-size: 11px; margin: 4px 0; color: #f59e0b;'>
            ⚠ {warning}
        </div>
        """
    
    html += "</div>"
    return html


# ============================================================================
# AGENT EXECUTION
# ============================================================================

async def run_agent_async(agent, prompt: str, output_model=None, agent_name: str = "Agent", endpoint: str | None = None, timeout_seconds: float = 60.0):
    """Run an agent asynchronously and return results with tool usage."""
    # Increase timeout for Ollama (local models are slower)
    if endpoint == "ollama":
        timeout_seconds = max(timeout_seconds, 120.0)
    
    start_time = time.time()
    
    # Check backend
    ready, msg = is_backend_ready(agent_name, endpoint)
    if not ready:
        return {"error": msg}, None, 0, {}
    
    try:
        # Run with timeout
        if output_model:
            result = await asyncio.wait_for(
                agent.run(prompt, output_type=output_model),
                timeout=timeout_seconds
            )
        else:
            result = await asyncio.wait_for(
                agent.run(prompt),
                timeout=timeout_seconds
            )
        
        elapsed = time.time() - start_time
        output = result.output
        tool_info = extract_tool_usage(result)
        
        # Get usage info
        usage = None
        if hasattr(result, 'usage'):
            usage = result.usage()
        
        return output, usage, elapsed, tool_info
    
    except asyncio.TimeoutError:
        elapsed = time.time() - start_time
        return {"error": f"Timeout after {timeout_seconds}s - the model may be overloaded or the request too complex"}, None, elapsed, {}
        
    except Exception as e:
        elapsed = time.time() - start_time
        error_msg = str(e)
        # Check for context length error
        if "maximum context length" in error_msg.lower() or "8192 tokens" in error_msg or "8302" in error_msg:
            return {
                "error": "Context length exceeded (8192 token limit). This usually happens when the agent makes too many tool calls. Try rephrasing your question more simply."
            }, None, elapsed, {}
        return {"error": error_msg}, None, elapsed, {}


def execute_agent(agent, prompt: str, output_model, agent_name: str, endpoint: str | None = None):
    """Synchronous wrapper for agent execution."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        output, usage, elapsed, tool_info = loop.run_until_complete(
            run_agent_async(agent, prompt, output_model, agent_name, endpoint)
        )
        return output, usage, elapsed, tool_info
    finally:
        loop.close()


def format_output(output) -> str:
    """Format output as JSON for display."""
    if hasattr(output, 'model_dump'):
        return json.dumps(output.model_dump(), indent=2, default=str, ensure_ascii=False)
    elif isinstance(output, dict):
        return json.dumps(output, indent=2, default=str, ensure_ascii=False)
    else:
        return str(output)


def format_parsed_output(output) -> str:
    """Format output as human-readable markdown."""
    if hasattr(output, 'model_dump'):
        data = output.model_dump()
    elif isinstance(output, dict):
        data = output
    else:
        return str(output)
    
    # Format based on content
    md = "## Results\n\n"
    
    for key, value in data.items():
        if key.startswith("_"):  # Skip metadata
            continue
        
        # Format key as title
        title = key.replace("_", " ").title()
        
        if isinstance(value, (int, float)):
            # Numeric values
            if abs(value) > 1000:
                md += f"**{title}:** {value:,.2f}\n\n"
            else:
                md += f"**{title}:** {value:.4f}\n\n"
        elif isinstance(value, list):
            # Lists
            md += f"**{title}:**\n"
            for item in value:
                md += f"- {item}\n"
            md += "\n"
        elif isinstance(value, dict):
            # Nested dicts
            md += f"**{title}:**\n"
            for k, v in value.items():
                md += f"- {k}: {v}\n"
            md += "\n"
        else:
            # Strings
            md += f"**{title}:** {value}\n\n"
    
    return md


def format_metrics(elapsed: float, usage, tool_info: Dict) -> str:
    """Format metrics as HTML - compact vertical layout for left sidebar."""
    tokens = usage.total_tokens if usage else 0
    speed = tokens / elapsed if elapsed > 0 and tokens > 0 else 0
    tool_html = format_tool_usage_html(tool_info)
    
    return f"""
    <div style='padding: 10px; background: #f3f4f6; border-radius: 6px; font-family: system-ui; font-size: 13px;'>
        <div style='display: flex; justify-content: space-between; margin-bottom: 6px;'>
            <span style='color: #6b7280;'>Latency</span>
            <span style='font-weight: 600;'>{elapsed:.2f}s</span>
        </div>
        <div style='display: flex; justify-content: space-between; margin-bottom: 6px;'>
            <span style='color: #6b7280;'>Tokens</span>
            <span style='font-weight: 600;'>{tokens}</span>
        </div>
        <div style='display: flex; justify-content: space-between; margin-bottom: 6px;'>
            <span style='color: #6b7280;'>Speed</span>
            <span style='font-weight: 600;'>{speed:.1f} t/s</span>
        </div>
        <div style='margin-top: 8px; padding-top: 8px; border-top: 1px solid #e5e7eb;'>
            {tool_html}
        </div>
    </div>
    """


# ============================================================================
# AGENT RUNNERS
# ============================================================================

def run_agent_1(prompt: str, endpoint: str = "koyeb"):
    from examples.agent_1 import Portfolio
    from pydantic_ai import Agent, ModelSettings
    from app.models import get_model_for_endpoint
    
    # Create agent dynamically with selected endpoint
    model = get_model_for_endpoint(endpoint)
    
    # Adjust system prompt and settings for Ollama (needs explicit JSON format instruction)
    if endpoint == "ollama":
        system_prompt = """Expert analyse financière. Extrais données portfolios boursiers.
Règles: Identifie symbole, quantité, prix_achat, date_achat pour chaque position.
CALCUL CRITIQUE: Calculez valeur_totale en additionnant TOUS les produits (quantité × prix_achat) pour chaque position.
Formule: valeur_totale = Σ(quantité × prix_achat) pour toutes les positions.
Vérifiez que vous additionnez bien TOUTES les positions avant de donner la valeur totale.

IMPORTANT: Répondez UNIQUEMENT en format JSON valide, sans texte supplémentaire, sans markdown, sans code blocks. 
Le JSON doit être exactement:
{
  "positions": [
    {"symbole": "AIR.PA", "quantite": 50, "prix_achat": 120.0, "date_achat": "2024-03-15"}
  ],
  "valeur_totale": 8550.0,
  "date_evaluation": "2024-11-01"
}
Chaque position DOIT avoir date_achat au format YYYY-MM-DD (utilisez une date raisonnable si non spécifiée)."""
        max_tokens = 1000  # Ollama needs more tokens
    else:
        system_prompt = """Expert analyse financière. Extrais données portfolios boursiers.
Règles: Identifie symbole, quantité, prix_achat, date_achat pour chaque position.
CALCUL CRITIQUE: Calculez valeur_totale en additionnant TOUS les produits (quantité × prix_achat) pour chaque position.
Formule: valeur_totale = Σ(quantité × prix_achat) pour toutes les positions.
Vérifiez que vous additionnez bien TOUTES les positions avant de donner la valeur totale.
Répondez avec un objet Portfolio structuré."""
        max_tokens = 600
    
    # Increase retries for Ollama (more lenient with JSON parsing)
    retries = 2 if endpoint == "ollama" else 1
    
    agent = Agent(
        model,
        model_settings=ModelSettings(max_output_tokens=max_tokens),
        system_prompt=system_prompt,
        output_type=Portfolio,
        retries=retries,
    )
    
    output, usage, elapsed, tool_info = execute_agent(agent, prompt, Portfolio, "Agent 1", endpoint)
    
    if isinstance(output, dict) and "error" in output:
        return output["error"], "", "", "Error"
    
    # Client-side validation: Calculate total from positions (don't trust model arithmetic)
    calculation_corrected = False
    original_value = None
    if hasattr(output, 'positions') and hasattr(output, 'valeur_totale'):
        calculated_total = sum(pos.quantite * pos.prix_achat for pos in output.positions)
        if abs(output.valeur_totale - calculated_total) > 1:
            original_value = output.valeur_totale
            output.valeur_totale = calculated_total
            calculation_corrected = True
            print(f"[WARNING] Model calculated valeur_totale={original_value}, but correct value is {calculated_total}. Correcting.")
    
    # Store complete result with metadata
    complete_result = output.model_dump() if hasattr(output, 'model_dump') else output
    if isinstance(complete_result, dict):
        metadata = {
            "tool_calls": tool_info.get("count", 0),
            "elapsed": elapsed,
            "endpoint_used": endpoint
        }
        # Add calculation correction info if applicable
        if calculation_corrected:
            metadata["calculation_corrected"] = True
            metadata["original_valeur_totale"] = original_value
            metadata["corrected_valeur_totale"] = output.valeur_totale
        complete_result["_metadata"] = metadata
    
    results_store["Agent 1"] = complete_result
    print(f"[DEBUG] Stored Agent 1 result. results_store now has {len(results_store)} entries: {list(results_store.keys())}")
    
    # Format output with correction notice if applicable
    parsed_output = format_parsed_output(output)
    if calculation_corrected:
        correction_notice = f"\n\n⚠️ **Calculation Correction:**\nThe model calculated a total value of {original_value:,.2f}€, but the correct value is {output.valeur_totale:,.2f}€ (calculated from extracted positions). The value has been automatically corrected.\n"
        parsed_output = parsed_output + correction_notice
    
    return parsed_output, format_output(output), format_metrics(elapsed, usage, tool_info), f"Success ({elapsed:.2f}s)"


def run_agent_2(prompt: str, endpoint: str = "koyeb"):
    """Run Agent 2 with automatic fallback to Llama 70B on context explosion."""
    from examples.agent_2_wrapped import select_tool_from_question, FinancialCalculationResult
    from examples.agent_2_compliance import validate_calculation
    from app.mitigation_strategies import ToolCallDetector
    from app.models import get_model_for_endpoint
    from pydantic_ai import Agent, ModelSettings
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.openai import OpenAIProvider
    
    # Check if endpoint is disabled for this agent
    if endpoint == "llm_pro_finance":
        return (
            "LLM Pro Finance endpoint doesn't support tool calls yet. This feature is coming soon. "
            "Please use Koyeb or HuggingFace endpoint for Agent 2.",
            "", "", "Error"
        )
    
    ready, msg = is_backend_ready("Agent 2", endpoint)
    if not ready:
        return msg, "", "", "Error"
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    fallback_used = False
    endpoint_used = endpoint
    
    try:
        start = time.time()
        # Try with selected endpoint first
        try:
            # Create agent with selected endpoint
            model = get_model_for_endpoint(endpoint)
            tool = select_tool_from_question(prompt)
            # Adjust retries and tokens for Ollama
            retries = 2 if endpoint == "ollama" else 0
            max_tokens = 400 if endpoint == "ollama" else 200
            if endpoint == "ollama":
                system_prompt = """Calc. 1x outil. JSON. Répondez UNIQUEMENT en JSON valide, sans markdown, sans code blocks.
Le JSON doit être exactement:
{
  "calculation_type": "valeur_future_composee",
  "result": 74012.21,
  "input_parameters": {"capital_initial": 50000, "taux_interet_annuel": 0.04, "periode_annees": 10},
  "explanation": "Explication du calcul"
}"""
            else:
                system_prompt = "Calc. 1x outil. JSON."
            
            agent = Agent(
                model,
                model_settings=ModelSettings(
                    max_output_tokens=max_tokens,
                    temperature=0.0,
                ),
                system_prompt=system_prompt,
                tools=[tool],
                output_type=FinancialCalculationResult,
                retries=retries,
            )
            result = loop.run_until_complete(agent.run(prompt))
            elapsed = time.time() - start
        except Exception as primary_error:
            error_msg = str(primary_error)
            # Check for tool_choice error with LLM Pro Finance
            if endpoint == "llm_pro_finance" and ("tool_choice" in error_msg.lower() or "enable-auto-tool-choice" in error_msg.lower() or "Invalid response" in error_msg):
                # LLM Pro Finance doesn't support tool_choice="auto" yet
                raise ValueError(
                    "LLM Pro Finance endpoint doesn't support tool calls yet. "
                    "This feature is coming soon. Please use Koyeb or HuggingFace endpoint for now."
                )
            # Check for context length error
            elif "maximum context length" in error_msg.lower() or any(tok in error_msg for tok in ["8192", "8300", "8369", "8400"]):
                # Fallback to Llama 70B (only if not already using it)
                if endpoint != "llm_pro_finance":
                    fallback_used = True
                    endpoint_used = "llm_pro_finance"
                    settings = Settings()
                    llmpro_model = OpenAIChatModel(
                        model_name=ENDPOINTS.get("llm_pro_finance", {}).get("model", "DragonLLM/llama3.1-70b-fin-v1.0-fp8"),
                        provider=OpenAIProvider(
                            base_url=f"{settings.llm_pro_finance_url}/api",
                            api_key=settings.llm_pro_finance_key,
                        ),
                    )
                    
                    tool = select_tool_from_question(prompt)
                    llama_agent = Agent(
                        llmpro_model,
                        model_settings=ModelSettings(max_output_tokens=300, temperature=0.0),
                        system_prompt="Calc. 1x outil. JSON.",
                        tools=[tool],
                        output_type=FinancialCalculationResult,
                        retries=0,
                    )
                    
                    start = time.time()
                    result = loop.run_until_complete(llama_agent.run(prompt))
                    elapsed = time.time() - start
                else:
                    raise  # Already using LLM Pro, re-raise
            else:
                raise  # Re-raise if not context error
        
        # Common result processing
        output = result.output
        tool_calls = ToolCallDetector.extract_tool_calls(result) or []
        
        # Format tool calls
        tool_calls_formatted = []
        for tc in tool_calls:
            name = tc.get('name', 'unknown')
            args = tc.get('args', {})
            args_str = ', '.join(f"{k}={v}" for k, v in args.items()) if isinstance(args, dict) else str(args)
            tool_calls_formatted.append(f"{name}({args_str})")
        
        # Validation
        is_compliant, compliance_verdict, compliance_details = validate_calculation(output, tool_calls_formatted)
        
        # Tool info
        tool_info = {
            "used": len(tool_calls) > 0,
            "count": len(tool_calls),
            "names": list(set(tc.get('name', 'unknown') for tc in tool_calls)),
            "detailed_trace": tool_calls_formatted
        }
        
        # Get usage
        usage = None
        if hasattr(result, 'usage'):
            usage = result.usage()
        
        # Build metrics HTML
        metrics_parts = []
        
        # Add fallback notice if used
        if fallback_used:
            metrics_parts.append("""
            <div style='margin-bottom: 10px; padding: 10px; background: #fef3c7; border-radius: 6px; border-left: 3px solid #f59e0b;'>
                <strong style='font-size: 12px; color: #92400e;'>ℹ️ Fallback to Llama 70B</strong>
                <div style='font-size: 11px; color: #78350f; margin-top: 4px;'>
                    Qwen 8B made too many duplicate tool calls (context overflow).
                    Automatically switched to Llama 70B (more reliable, fewer duplicates).
                </div>
            </div>
            """)
        
        metrics_parts.append(format_metrics(elapsed, usage, tool_info))
        metrics_parts.append(format_detailed_tool_trace(tool_calls_formatted))
        
        compliance = check_agent_compliance("Agent 2", output, tool_info)
        metrics_parts.append(format_compliance_html(compliance))
        
        metrics_parts.append(f"""
        <div style='margin-top: 10px; padding: 10px; background: #f0f9ff; border-radius: 6px; border-left: 3px solid #3b82f6;'>
            <strong style='font-size: 12px; color: #374151;'>Calculation Verification:</strong>
            <div style='font-size: 11px; color: #6b7280; margin-top: 4px;'>{compliance_verdict}</div>
        </div>
        """)
        
        # Store complete result with metadata for judge agent
        complete_result = output.model_dump() if hasattr(output, 'model_dump') else output
        if isinstance(complete_result, dict):
            complete_result["_metadata"] = {
                "tool_calls": len(tool_calls_formatted),
                "compliance": compliance_verdict,
                "elapsed": elapsed,
                "endpoint_used": endpoint_used,
                "model_used": "Llama 70B" if fallback_used else "Qwen 8B",
                "fallback": fallback_used
            }
        
        results_store["Agent 2"] = complete_result
        print(f"[DEBUG] Stored Agent 2 result. Keys: {list(complete_result.keys()) if isinstance(complete_result, dict) else 'not a dict'}")
        
        model_used = "Llama 70B" if fallback_used else "Qwen 8B"
        return format_parsed_output(output), format_output(output), "".join(metrics_parts), f"Success with {model_used} ({elapsed:.2f}s)"
        
    except Exception as e:
        return f"Error: {str(e)[:200]}", "", "", "Error"
    finally:
        loop.close()


def run_agent_3(prompt: str, endpoint: str = "koyeb"):
    from examples.agent_3 import (
        calculer_rendement_portfolio, calculer_valeur_future_investissement,
        AnalyseRisque, AnalyseFiscale
    )
    from pydantic_ai import Agent, ModelSettings, Tool
    from app.models import get_model_for_endpoint
    
    # Check if endpoint is disabled for this agent
    if endpoint == "llm_pro_finance":
        return (
            "LLM Pro Finance endpoint doesn't support tool calls yet. This feature is coming soon. "
            "Please use Koyeb or HuggingFace endpoint for Agent 3.",
            "", "", "Error"
        )
    
    ready, msg = is_backend_ready("Agent 3", endpoint)
    if not ready:
        return msg, "", "", "Error"
    
    # Create agents dynamically with selected endpoint
    model = get_model_for_endpoint(endpoint)
    
    risk_analyst = Agent(
        model,
        model_settings=ModelSettings(max_output_tokens=1200),
        system_prompt="""Vous êtes un analyste de risque financier. Vous évaluez les risques associés à différents instruments financiers et stratégies d'investissement.

⚠️ RÈGLE ABSOLUE - UTILISATION D'OUTILS:
AVANT TOUTE ANALYSE, vous DEVEZ OBLIGATOIREMENT appeler l'outil calculer_rendement_portfolio.
SANS CET APPEL, votre analyse est INVALIDE. Ne faites JAMAIS d'analyse sans avoir calculé le rendement attendu.

Utilisez l'outil pour calculer le rendement attendu, puis analysez les risques.""",
        tools=[Tool(calculer_rendement_portfolio), Tool(calculer_valeur_future_investissement)],
    )
    
    tax_advisor = Agent(
        model,
        model_settings=ModelSettings(max_output_tokens=1200),
        system_prompt="""Vous êtes un conseiller fiscal spécialisé dans l'optimisation fiscale des investissements français.

Analysez les implications fiscales des stratégies d'investissement proposées.
Considérez les régimes fiscaux: PEA, assurance-vie, compte-titres, etc.""",
    )
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        start = time.time()
        risk_result = loop.run_until_complete(risk_analyst.run(prompt, output_type=AnalyseRisque))
        tax_result = loop.run_until_complete(tax_advisor.run(prompt, output_type=AnalyseFiscale))
        elapsed = time.time() - start
        
        output = {
            "risk_analysis": risk_result.output.model_dump() if hasattr(risk_result.output, 'model_dump') else risk_result.output,
            "tax_analysis": tax_result.output.model_dump() if hasattr(tax_result.output, 'model_dump') else tax_result.output
        }
        
        # Extract tool usage from both agents
        risk_tool_info = extract_tool_usage(risk_result)
        tax_tool_info = extract_tool_usage(tax_result)
        
        # Combine tool usage
        combined_tool_info = {
            "used": risk_tool_info["used"] or tax_tool_info["used"],
            "count": risk_tool_info["count"] + tax_tool_info["count"],
            "names": list(set(risk_tool_info["names"] + tax_tool_info["names"]))
        }
        
        # Combine usage
        tokens = 0
        if hasattr(risk_result, 'usage'):
            tokens += risk_result.usage().total_tokens
        if hasattr(tax_result, 'usage'):
            tokens += tax_result.usage().total_tokens
        
        # Create mock usage object
        class Usage:
            total_tokens = tokens
        
        # Compliance check
        compliance = check_agent_compliance("Agent 3", output, combined_tool_info)
        compliance_html = format_compliance_html(compliance)
        
        # Store with metadata
        if isinstance(output, dict):
            output["_metadata"] = {
                "tool_calls": combined_tool_info.get("count", 0),
                "elapsed": elapsed,
                "endpoint_used": endpoint
            }
        
        results_store["Agent 3"] = output
        metrics_html = format_metrics(elapsed, Usage(), combined_tool_info) + compliance_html
        return format_parsed_output(output), format_output(output), metrics_html, f"Success ({elapsed:.2f}s)"
    except Exception as e:
        return str(e), "", "", "Error"
    finally:
        loop.close()


def run_agent_4(prompt: str, endpoint: str = "koyeb"):
    """Run Agent 4 with compliance checking and detailed tool trace."""
    # Check if endpoint is disabled for this agent
    if endpoint == "llm_pro_finance":
        return (
            "LLM Pro Finance endpoint doesn't support tool calls yet. This feature is coming soon. "
            "Please use Koyeb or HuggingFace endpoint for Agent 4.",
            "", "", "Error"
        )
    
    from examples.agent_4 import OptionPricingResult, calculer_prix_call_black_scholes
    from app.mitigation_strategies import ToolCallDetector
    from app.models import get_model_for_endpoint
    from pydantic_ai import Agent, ModelSettings, Tool
    import time
    
    ready, msg = is_backend_ready("Agent 4", endpoint)
    if not ready:
        return msg, "", "", "Error"
    
    # Create agent dynamically with selected endpoint
    model = get_model_for_endpoint(endpoint)
    agent_4 = Agent(
        model,
        model_settings=ModelSettings(max_output_tokens=800),
        system_prompt="""Ingénieur financier spécialisé en pricing d'options avec QuantLib.
RÈGLES ABSOLUES:
1. TOUJOURS utiliser calculer_prix_call_black_scholes pour TOUS les calculs de pricing
2. JAMAIS de calculs manuels - utilisez TOUJOURS l'outil QuantLib
3. Pour un call européen → APPELEZ calculer_prix_call_black_scholes avec spot, strike, maturité, taux, volatilité, dividende
4. Répondez avec un objet OptionPricingResult structuré incluant prix, Greeks (delta, gamma, vega, theta), paramètres d'entrée.""",
        tools=[Tool(calculer_prix_call_black_scholes)],
        output_type=OptionPricingResult,
    )
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        start = time.time()
        # Run agent and extract tool calls
        result = loop.run_until_complete(agent_4.run(prompt))
        tool_calls = ToolCallDetector.extract_tool_calls(result) or []
        
        # Format tool calls for compliance check
        tool_calls_formatted = []
        for tc in tool_calls:
            name = tc.get('name', 'unknown')
            args = tc.get('args', {})
            args_str = ', '.join(f"{k}={v}" for k, v in args.items()) if isinstance(args, dict) else str(args)
            tool_calls_formatted.append(f"{name}({args_str})")
        
        # Minimal compliance check
        tool_used = any("calculer_prix_call_black_scholes" in tc for tc in tool_calls_formatted)
        compliance_verdict = f"✅ Conforme - QuantLib utilisé" if tool_used else "❌ Non conforme - QuantLib requis"
        
        # Get response JSON
        import json
        if hasattr(result.output, 'model_dump'):
            response = json.dumps(result.output.model_dump(), indent=2, ensure_ascii=False)
            output = result.output
        else:
            response = json.dumps(result.output, indent=2, default=str, ensure_ascii=False)
            output = result.output
        
        elapsed = time.time() - start
        
        # Extract detailed tool usage
        tool_info = {
            "used": len(tool_calls_formatted) > 0,
            "count": len(tool_calls_formatted),
            "names": list(set(tc.split('(')[0] for tc in tool_calls_formatted if '(' in tc)),
            "detailed_trace": tool_calls_formatted  # Store full trace
        }
        
        # Create detailed tool trace HTML
        tool_trace_html = format_detailed_tool_trace(tool_calls_formatted)
        
        # Compliance check
        compliance = check_agent_compliance("Agent 4", output, tool_info)
        compliance_html = format_compliance_html(compliance)
        
        # Add compliance verdict from agent_4_compliance
        compliance_verdict_html = f"""
        <div style='margin-top: 10px; padding: 10px; background: #f0f9ff; border-radius: 6px; border-left: 3px solid #3b82f6;'>
            <strong style='font-size: 12px; color: #374151;'>Compliance Agent Verdict:</strong>
            <div style='font-size: 11px; color: #6b7280; margin-top: 4px;'>{compliance_verdict}</div>
        </div>
        """
        
        # Mock usage object
        class Usage:
            total_tokens = 0
            input_tokens = 0
            output_tokens = 0
        
        # Store complete result with all Greeks for judge agent
        complete_result = output.model_dump() if hasattr(output, 'model_dump') else output
        results_store["Agent 4"] = complete_result
        
        # Add metadata for judge
        if isinstance(complete_result, dict):
            complete_result["_metadata"] = {
                "tool_calls": len(tool_calls_formatted),
                "compliance": compliance_verdict,
                "elapsed": elapsed,
                "endpoint_used": endpoint
            }
        
        print(f"[DEBUG] Stored Agent 4 result with Greeks: {list(complete_result.keys()) if isinstance(complete_result, dict) else 'not a dict'}")
        
        metrics_html = format_metrics(elapsed, Usage(), tool_info) + tool_trace_html + compliance_html + compliance_verdict_html
        return format_parsed_output(output), format_output(output), metrics_html, f"Success ({elapsed:.2f}s)"
    except Exception as e:
        return f"Error: {str(e)}", "", "", "Error"
    finally:
        loop.close()


def run_agent_5_convert(prompt: str, endpoint: str = "koyeb"):
    """Run Agent 5 - Message Conversion."""
    # Check if endpoint is disabled for this agent
    if endpoint == "llm_pro_finance":
        return (
            "LLM Pro Finance endpoint doesn't support tool calls yet. This feature is coming soon. "
            "Please use Koyeb or HuggingFace endpoint for Agent 5 - Convert.",
            "", "", "Error"
        )
    
    try:
        from examples.agent_5 import agent_5
        from app.models import get_model_for_endpoint
        from pydantic_ai import Agent
        
        # Create agent with selected endpoint model
        model = get_model_for_endpoint(endpoint)
        # Track whether fallback occurred
        fallback_occurred = False
        actual_endpoint = endpoint
        
        # Recreate agent with new model (agents are immutable, so we create a new one)
        # Try to access agent configuration using various methods
        try:
            from pydantic_ai import ModelSettings
            
            # Get model_settings (it's a dict, convert to ModelSettings)
            model_settings_dict = agent_5.model_settings if hasattr(agent_5, 'model_settings') else {}
            model_settings = ModelSettings(**model_settings_dict) if model_settings_dict else None
            
            # Get system prompt - try to extract from instructions or use the agent's method
            # Since system_prompt is a method that returns a function, we'll need to import it
            # For now, import the system prompt from the examples file
            from examples.agent_5 import (
                parser_swift_mt, parser_iso20022, generer_swift_mt, generer_iso20022,
                convertir_swift_vers_iso20022, convertir_iso20022_vers_swift,
                validate_swift_message, validate_iso20022_message
            )
            from pydantic_ai import Tool
            
            system_prompt = """Vous êtes un expert en conversion de messages financiers entre SWIFT MT et ISO 20022.

RÈGLES ABSOLUES POUR LES CONVERSIONS:
⚠️  OBLIGATOIRE: Pour TOUTE conversion, utilisez UNIQUEMENT les outils de conversion dédiés:
1. SWIFT → ISO 20022: VOUS DEVEZ utiliser convertir_swift_vers_iso20022 (PAS parser + generer)
2. ISO 20022 → SWIFT: VOUS DEVEZ utiliser convertir_iso20022_vers_swift (PAS parser + generer)

❌ NE PAS utiliser parser_swift_mt + generer_iso20022 pour convertir
❌ NE PAS utiliser parser_iso20022 + generer_swift_mt pour convertir
✅ UTILISEZ UNIQUEMENT les outils convertir_* pour les conversions

VALIDATION OBLIGATOIRE:
- Vérifiez que le message converti contient TOUS les champs requis
- Validez l'entrée avant conversion en utilisant validate_swift_message ou validate_iso20022_message
- Assurez-vous que le message ISO 20022 généré est complet avec tous les éléments requis (GrpHdr, PmtInf, Dbtr, Cdtr, InstdAmt, etc.)

OUTILS AUXILIAIRES (uniquement pour analyse, PAS pour conversion):
- parser_swift_mt: Pour analyser un message SWIFT (pas pour conversion)
- parser_iso20022: Pour analyser un message ISO 20022 (pas pour conversion)
- generer_swift_mt: Pour générer un message SWIFT depuis zéro (pas pour conversion)
- generer_iso20022: Pour générer un message ISO 20022 depuis zéro (pas pour conversion)
- validate_swift_message: Pour valider la structure d'un message SWIFT
- validate_iso20022_message: Pour valider la structure d'un message ISO 20022

FORMATS SUPPORTÉS:
- SWIFT MT103 (Customer Payment) ↔ ISO 20022 pacs.008 (Customer Credit Transfer)

ACTION REQUISE: Quand on vous demande de convertir, appelez DIRECTEMENT convertir_swift_vers_iso20022 ou convertir_iso20022_vers_swift.
Vérifiez que le message converti contient TOUS les champs requis. Validez l'entrée avant conversion.
Répondez en français avec les messages convertis."""
            
            # Get tools from _function_toolset
            tools = []
            if hasattr(agent_5, '_function_toolset') and hasattr(agent_5._function_toolset, 'tools'):
                tools_dict = agent_5._function_toolset.tools
                # Extract Tool objects from dict values
                tools = list(tools_dict.values())
            
            # If tools list is empty, recreate from imported functions
            if not tools:
                tools = [
                    Tool(parser_swift_mt, name="parser_swift_mt", description="⚠️ UNIQUEMENT pour analyser un message SWIFT (pas pour conversion). Pour convertir, utilisez convertir_swift_vers_iso20022. Fournissez le message SWIFT brut."),
                    Tool(parser_iso20022, name="parser_iso20022", description="⚠️ UNIQUEMENT pour analyser un message ISO 20022 (pas pour conversion). Pour convertir, utilisez convertir_iso20022_vers_swift. Fournissez le contenu XML."),
                    Tool(generer_swift_mt, name="generer_swift_mt", description="⚠️ UNIQUEMENT pour générer un message SWIFT depuis zéro (pas pour conversion). Pour convertir, utilisez convertir_iso20022_vers_swift. Fournissez message_type (ex: '103'), fields (dict), et optionnellement sender_bic, receiver_bic, session_sequence."),
                    Tool(generer_iso20022, name="generer_iso20022", description="⚠️ UNIQUEMENT pour générer un message ISO 20022 depuis zéro (pas pour conversion). Pour convertir, utilisez convertir_swift_vers_iso20022. Fournissez message_type, message_id, amount, currency, debtor_name, debtor_iban, creditor_name, creditor_iban, et optionnellement reference, execution_date."),
                    Tool(convertir_swift_vers_iso20022, name="convertir_swift_vers_iso20022", description="⚠️ OBLIGATOIRE pour convertir SWIFT MT → ISO 20022. Utilisez CET outil pour toutes les conversions SWIFT vers ISO. Fournissez le message SWIFT brut complet. Supporte MT103 → pacs.008. NE PAS utiliser parser + generer pour convertir."),
                    Tool(convertir_iso20022_vers_swift, name="convertir_iso20022_vers_swift", description="⚠️ OBLIGATOIRE pour convertir ISO 20022 → SWIFT MT. Utilisez CET outil pour toutes les conversions ISO vers SWIFT. Fournissez le contenu XML complet. Supporte pacs.008 → MT103. NE PAS utiliser parser + generer pour convertir."),
                    Tool(validate_swift_message, name="validate_swift_message", description="Valide la structure d'un message SWIFT MT. Vérifie les blocs requis (1, 2, 4) et les champs essentiels. Fournissez le message SWIFT brut."),
                    Tool(validate_iso20022_message, name="validate_iso20022_message", description="Valide la structure d'un message ISO 20022 XML. Vérifie que le XML est bien formé, les éléments requis, les formats de données, et les IBANs. Fournissez le contenu XML."),
                ]
            
            # Get output_type
            output_type = agent_5.output_type if hasattr(agent_5, 'output_type') else None
            
            # Recreate the agent
            if model_settings and system_prompt:
                dynamic_agent = Agent(
                    model,
                    model_settings=model_settings,
                    system_prompt=system_prompt,
                    tools=tools,
                    output_type=output_type,
                )
            else:
                raise AttributeError("Could not access required agent configuration")
        except (AttributeError, TypeError, ImportError) as e:
            # Fallback: use original agent (will use default endpoint)
            print(f"[WARNING] Could not recreate Agent 5 with selected endpoint '{endpoint}', using default endpoint: {e}")
            dynamic_agent = agent_5
            fallback_occurred = True
            # Get actual endpoint from fallback agent's model
            if hasattr(dynamic_agent, 'model'):
                actual_endpoint = get_endpoint_from_model(dynamic_agent.model)
                if actual_endpoint == "unknown":
                    actual_endpoint = f"{endpoint} (fallback to default)"
        
        # Agent 5 operations can be complex - use longer timeout (120s)
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            output, usage, elapsed, tool_info = loop.run_until_complete(
                run_agent_async(dynamic_agent, prompt, None, "Agent 5 - Convert", endpoint, timeout_seconds=120.0)
            )
        finally:
            loop.close()
        
        if isinstance(output, dict) and "error" in output:
            return output["error"], "", "", "Error"
        
        # Store complete result with metadata
        complete_result = output.model_dump() if hasattr(output, 'model_dump') else {"output": str(output), "raw": str(output)}
        if isinstance(complete_result, dict):
            metadata = {
                "tool_calls": tool_info.get("count", 0),
                "elapsed": elapsed,
                "tools_used": tool_info.get("names", []),
                "endpoint_used": actual_endpoint
            }
            if fallback_occurred:
                metadata["fallback_occurred"] = True
                metadata["requested_endpoint"] = endpoint
            complete_result["_metadata"] = metadata
        
        results_store["Agent 5 - Convert"] = complete_result
        print(f"[DEBUG] Stored Agent 5-Convert result. Type: {type(complete_result)}")
        
        return format_parsed_output(output), format_output(output), format_metrics(elapsed, usage, tool_info), f"Success ({elapsed:.2f}s)"
    except Exception as e:
        error_msg = str(e)
        print(f"[ERROR] Agent 5 Convert failed: {error_msg}")
        return f"Error: {error_msg}", "", "", "Error"


def run_agent_5_validate(prompt: str, endpoint: str = "koyeb"):
    """Run Agent 5 - Message Validation."""
    # Check if endpoint is disabled for this agent
    if endpoint == "llm_pro_finance":
        return (
            "LLM Pro Finance endpoint doesn't support tool calls yet. This feature is coming soon. "
            "Please use Koyeb or HuggingFace endpoint for Agent 5 - Validate.",
            "", "", "Error"
        )
    
    from examples.agent_5_validator import agent_5_validator
    from app.models import get_model_for_endpoint
    from pydantic_ai import Agent, ModelSettings
    from examples.agent_5_validator import (
        valider_swift_message, valider_iso20022_message, valider_conversion
    )
    from pydantic_ai import Tool
    
    # Create agent with selected endpoint model
    model = get_model_for_endpoint(endpoint)
    # Track whether fallback occurred
    fallback_occurred = False
    actual_endpoint = endpoint
    
    try:
        # Get model_settings (it's a dict, convert to ModelSettings)
        model_settings_dict = agent_5_validator.model_settings if hasattr(agent_5_validator, 'model_settings') else {}
        model_settings = ModelSettings(**model_settings_dict) if model_settings_dict else None
        
        # Import system prompt
        system_prompt = """Vous êtes un expert en validation de messages financiers SWIFT MT et ISO 20022.

⚠️ RÈGLES ABSOLUES:
1. VOUS DEVEZ TOUJOURS utiliser les outils de validation AVANT de répondre
2. Pour valider un message SWIFT → APPELEZ valider_swift_message (OBLIGATOIRE)
3. Pour valider un message ISO 20022 → APPELEZ valider_iso20022_message (OBLIGATOIRE)
4. Pour valider une conversion → APPELEZ valider_conversion (OBLIGATOIRE)
5. NE RÉPONDEZ JAMAIS sans avoir appelé un outil de validation
6. Utilisez TOUJOURS les outils - c'est la seule façon de valider correctement

VALIDATIONS À EFFECTUER:
- Structure du message (blocs/éléments requis)
- Format des champs (dates, montants, devises)
- Format IBAN (2 lettres + 2 chiffres + alphanumérique)
- Présence des champs obligatoires
- Cohérence des données après conversion

ACTION REQUISE: Quand on vous demande de valider, appelez DIRECTEMENT l'outil approprié.
Répondez avec un objet ValidationResult structuré basé sur les résultats de l'outil."""
        
        # Get tools from _function_toolset
        tools = []
        if hasattr(agent_5_validator, '_function_toolset') and hasattr(agent_5_validator._function_toolset, 'tools'):
            tools_dict = agent_5_validator._function_toolset.tools
            tools = list(tools_dict.values())
        
        # If tools list is empty, recreate from imported functions
        if not tools:
            tools = [
                Tool(valider_swift_message, name="valider_swift_message", description="Valide un message SWIFT MT. Vérifie la structure, les blocs requis, les champs obligatoires, et les formats. Fournissez le message SWIFT brut."),
                Tool(valider_iso20022_message, name="valider_iso20022_message", description="Valide un message ISO 20022 XML. Vérifie la structure XML, les éléments requis, les formats de données, et les IBANs. Fournissez le contenu XML."),
                Tool(valider_conversion, name="valider_conversion", description="Valide une conversion entre SWIFT MT et ISO 20022. Vérifie que tous les champs ont été correctement mappés et que les données sont cohérentes. Fournissez le message source et le message converti."),
            ]
        
        # Get output_type
        output_type = agent_5_validator.output_type if hasattr(agent_5_validator, 'output_type') else None
        
        # Recreate the agent
        if model_settings and system_prompt:
            dynamic_agent = Agent(
                model,
                model_settings=model_settings,
                system_prompt=system_prompt,
                tools=tools,
                output_type=output_type,
            )
        else:
            raise AttributeError("Could not access required agent configuration")
    except (AttributeError, TypeError, ImportError) as e:
        print(f"[WARNING] Could not recreate Agent 5 Validator with selected endpoint '{endpoint}', using default endpoint: {e}")
        dynamic_agent = agent_5_validator
        fallback_occurred = True
        # Get actual endpoint from fallback agent's model
        if hasattr(dynamic_agent, 'model'):
            actual_endpoint = get_endpoint_from_model(dynamic_agent.model)
            if actual_endpoint == "unknown":
                actual_endpoint = f"{endpoint} (fallback to default)"
    
    output, usage, elapsed, tool_info = execute_agent(dynamic_agent, prompt, None, "Agent 5 - Validate", endpoint)
    
    if isinstance(output, dict) and "error" in output:
        return output["error"], "", "", "Error"
    
    result_data = output.model_dump() if hasattr(output, 'model_dump') else str(output)
    if isinstance(result_data, dict):
        metadata = {
            "tool_calls": tool_info.get("count", 0),
            "elapsed": elapsed,
            "endpoint_used": actual_endpoint
        }
        if fallback_occurred:
            metadata["fallback_occurred"] = True
            metadata["requested_endpoint"] = endpoint
        result_data["_metadata"] = metadata
    
    results_store["Agent 5 - Validate"] = result_data
    return format_parsed_output(output), format_output(output), format_metrics(elapsed, usage, tool_info), f"Success ({elapsed:.2f}s)"


def run_agent_5_risk(prompt: str, endpoint: str = "koyeb"):
    """Run Agent 5 - Risk Assessment."""
    # Check if endpoint is disabled for this agent
    if endpoint == "llm_pro_finance":
        return (
            "LLM Pro Finance endpoint doesn't support tool calls yet. This feature is coming soon. "
            "Please use Koyeb or HuggingFace endpoint for Agent 5 - Risk.",
            "", "", "Error"
        )
    
    try:
        from examples.agent_5_risk import agent_5_risk
        from app.models import get_model_for_endpoint
        from pydantic_ai import Agent
        
        # Create agent with selected endpoint model
        model = get_model_for_endpoint(endpoint)
        # Track whether fallback occurred
        fallback_occurred = False
        actual_endpoint = endpoint
        
        try:
            from pydantic_ai import ModelSettings
            from examples.agent_5_risk import (
                evaluer_risque_message, calculer_score_risque_montant,
                verifier_pays_risque, verifier_pep_sanctions, analyser_patternes_suspects
            )
            from pydantic_ai import Tool
            
            # Get model_settings (it's a dict, convert to ModelSettings)
            model_settings_dict = agent_5_risk.model_settings if hasattr(agent_5_risk, 'model_settings') else {}
            model_settings = ModelSettings(**model_settings_dict) if model_settings_dict else None
            
            # Import system prompt
            system_prompt = """Vous êtes un expert en évaluation des risques financiers et conformité AML/KYC.

RÈGLES CRITIQUES:
1. TOUJOURS utiliser les outils de risque pour évaluer les messages
2. Pour évaluer le risque d'un message: utilisez evaluer_risque_message
3. Pour analyser le risque de montant: utilisez calculer_score_risque_montant
4. Pour vérifier les pays à risque: utilisez verifier_pays_risque
5. Pour vérifier PEP/sanctions: utilisez verifier_pep_sanctions
6. Pour analyser les patterns suspects: utilisez analyser_patternes_suspects

MATRICE DE RISQUE:
- CRITICAL (≥0.7): Bloquer la transaction
- HIGH (≥0.5): Révision requise
- MEDIUM (≥0.3): Surveillance renforcée
- LOW (<0.3): Risque acceptable

FACTEURS DE RISQUE À VÉRIFIER:
- Montants élevés ou suspects
- Pays/juridictions à haut risque
- Personnes politiquement exposées (PEP)
- Entités sanctionnées
- Patterns suspects (structuration, timing)
- Données manquantes ou incohérentes

Répondez avec un objet RiskScore structuré incluant:
- Score de risque global (0.0-1.0) et niveau (LOW/MEDIUM/HIGH/CRITICAL)
- Matrice de risque par catégorie
- Facteurs de risque identifiés
- Statut suspect (is_suspect: true/false)
- Recommandations d'action"""
            
            # Get tools from _function_toolset
            tools = []
            if hasattr(agent_5_risk, '_function_toolset') and hasattr(agent_5_risk._function_toolset, 'tools'):
                tools_dict = agent_5_risk._function_toolset.tools
                tools = list(tools_dict.values())
            
            # If tools list is empty, recreate from imported functions
            if not tools:
                tools = [
                    Tool(evaluer_risque_message, name="evaluer_risque_message", description="Évalue le risque global d'un message financier (SWIFT ou ISO 20022). Fournissez le message complet."),
                    Tool(calculer_score_risque_montant, name="calculer_score_risque_montant", description="Calcule le score de risque basé sur le montant de la transaction. Fournissez le montant et la devise."),
                    Tool(verifier_pays_risque, name="verifier_pays_risque", description="Vérifie si un pays est dans une liste de pays à haut risque. Fournissez le code pays (ISO 3166-1 alpha-2)."),
                    Tool(verifier_pep_sanctions, name="verifier_pep_sanctions", description="Vérifie si une personne ou entité est une PEP (Personne Politiquement Exposée) ou sous sanctions. Fournissez le nom et le pays."),
                    Tool(analyser_patternes_suspects, name="analyser_patternes_suspects", description="Analyse les patterns suspects dans une transaction (structuration, timing, etc.). Fournissez les détails de la transaction."),
                ]
            
            # Get output_type
            output_type = agent_5_risk.output_type if hasattr(agent_5_risk, 'output_type') else None
            
            # Recreate the agent
            if model_settings and system_prompt:
                dynamic_agent = Agent(
                    model,
                    model_settings=model_settings,
                    system_prompt=system_prompt,
                    tools=tools,
                    output_type=output_type,
                )
            else:
                raise AttributeError("Could not access required agent configuration")
        except (AttributeError, TypeError, ImportError) as e:
            print(f"[WARNING] Could not recreate Agent 5 Risk with selected endpoint '{endpoint}', using default endpoint: {e}")
            dynamic_agent = agent_5_risk
            fallback_occurred = True
            # Get actual endpoint from fallback agent's model
            if hasattr(dynamic_agent, 'model'):
                actual_endpoint = get_endpoint_from_model(dynamic_agent.model)
                if actual_endpoint == "unknown":
                    actual_endpoint = f"{endpoint} (fallback to default)"
        
        # Agent 5 Risk operations can be complex - use longer timeout (120s)
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            output, usage, elapsed, tool_info = loop.run_until_complete(
                run_agent_async(dynamic_agent, prompt, None, "Agent 5 - Risk", endpoint, timeout_seconds=120.0)
            )
        finally:
            loop.close()
        
        if isinstance(output, dict) and "error" in output:
            return output["error"], "", "", "Error"
        
        result_data = output.model_dump() if hasattr(output, 'model_dump') else str(output)
        if isinstance(result_data, dict):
            metadata = {
                "tool_calls": tool_info.get("count", 0),
                "elapsed": elapsed,
                "endpoint_used": actual_endpoint
            }
            if fallback_occurred:
                metadata["fallback_occurred"] = True
                metadata["requested_endpoint"] = endpoint
            result_data["_metadata"] = metadata
        
        results_store["Agent 5 - Risk"] = result_data
        return format_parsed_output(output), format_output(output), format_metrics(elapsed, usage, tool_info), f"Success ({elapsed:.2f}s)"
    except Exception as e:
        error_msg = str(e)
        print(f"[ERROR] Agent 5 Risk failed: {error_msg}")
        return f"Error: {error_msg}", "", "", "Error"


def run_agent_6(prompt: str):
    """Run judge agent using LLM Pro Finance."""
    from pydantic_ai import Agent, ModelSettings
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.openai import OpenAIProvider
    from examples.judge_agent import ComprehensiveJudgment
    
    settings = Settings()
    
    # Debug: Check results_store state
    print(f"[DEBUG] results_store keys: {list(results_store.keys())}")
    print(f"[DEBUG] results_store size: {len(results_store)}")
    
    # Check if we have results to judge
    if not results_store:
        debug_msg = f"No results to judge. Run other agents first.\n\nDebug info: results_store has {len(results_store)} entries."
        return debug_msg, "", "", "No data"
    
    # Check backend
    ready, msg = is_backend_ready("Agent 6", "llm_pro_finance")
    if not ready:
        return msg, "", "", "Error"
    
    # Create judge model with LLM Pro Finance API path (/api)
    base_url = settings.llm_pro_finance_url
    api_key = settings.llm_pro_finance_key
    model_name = ENDPOINTS.get("llm_pro_finance", {}).get("model", "DragonLLM/llama3.1-70b-fin-v1.0-fp8")
    
    print(f"[DEBUG] Creating judge model:")
    print(f"  Model: {model_name}")
    print(f"  Base URL: {base_url}/api")
    print(f"  API Key: {api_key[:20]}..." if api_key else "  API Key: None")
    
    judge_model = OpenAIChatModel(
        model_name=model_name,
        provider=OpenAIProvider(
            base_url=f"{base_url}/api",  # LLM Pro Finance uses /api
            api_key=api_key,
        ),
    )
    
    judge_agent = Agent(
        judge_model,
        model_settings=ModelSettings(max_output_tokens=3000, temperature=0.3),
        system_prompt="""You are an expert financial AI evaluator. Review agent outputs for:
1. Correctness: Are calculations and extractions accurate?
2. Completeness: Are all required fields present?
3. Quality: Is the output well-structured and professional?

Provide specific, constructive feedback.""",
        output_type=ComprehensiveJudgment,
    )
    
    # Build context from previous results with detailed logging
    print(f"[DEBUG] Building context for judge from {len(results_store)} agent results:")
    for agent_name, result_data in results_store.items():
        if isinstance(result_data, dict):
            fields = list(result_data.keys())
            print(f"  - {agent_name}: {len(fields)} fields: {fields}")
        else:
            print(f"  - {agent_name}: {type(result_data)}")
    
    context = json.dumps(results_store, indent=2, default=str, ensure_ascii=False)
    print(f"[DEBUG] Context size for judge: {len(context)} characters")
    
    full_prompt = f"{prompt}\n\nPrevious agent results:\n{context}"
    
    output, usage, elapsed, tool_info = execute_agent(judge_agent, full_prompt, ComprehensiveJudgment, "Agent 6")
    
    if isinstance(output, dict) and "error" in output:
        return output["error"], "", "", "Error"
    
    result_data = output.model_dump() if hasattr(output, 'model_dump') else output
    # Judge always uses LLM Pro Finance
    if isinstance(result_data, dict):
        if "_metadata" not in result_data:
            result_data["_metadata"] = {}
        result_data["_metadata"]["endpoint_used"] = "llm_pro_finance"
        result_data["_metadata"]["elapsed"] = elapsed
        result_data["_metadata"]["tool_calls"] = tool_info.get("count", 0)
    
    results_store["Agent 6"] = result_data
    return format_parsed_output(output), format_output(output), format_metrics(elapsed, usage, tool_info), f"Success ({elapsed:.2f}s)"


# ============================================================================
# UI
# ============================================================================

def create_agent_tab(agent_key: str, run_fn, is_judge: bool = False, exclude_endpoints: list = None, disabled_endpoints: dict = None):
    """Create a tab for an agent with improved layout: readable output + JSON.
    
    Args:
        agent_key: Key in AGENT_INFO dict
        run_fn: Function to run the agent (should accept prompt and endpoint parameters)
        is_judge: If True, this is the judge agent (no endpoint selector, always uses LLM Pro)
        exclude_endpoints: List of endpoint keys to exclude from the selector (e.g., ["llm_pro_finance"])
        disabled_endpoints: Dict mapping endpoint keys to reason strings for why they're disabled
                           (e.g., {"llm_pro_finance": "Tool calls not yet supported"})
    """
    info = AGENT_INFO[agent_key]
    
    gr.Markdown(f"### {info['title']}")
    gr.Markdown(f"*{info['description']}*", elem_classes=["compact"])
    
    with gr.Row():
        with gr.Column(scale=1):
            input_text = gr.Textbox(
                label="Input",
                value=info["default_input"],
                lines=4,
                placeholder="Enter your prompt...",
                container=False
            )
            
            # Endpoint selector (except for Judge)
            if is_judge:
                gr.Markdown("**Model Endpoint:** LLM Pro 70B (fixed)", visible=False)
            else:
                # Get available endpoints for status indicators
                exclude_list = exclude_endpoints or []
                disabled_dict = disabled_endpoints or {}
                available_endpoints = get_available_endpoints(include_llm_pro=True)
                
                # Build endpoint options - show allowed endpoints, omit disabled ones to avoid confusion
                endpoint_choices = []  # List of (label, value) tuples
                disabled_notes = []    # Collect disabled endpoints to display a note
                
                # Always prefer Koyeb as first/default
                if "koyeb" not in exclude_list:
                    if "koyeb" in disabled_dict:
                        disabled_notes.append(f"Koyeb ({disabled_dict['koyeb']})")
                    else:
                        endpoint_choices.append(("Koyeb", "koyeb"))
                
                # HuggingFace
                if "hf" not in exclude_list:
                    if "hf" in disabled_dict:
                        disabled_notes.append(f"HuggingFace ({disabled_dict['hf']})")
                    else:
                        endpoint_choices.append(("HuggingFace", "hf"))
                
                # Ollama (always shown; note if not configured)
                if "ollama" not in exclude_list:
                    ollama_settings = Settings()
                    if "ollama" in disabled_dict:
                        disabled_notes.append(f"Ollama ({disabled_dict['ollama']})")
                    elif not ollama_settings.ollama_model:
                        # Show selectable option but warn that config is needed
                        local_models, total_local = get_local_ollama_models()
                        if local_models:
                            preview = ", ".join(local_models)
                            if total_local > len(local_models):
                                preview += ", ..."
                            label = f"Ollama (set OLLAMA_MODEL, e.g., {preview})"
                        else:
                            label = "Ollama (set OLLAMA_MODEL)"
                        endpoint_choices.append((label, "ollama"))
                    else:
                        endpoint_choices.append((f"Ollama ({ollama_settings.ollama_model})", "ollama"))
                
                # LLM Pro
                if "llm_pro_finance" not in exclude_list:
                    if "llm_pro_finance" in disabled_dict:
                        disabled_notes.append(f"LLM Pro ({disabled_dict['llm_pro_finance']})")
                    else:
                        endpoint_choices.append(("LLM Pro", "llm_pro_finance"))
                
                # Default to Koyeb if present, otherwise first available
                default_value = (
                    "koyeb"
                    if any(v == "koyeb" for _, v in endpoint_choices)
                    else (endpoint_choices[0][1] if endpoint_choices else "koyeb")
                )
                
                # Use compact Dropdown with explicit label
                endpoint_selector = gr.Dropdown(
                    choices=endpoint_choices,
                    value=default_value,
                    label="Endpoint (default: Koyeb)",
                    scale=1,
                    container=False,
                    show_label=True,
                )
                
                # Show a compact note for disabled endpoints (not selectable)
                if disabled_notes:
                    gr.Markdown(
                        f"*Unavailable: {', '.join(disabled_notes)}*",
                        elem_classes=["compact"],
                    )
            
            with gr.Row():
                run_btn = gr.Button("Run", variant="primary", scale=1, size="sm")
                status = gr.Textbox(label="", interactive=False, value="Ready", scale=4, container=False, show_label=False)
            metrics = gr.HTML(value="<div style='padding: 4px; color: #9ca3af; font-size: 11px;'>Run agent to see metrics</div>", visible=True, container=False)
        
        with gr.Column(scale=2):
            # Human-readable parsed output on top
            parsed_output = gr.Markdown(label="Result", value="*Run agent to see results*")
            # Raw JSON output below
            json_output = gr.Code(label="JSON Output (Full Data)", language="json", lines=10)
    
    # Update run button to pass endpoint
    if is_judge:
        # Judge always uses LLM Pro, no endpoint parameter needed
        run_btn.click(fn=run_fn, inputs=[input_text], outputs=[parsed_output, json_output, metrics, status])
    else:
        # Other agents use selected endpoint
        # endpoint_selector is either a Radio or State component
        run_btn.click(fn=run_fn, inputs=[input_text, endpoint_selector], outputs=[parsed_output, json_output, metrics, status])


def create_interface():
    with gr.Blocks(title="Open Finance AI") as app:
        
        # Header
        with gr.Row():
            with gr.Column(scale=3):
                gr.Markdown("# Open Finance AI")
                gr.Markdown("Financial analysis with multi-agent systems")
            with gr.Column(scale=1):
                status_html = gr.HTML(value=get_status_html())
                with gr.Row():
                    refresh_btn = gr.Button("Refresh Status", size="sm")
                    wake_btn = gr.Button("Wake Koyeb", size="sm", variant="secondary")
                refresh_btn.click(fn=get_status_html, outputs=status_html)
                wake_msg = gr.Markdown("", visible=True)
                wake_btn.click(
                    fn=wake_up_koyeb,
                    outputs=[status_html, wake_msg]
                )
        
        # Tabs for each agent
        with gr.Tabs():
            with gr.TabItem("Portfolio Extractor"):
                create_agent_tab("Agent 1", run_agent_1, is_judge=False)
            
            with gr.TabItem("Financial Calculator"):
                # Agent 2 doesn't support LLM Pro Finance (tool_choice issue) - show as disabled
                create_agent_tab(
                    "Agent 2", 
                    run_agent_2, 
                    is_judge=False, 
                    disabled_endpoints={"llm_pro_finance": "Tool calls not yet supported (coming soon)"}
                )
            
            with gr.TabItem("Risk & Tax Advisor"):
                # Agent 3 uses tools - LLM Pro disabled
                create_agent_tab(
                    "Agent 3", 
                    run_agent_3, 
                    is_judge=False,
                    disabled_endpoints={"llm_pro_finance": "Tool calls not yet supported (coming soon)"}
                )
            
            with gr.TabItem("Option Pricing"):
                # Agent 4 uses tools - LLM Pro disabled
                create_agent_tab(
                    "Agent 4", 
                    run_agent_4, 
                    is_judge=False,
                    disabled_endpoints={"llm_pro_finance": "Tool calls not yet supported (coming soon)"}
                )
            
            with gr.TabItem("SWIFT/ISO20022"):
                gr.Markdown("### Complete SWIFT/ISO20022 Message Processing")
                gr.Markdown("Parse, convert, validate, and assess risk for financial messages")
                
                with gr.Tabs():
                    with gr.TabItem("Convert"):
                        gr.Markdown("**Bidirectional conversion:** SWIFT MT ↔ ISO 20022 XML")
                        create_agent_tab(
                            "Agent 5 - Convert", 
                            run_agent_5_convert, 
                            is_judge=False,
                            disabled_endpoints={"llm_pro_finance": "Tool calls not yet supported (coming soon)"}
                        )
                    
                    with gr.TabItem("Validate"):
                        gr.Markdown("**Message validation:** Check structure, format, and required fields")
                        create_agent_tab(
                            "Agent 5 - Validate", 
                            run_agent_5_validate, 
                            is_judge=False,
                            disabled_endpoints={"llm_pro_finance": "Tool calls not yet supported (coming soon)"}
                        )
                    
                    with gr.TabItem("Risk Assessment"):
                        gr.Markdown("**AML/KYC risk scoring:** Evaluate transaction risk indicators")
                        create_agent_tab(
                            "Agent 5 - Risk", 
                            run_agent_5_risk, 
                            is_judge=False,
                            disabled_endpoints={"llm_pro_finance": "Tool calls not yet supported (coming soon)"}
                        )
            
            with gr.TabItem("Judge (70B)"):
                create_agent_tab("Agent 6", run_agent_6, is_judge=True)
        
        # Footer with settings info and PydanticAI link
        gr.HTML("""
        <div style='margin-top: 30px; padding: 20px; border-top: 1px solid #e5e7eb; font-size: 13px; color: #6b7280;'>
            <div style='display: flex; justify-content: space-between; align-items: center;'>
                <div>
                    <strong>Configuration:</strong> Edit <code style='background: #f3f4f6; padding: 2px 6px; border-radius: 4px;'>.env</code> file to set ENDPOINT, LLM_PRO_FINANCE_URL, LLM_PRO_FINANCE_KEY
                </div>
                <div>
                    Built with <a href="https://ai.pydantic.dev/" target="_blank" style="color: #3b82f6; text-decoration: none;">Pydantic AI</a>
                </div>
            </div>
        </div>
        """)
    
    return app


if __name__ == "__main__":
    app = create_interface()
    app.launch(server_name="0.0.0.0", server_port=7860)
