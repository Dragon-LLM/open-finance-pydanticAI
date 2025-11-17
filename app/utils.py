"""Utility functions for handling reasoning model responses."""

import re


def extract_answer_from_reasoning(response: str) -> str:
    """Extract the final answer from a response containing reasoning tags.
    
    The Qwen3 model returns responses in the format:
    <think>...reasoning...</think>
    Final answer here...
    
    Or sometimes just the reasoning tags without closing tag.
    This function extracts only the final answer part.
    """
    if not response:
        return ""
    
    # Method 1: Split on </think> tag (most common format)
    if "</think>" in response:
        parts = response.split("</think>", 1)
        if len(parts) > 1:
            return parts[1].strip()
    
    # Method 2: Remove reasoning tags and their content
    # Match <think>...</think> (case insensitive, multi-line)
    cleaned = re.sub(
        r'<think>.*?</think>',
        '',
        response,
        flags=re.DOTALL | re.IGNORECASE
    )
    
    # Clean up any remaining whitespace
    cleaned = cleaned.strip()
    
    # If we removed everything, return original (fallback)
    if not cleaned:
        return response.strip()
    
    return cleaned


def extract_key_terms(text: str) -> list[str]:
    """Extract key financial terms from text.
    
    This is a simple heuristic - could be improved with NLP.
    """
    # Common French financial terms patterns
    financial_patterns = [
        r'\bcrédit\b', r'\bprêt\b', r'\bdette\b', r'\bintérêt\b',
        r'\btaux\b', r'\bcapital\b', r'\bdividende\b', r'\baction\b',
        r'\bobligation\b', r'\bfonds\b', r'\bépargne\b', r'\binvestissement\b',
        r'\bhypothèque\b', r'\bamortissement\b', r'\bvalorisation\b',
        r'\bdate de valeur\b', r'\bescompte\b', r'\bconsignation\b',
        r'\bmain levée\b', r'\bséquestre\b', r'\bnantissement\b',
    ]
    
    found_terms = []
    text_lower = text.lower()
    
    for pattern in financial_patterns:
        if re.search(pattern, text_lower):
            # Extract the matched term
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                term = match.group(0).strip()
                if term not in found_terms:
                    found_terms.append(term)
    
    return found_terms[:10]  # Limit to 10 terms

