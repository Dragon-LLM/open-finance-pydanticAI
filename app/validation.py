"""
Shared validation utilities for all agents.

Provides common validation functions for:
- Numeric inputs
- Date formats
- Required fields in structured inputs
- Standardized error messages
"""

from typing import Any, Dict, List, Optional, Union
from datetime import datetime
import re


def validate_numeric_input(
    value: Any,
    field_name: str,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
    allow_zero: bool = True,
    allow_negative: bool = False
) -> Dict[str, Any]:
    """Validate numeric input parameters.
    
    Args:
        value: Value to validate
        field_name: Name of the field (for error messages)
        min_value: Minimum allowed value (optional)
        max_value: Maximum allowed value (optional)
        allow_zero: Whether zero is allowed (default: True)
        allow_negative: Whether negative values are allowed (default: False)
        
    Returns:
        Dict with:
        - valid: bool
        - error: str (if invalid)
        - normalized_value: float (if valid)
    """
    # Try to convert to float
    try:
        num_value = float(value)
    except (ValueError, TypeError):
        return {
            "valid": False,
            "error": f"{field_name} must be a valid number, got: {value}"
        }
    
    # Check for NaN or Infinity
    if not (num_value == num_value):  # NaN check
        return {
            "valid": False,
            "error": f"{field_name} cannot be NaN"
        }
    
    if abs(num_value) == float('inf'):
        return {
            "valid": False,
            "error": f"{field_name} cannot be Infinity"
        }
    
    # Check zero
    if not allow_zero and num_value == 0:
        return {
            "valid": False,
            "error": f"{field_name} cannot be zero"
        }
    
    # Check negative
    if not allow_negative and num_value < 0:
        return {
            "valid": False,
            "error": f"{field_name} cannot be negative, got: {num_value}"
        }
    
    # Check min/max
    if min_value is not None and num_value < min_value:
        return {
            "valid": False,
            "error": f"{field_name} must be >= {min_value}, got: {num_value}"
        }
    
    if max_value is not None and num_value > max_value:
        return {
            "valid": False,
            "error": f"{field_name} must be <= {max_value}, got: {num_value}"
        }
    
    return {
        "valid": True,
        "normalized_value": num_value
    }


def validate_date_format(
    date_str: str,
    field_name: str = "date",
    format: str = "YYYY-MM-DD"
) -> Dict[str, Any]:
    """Validate date format.
    
    Args:
        date_str: Date string to validate
        field_name: Name of the field (for error messages)
        format: Expected format (default: "YYYY-MM-DD")
        
    Returns:
        Dict with:
        - valid: bool
        - error: str (if invalid)
        - parsed_date: datetime (if valid)
    """
    if not isinstance(date_str, str):
        return {
            "valid": False,
            "error": f"{field_name} must be a string, got: {type(date_str).__name__}"
        }
    
    if format == "YYYY-MM-DD":
        # Validate YYYY-MM-DD format
        pattern = r'^\d{4}-\d{2}-\d{2}$'
        if not re.match(pattern, date_str):
            return {
                "valid": False,
                "error": f"{field_name} must be in YYYY-MM-DD format, got: {date_str}"
            }
        
        # Try to parse
        try:
            parsed = datetime.strptime(date_str, "%Y-%m-%d")
            return {
                "valid": True,
                "parsed_date": parsed
            }
        except ValueError as e:
            return {
                "valid": False,
                "error": f"{field_name} is not a valid date: {str(e)}"
            }
    
    # Add other formats as needed
    return {
        "valid": False,
        "error": f"Unsupported date format: {format}"
    }


def validate_required_fields(
    data: Dict[str, Any],
    required_fields: List[str],
    context: str = "input"
) -> Dict[str, Any]:
    """Validate that all required fields are present in structured input.
    
    Args:
        data: Dictionary to validate
        required_fields: List of required field names
        context: Context description (for error messages)
        
    Returns:
        Dict with:
        - valid: bool
        - errors: List[str] (missing fields)
        - warnings: List[str] (empty or None values)
    """
    errors = []
    warnings = []
    
    if not isinstance(data, dict):
        return {
            "valid": False,
            "errors": [f"{context} must be a dictionary, got: {type(data).__name__}"],
            "warnings": []
        }
    
    # Check for missing fields
    for field in required_fields:
        if field not in data:
            errors.append(f"Required field '{field}' is missing in {context}")
        elif data[field] is None:
            warnings.append(f"Field '{field}' is None in {context}")
        elif isinstance(data[field], str) and not data[field].strip():
            warnings.append(f"Field '{field}' is empty in {context}")
    
    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings
    }


def standardize_error_message(
    error_type: str,
    field_name: str,
    details: Optional[str] = None
) -> str:
    """Generate standardized error messages.
    
    Args:
        error_type: Type of error (e.g., "missing", "invalid", "out_of_range")
        field_name: Name of the field
        details: Additional details (optional)
        
    Returns:
        Standardized error message string
    """
    error_templates = {
        "missing": f"Required field '{field_name}' is missing",
        "invalid": f"Invalid value for '{field_name}'",
        "out_of_range": f"Value for '{field_name}' is out of allowed range",
        "type_error": f"'{field_name}' has incorrect type",
        "format_error": f"'{field_name}' has incorrect format"
    }
    
    base_message = error_templates.get(error_type, f"Error with '{field_name}'")
    
    if details:
        return f"{base_message}: {details}"
    
    return base_message


def validate_response_format(
    response: Any,
    expected_type: type,
    required_fields: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Validate response format consistency.
    
    Args:
        response: Response to validate
        expected_type: Expected type (e.g., dict, BaseModel)
        required_fields: List of required fields if response is a dict
        
    Returns:
        Dict with:
        - valid: bool
        - errors: List[str]
        - warnings: List[str]
    """
    errors = []
    warnings = []
    
    # Check type
    if not isinstance(response, expected_type):
        errors.append(
            standardize_error_message(
                "type_error",
                "response",
                f"Expected {expected_type.__name__}, got {type(response).__name__}"
            )
        )
        return {
            "valid": False,
            "errors": errors,
            "warnings": warnings
        }
    
    # If dict, check required fields
    if isinstance(response, dict) and required_fields:
        field_validation = validate_required_fields(response, required_fields, "response")
        errors.extend(field_validation["errors"])
        warnings.extend(field_validation["warnings"])
    
    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings
    }







