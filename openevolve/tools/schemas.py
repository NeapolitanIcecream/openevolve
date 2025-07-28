"""
This module provides the sanitize_parameters function to ensure schema
compatibility with the Gemini API.
"""

from typing import Any, Dict, MutableSet

# Assuming a simplified representation for Schema for now.
Schema = Dict[str, Any]


def _sanitize_parameters_recursive(schema: Schema, visited: MutableSet[int]):
    """
    Internal recursive implementation for sanitize_parameters.

    Args:
        schema: The schema object to sanitize.
        visited: A set used to track visited schema objects (by their id)
                 during recursion to handle circular references.
    """
    if not schema or id(schema) in visited:
        return
    visited.add(id(schema))

    if "anyOf" in schema:
        # Vertex AI gets confused if both anyOf and default are set.
        schema.pop("default", None)
        for item in schema["anyOf"]:
            if isinstance(item, dict):
                _sanitize_parameters_recursive(item, visited)

    if "items" in schema and isinstance(schema["items"], dict):
        _sanitize_parameters_recursive(schema["items"], visited)

    if "properties" in schema:
        for item in schema["properties"].values():
            if isinstance(item, dict):
                _sanitize_parameters_recursive(item, visited)

    # Handle enum values - Gemini API only allows enum for STRING type
    if "enum" in schema and isinstance(schema["enum"], list):
        # In Python, we can be more flexible, but for strict compatibility:
        # if schema.get("type") != "string":
        #     schema["type"] = "string"

        # Filter out null and undefined values, then convert to strings.
        schema["enum"] = [str(value) for value in schema["enum"] if value is not None]
        # Ensure type is string if enum is present
        schema["type"] = "string"

    # Vertex AI only supports 'enum' and 'date-time' for STRING format.
    if schema.get("type") == "string":
        if schema.get("format") not in ["enum", "date-time"]:
            schema.pop("format", None)


def sanitize_parameters(schema: Schema | None):
    """
    Sanitizes a schema object in-place to ensure compatibility with the Gemini API.

    NOTE: This function mutates the passed schema object.

    It performs the following actions:
    - Removes the `default` property when `anyOf` is present.
    - Removes unsupported `format` values from string properties.
    - Recursively sanitizes nested schemas.
    - Handles circular references within the schema.

    Args:
        schema: The schema object to sanitize. It will be modified directly.
    """
    if schema is None:
        return
    _sanitize_parameters_recursive(schema, set())
