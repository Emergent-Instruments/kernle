"""
Kernle MCP Server - Memory operations for Claude Code and other MCP clients.

This exposes Kernle's memory operations as MCP tools,
enabling synthetic intelligences to manage their stratified memory
through the Model Context Protocol.

Security Features:
- Comprehensive input validation and sanitization
- Secure error handling with no information disclosure
- Type safety and schema validation
- Structured logging for debugging

Usage:
    kernle mcp  # Start MCP server (stdio transport)
"""

import asyncio
import json
import logging
from typing import Any, Dict, List

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    TextContent,
    Tool,
)

from kernle.core import Kernle
from kernle.mcp.handlers import HANDLERS, VALIDATORS
from kernle.mcp.tool_definitions import TOOLS, VALID_SOURCE_TYPES  # noqa: F401 — re-export

logger = logging.getLogger(__name__)

# Optional dependency: prefer full JSON Schema validation when available.
try:
    from jsonschema import Draft7Validator
    from jsonschema.exceptions import SchemaError

    _HAS_JSONSCHEMA = True
except Exception as exc:
    logger.debug("Swallowed %s importing jsonschema (optional): %s", type(exc).__name__, exc)
    Draft7Validator = None  # type: ignore[assignment]
    SchemaError = Exception  # type: ignore[assignment]
    _HAS_JSONSCHEMA = False

# Initialize MCP server
mcp = Server("kernle")


# Global stack_id for MCP session
_mcp_stack_id: str = "default"

# Registry for plugin tools (populated via register_plugin_tools)
_plugin_tools: Dict[str, Tool] = {}  # namespaced_name -> Tool
_plugin_handlers: Dict[str, Any] = {}  # namespaced_name -> handler callable
_plugin_schemas: Dict[str, Dict[str, Any]] = {}  # namespaced_name -> schema dict
_plugin_schema_validators: Dict[str, Any] = {}  # namespaced_name -> compiled validator

_PLUGIN_TOOL_MAX_ARGUMENT_BYTES = 64 * 1024


def set_stack_id(stack_id: str) -> None:
    """Set the agent ID for this MCP session."""
    global _mcp_stack_id
    _mcp_stack_id = stack_id
    # Clear cached instance so next get_kernle uses new stack_id
    if hasattr(get_kernle, "_instance"):
        delattr(get_kernle, "_instance")


def get_kernle() -> Kernle:
    """Get or create Kernle instance."""
    if not hasattr(get_kernle, "_instance"):
        get_kernle._instance = Kernle(_mcp_stack_id)  # type: ignore[attr-defined]
    return get_kernle._instance  # type: ignore[attr-defined]


def register_plugin_tools(plugin_name: str, tools: list) -> None:
    """Register a plugin's tools with the MCP server.

    Tools are namespaced as ``{plugin_name}.{tool_name}`` to avoid
    collisions with built-in tools or other plugins.
    """
    for td in tools:
        namespaced = f"{plugin_name}.{td.name}"
        schema = td.input_schema
        if not isinstance(schema, dict):
            raise ValueError(f"Plugin tool '{namespaced}' must provide an input schema dictionary.")

        schema_type = schema.get("type")
        if not schema_type:
            raise ValueError(f"Plugin tool '{namespaced}' must provide an input schema.")
        if schema_type != "object":
            raise ValueError(
                f"Plugin tool '{namespaced}' must use an object input schema, got: {schema_type}"
            )

        if _HAS_JSONSCHEMA:
            try:
                Draft7Validator.check_schema(schema)
            except SchemaError as e:
                raise ValueError(
                    f"Plugin tool '{namespaced}' has invalid schema: {e.message}"
                ) from e

        properties = schema.get("properties")
        if properties is not None and not isinstance(properties, dict):
            raise ValueError(f"Plugin tool '{namespaced}' schema 'properties' must be an object.")
        required = schema.get("required")
        if required is not None and not isinstance(required, list):
            raise ValueError(f"Plugin tool '{namespaced}' schema 'required' must be an array.")

        validator = None
        if _HAS_JSONSCHEMA:
            validator = Draft7Validator(schema)

        _plugin_tools[namespaced] = Tool(
            name=namespaced,
            description=f"[{plugin_name}] {td.description}",
            inputSchema=schema,
        )
        _plugin_schemas[namespaced] = schema
        if validator is not None:
            _plugin_schema_validators[namespaced] = validator

        if td.handler is not None:
            _plugin_handlers[namespaced] = td.handler


def unregister_plugin_tools(plugin_name: str) -> None:
    """Remove all tools registered by a plugin."""
    prefix = f"{plugin_name}."
    to_remove = [name for name in _plugin_tools if name.startswith(prefix)]
    for name in to_remove:
        _plugin_tools.pop(name, None)
        _plugin_handlers.pop(name, None)
        _plugin_schemas.pop(name, None)
        _plugin_schema_validators.pop(name, None)


# =============================================================================
# INPUT VALIDATION & SANITIZATION
# =============================================================================


def _json_type_matches(value: Any, type_name: str) -> bool:
    if type_name == "object":
        return isinstance(value, dict)
    if type_name == "array":
        return isinstance(value, list)
    if type_name == "string":
        return isinstance(value, str)
    if type_name == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if type_name == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if type_name == "boolean":
        return isinstance(value, bool)
    if type_name == "null":
        return value is None
    return True


def _validate_fallback_schema(arguments: Dict[str, Any], schema: Dict[str, Any]) -> None:
    """Minimal schema checks when jsonschema isn't available."""
    required = schema.get("required", [])
    for key in required:
        if key not in arguments:
            raise ValueError(f"Missing required property: {key}")

    properties = schema.get("properties", {})
    additional = schema.get("additionalProperties", True)
    if additional is False:
        allowed_keys = set(properties.keys())
        extras = [key for key in arguments if key not in allowed_keys]
        if extras:
            raise ValueError(f"Unexpected properties: {', '.join(sorted(extras))}")

    for key, value in arguments.items():
        prop_schema = properties.get(key)
        if not isinstance(prop_schema, dict):
            continue

        prop_type = prop_schema.get("type")
        if isinstance(prop_type, list):
            if not any(_json_type_matches(value, candidate) for candidate in prop_type):
                raise ValueError(f"Property '{key}' has invalid type")
        elif isinstance(prop_type, str) and not _json_type_matches(value, prop_type):
            raise ValueError(f"Property '{key}' has invalid type (expected {prop_type})")

        enum_values = prop_schema.get("enum")
        if enum_values is not None and value not in enum_values:
            raise ValueError(f"Property '{key}' must be one of {enum_values}")


def _validate_plugin_tool_input(name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    try:
        payload = json.dumps(arguments, default=str, separators=(",", ":"))
    except (TypeError, ValueError) as e:
        raise ValueError(f"arguments are not JSON-serializable: {e}") from e

    payload_size = len(payload.encode("utf-8"))
    if payload_size > _PLUGIN_TOOL_MAX_ARGUMENT_BYTES:
        raise ValueError(
            f"arguments payload too large ({payload_size} bytes, max {_PLUGIN_TOOL_MAX_ARGUMENT_BYTES})"
        )

    schema = _plugin_schemas.get(name)
    if not isinstance(schema, dict):
        raise ValueError(f"plugin tool metadata missing for: {name}")

    if _HAS_JSONSCHEMA:
        validator = _plugin_schema_validators.get(name)
        if validator is None:
            validator = Draft7Validator(schema)
            _plugin_schema_validators[name] = validator

        errors = sorted(validator.iter_errors(arguments), key=lambda err: list(err.path))
        if errors:
            first = errors[0]
            path = ".".join(str(part) for part in first.path) or "(root)"
            raise ValueError(f"Schema validation failed at {path}: {first.message}")
    else:
        _validate_fallback_schema(arguments, schema)

    return dict(arguments)


def validate_tool_input(name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and sanitize MCP tool inputs."""
    try:
        if not isinstance(name, str):
            raise ValueError(f"tool name must be a string, got {type(name).__name__}")
        if not name:
            raise ValueError("tool name must not be empty")
        if not isinstance(arguments, dict):
            raise ValueError(f"arguments must be an object, got {type(arguments).__name__}")

        validator = VALIDATORS.get(name)
        if validator is not None:
            return validator(arguments)

        # Plugin tools are validated against registered schemas.
        if name in _plugin_handlers:
            return _validate_plugin_tool_input(name, arguments)

        raise ValueError(f"Unknown tool: {name}")

    except (ValueError, TypeError) as e:
        logger.warning(f"Input validation failed for tool {name}: {e}")
        raise ValueError(f"Invalid input: {str(e)}")


def handle_tool_error(e: Exception, tool_name: str, arguments: Dict[str, Any]) -> List[TextContent]:
    """Handle tool errors securely."""
    if isinstance(e, ValueError):
        # Input validation or business logic error
        logger.warning(f"Invalid input for tool {tool_name}: {e}")
        return [TextContent(type="text", text=f"Invalid input: {str(e)}")]

    elif isinstance(e, PermissionError):
        logger.warning(f"Permission denied for tool {tool_name}")
        return [TextContent(type="text", text="Access denied")]

    elif isinstance(e, FileNotFoundError):
        logger.warning(f"Resource not found for tool {tool_name}")
        return [TextContent(type="text", text="Resource not found")]

    elif isinstance(e, ConnectionError):
        logger.error(f"Database connection error for tool {tool_name}")
        return [TextContent(type="text", text="Service temporarily unavailable")]

    else:
        # Unknown error - log full details but return generic message
        argument_keys = list(arguments.keys()) if isinstance(arguments, dict) else []
        logger.error(
            f"Internal error in tool {tool_name}",
            extra={
                "tool_name": tool_name,
                "arguments_keys": argument_keys,
                "error_type": type(e).__name__,
                "error_message": str(e),
            },
            exc_info=True,
        )
        return [TextContent(type="text", text="Internal server error")]


# =============================================================================
# MCP PROTOCOL HANDLERS
# =============================================================================


@mcp.list_tools()
async def list_tools() -> list[Tool]:
    """List available memory tools, including plugin-provided tools."""
    return list(TOOLS) + list(_plugin_tools.values())


@mcp.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    """Handle tool calls with comprehensive validation and error handling."""
    try:
        sanitized_args = validate_tool_input(name, arguments)
        k = get_kernle()

        # Built-in tool handler
        handler = HANDLERS.get(name)
        if handler is not None:
            result = handler(sanitized_args, k)
            return [TextContent(type="text", text=result)]

        # Plugin tool handler
        plugin_handler = _plugin_handlers.get(name)
        if plugin_handler is not None:
            handler_result = plugin_handler(sanitized_args)
            if isinstance(handler_result, str):
                result = handler_result
            else:
                result = json.dumps(handler_result, indent=2, default=str)
            return [TextContent(type="text", text=result)]

        # Should not reach here due to validation, but handle gracefully
        logger.error(f"Unexpected tool name after validation: {name}")
        return [TextContent(type="text", text=f"Tool '{name}' is not available")]

    except Exception as e:
        return handle_tool_error(e, name, arguments)


async def run_server():
    """Run the MCP server."""
    async with stdio_server() as (read_stream, write_stream):
        await mcp.run(
            read_stream,
            write_stream,
            mcp.create_initialization_options(),
        )


def main(stack_id: str = "default"):
    """Entry point for MCP server.

    Stack ID resolution (in order):
    1. Explicit stack_id argument (if not "default")
    2. KERNLE_STACK_ID environment variable
    3. Auto-generated from machine + project path
    """
    from kernle.utils import resolve_stack_id

    # Use resolve_stack_id for consistent fallback logic
    resolved_id = resolve_stack_id(stack_id if stack_id != "default" else None)

    set_stack_id(resolved_id)
    asyncio.run(run_server())


if __name__ == "__main__":
    main()
