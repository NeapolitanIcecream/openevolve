"""
This module implements the ToolRegistry and DiscoveredTool classes for managing
and executing tools discovered from the environment.
"""

import asyncio
import json
import shlex
from typing import Any, Dict, List, Optional

from .base import BaseTool, Icon, Tool, ToolResult, Schema
from .edit import EditTool
from .read_file import ReadFileTool
from .read_many_files import ReadManyFilesTool
from .config import Config as ToolConfig
from ..llm.base import LLMInterface
from .submit import SubmitTool
from .glob import GlobTool
from .grep import GrepTool
from .ls import LSTool
from .lint_file import LintFileTool
from .write_file import WriteFileTool


# A simple type alias for the config object for now.
Config = Dict[str, Any]
MAX_STDOUT_SIZE = 10 * 1024 * 1024  # 10MB limit


class DiscoveredTool(BaseTool):
    """
    A tool discovered by executing a command from the project configuration.
    """

    def __init__(
        self,
        config: Config,
        name: str,
        description: str,
        parameter_schema: Schema,
    ):
        self.config = config
        discovery_cmd = self.config.get("tool_discovery_command", "")
        call_command = self.config.get("tool_call_command", "")

        full_description = f"""{description}

This tool was discovered from the project by executing the command `{discovery_cmd}` on project root.
When called, this tool will execute the command `{call_command} {name}` on project root.
Tool discovery and call commands can be configured in project or user settings.

When called, the tool call command is executed as a subprocess.
On success, tool output is returned as a json string.
Otherwise, the following information is returned:

Stdout: Output on stdout stream. Can be `(empty)` or partial.
Stderr: Output on stderr stream. Can be `(empty)` or partial.
Error: Error or `(none)` if no error was reported for the subprocess.
Exit Code: Exit code or `(none)` if terminated by signal.
Signal: Signal number or `(none)` if no signal was received.
"""
        super().__init__(
            name=name,
            display_name=name,
            description=full_description,
            icon=Icon.HAMMER,
            parameter_schema=parameter_schema,
            is_output_markdown=False,
            can_update_output=False,
        )

    async def execute(self, params: Dict[str, Any]) -> ToolResult:
        call_command_str = self.config.get("tool_call_command")
        if not call_command_str:
            raise ValueError("Tool call command is not configured.")

        try:
            # Use shlex to split the base command and add the tool name as an argument
            cmd_parts = shlex.split(call_command_str)
            cmd_parts.append(self.name)

            proc = await asyncio.create_subprocess_exec(
                *cmd_parts,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except Exception as e:
            error_msg = f"Error creating subprocess: {e}"
            return ToolResult(llm_content=error_msg, return_display=error_msg)

        # Write params to stdin
        if proc.stdin:
            proc.stdin.write(json.dumps(params).encode())
            await proc.stdin.drain()
            proc.stdin.close()

        # Read stdout and stderr
        stdout_bytes, stderr_bytes = await proc.communicate()
        returncode = proc.returncode

        # The TS version checks for a signal, but asyncio's communicate() doesn't
        # directly expose the signal. The return code will be negative if killed by a signal.
        signal_info = (
            f"Signal: killed by signal {-returncode}" if returncode is not None and returncode < 0 else "Signal: (none)"
        )

        if returncode != 0 or stderr_bytes:
            stdout_str = stdout_bytes.decode() if stdout_bytes else '(empty)'
            stderr_str = stderr_bytes.decode() if stderr_bytes else '(empty)'
            llm_content = (
                f"Stdout: {stdout_str}\n"
                f"Stderr: {stderr_str}\n"
                f"Exit Code: {returncode if returncode is not None else '(none)'}\n"
                f"{signal_info}"
            )
            return ToolResult(llm_content=llm_content, return_display=llm_content)

        stdout_str = stdout_bytes.decode() if stdout_bytes else ""
        return ToolResult(llm_content=stdout_str, return_display=stdout_str)


class ToolRegistry:
    """
    Manages the registration and discovery of tools.
    """

    def __init__(self, config: Config, llm_client: Optional[LLMInterface] = None, evaluator: Optional[Any] = None, write_llm_client: Optional[LLMInterface] = None):
        self.config = config
        self.tool_config = ToolConfig(root_dir=self.config.get("root_dir") or ".")
        self._tools: Dict[str, Tool] = {}
        self.llm_client = llm_client
        self.write_llm_client = write_llm_client
        self._evaluator = evaluator
        self._register_builtin_tools()

    def _register_builtin_tools(self):
        """Registers all the built-in tools."""
        root_dir = self.config.get("root_dir")
        if not root_dir:
            print("Warning: 'root_dir' not found in config. EditTool may not work correctly.")
            root_dir = "."

        if self.write_llm_client is not None:
            self.register_tool(EditTool(root_dir=root_dir, llm_client=self.write_llm_client))
        elif self.llm_client is not None:
            self.register_tool(EditTool(root_dir=root_dir, llm_client=self.llm_client))
        
        # WriteFileTool does not require an LLM client
        self.register_tool(WriteFileTool(root_dir=root_dir))

        self.register_tool(ReadFileTool(config=self.tool_config))
        self.register_tool(ReadManyFilesTool(config=self.tool_config))
        # File system discovery tools
        self.register_tool(GlobTool(config=self.tool_config))
        self.register_tool(GrepTool(config=self.tool_config))
        self.register_tool(LSTool(config=self.tool_config))
        self.register_tool(LintFileTool(config=self.tool_config))
        if self._evaluator is not None:
            self.register_tool(SubmitTool(evaluator=self._evaluator, config=self.tool_config))

    def set_llm_client(self, llm_client: LLMInterface):
        """Sets the LLM client and re-registers tools that require it."""
        self.llm_client = llm_client
        self._register_builtin_tools()

    def set_write_llm_client(self, llm_client: LLMInterface):
        """Sets the dedicated write LLM client (for file editing) and re-registers relevant tools."""
        self.write_llm_client = llm_client
        self._register_builtin_tools()

    def set_evaluator(self, evaluator: Any):
        """Sets evaluator for the evaluation tool and re-registers it."""
        self._evaluator = evaluator
        self._register_builtin_tools()

    def register_tool(self, tool: Tool):
        """Registers a tool definition."""
        if tool.name in self._tools:
            print(f"Warning: Tool with name '{tool.name}' is already registered. Overwriting.")
        self._tools[tool.name] = tool

    async def discover_tools(self):
        """
        Discovers tools from the project by running the discovery command.
        """
        # Remove any previously discovered tools
        self._tools = {
            name: tool for name, tool in self._tools.items() if not isinstance(tool, DiscoveredTool)
        }

        discovery_cmd = self.config.get("tool_discovery_command")
        if not discovery_cmd:
            return

        try:
            cmd_parts = shlex.split(discovery_cmd)
            if not cmd_parts:
                raise ValueError("Tool discovery command is empty.")

            proc = await asyncio.create_subprocess_exec(
                *cmd_parts,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            # Stream stdout to prevent memory issues
            stdout_chunks = []
            stdout_size = 0
            if proc.stdout:
                while True:
                    chunk = await proc.stdout.read(4096)  # Read in 4KB chunks
                    if not chunk:
                        break
                    stdout_size += len(chunk)
                    if stdout_size > MAX_STDOUT_SIZE:
                        proc.kill()
                        raise RuntimeError(
                            f"Tool discovery command output exceeded size limit of {MAX_STDOUT_SIZE} bytes."
                        )
                    stdout_chunks.append(chunk)

            stdout = b"".join(stdout_chunks)
            stderr = await proc.stderr.read() if proc.stderr else b""
            await proc.wait()

            if proc.returncode != 0:
                print(f"Command failed with code {proc.returncode}")
                print(stderr.decode())
                raise RuntimeError(
                    f"Tool discovery command failed with exit code {proc.returncode}"
                )

            discovered_items = json.loads(stdout.decode().strip())
            if not isinstance(discovered_items, list):
                raise TypeError("Tool discovery command must return a JSON array of tools.")

            # Only accept OpenAI-style: {"type":"function","function":{"name":...,"description":...,"parameters":{...}}}
            functions: List[Schema] = []
            for item in discovered_items:
                if not isinstance(item, dict):
                    continue
                if item.get("type") == "function" and isinstance(item.get("function"), dict):
                    fn = item["function"]
                    if fn.get("name"):
                        functions.append(fn)
                elif item.get("name"):
                    # Compatibility removed; only accept when input is directly a function object
                    functions.append(item)

            for func in functions:
                if not func.get("name"):
                    print("Warning: Discovered a tool with no name. Skipping.")
                    continue

                parameters = func.get("parameters") or {}
                if not isinstance(parameters, dict):
                    parameters = {}

                self.register_tool(
                    DiscoveredTool(
                        config=self.config,
                        name=func["name"],
                        description=func.get("description", ""),
                        parameter_schema=parameters,
                    )
                )

        except (FileNotFoundError, ValueError, TypeError, RuntimeError, json.JSONDecodeError) as e:
            print(f"Tool discovery command '{discovery_cmd}' failed: {e}")
            raise

    async def discover_mcp_tools(self):
        """(Placeholder) Discover tools from MCP servers."""
        print("MCP tool discovery is not yet implemented in the Python version.")
        await asyncio.sleep(0)  # To make it awaitable

    async def discover_all_tools(self):
        """Discovers all tools from command and MCP servers."""
        # First, discover from command
        await self.discover_tools()
        # Then, discover from MCP
        await self.discover_mcp_tools()

    def get_tool(self, name: str) -> Optional[Tool]:
        """Get the definition of a specific tool."""
        return self._tools.get(name)

    def get_all_tools(self) -> List[Tool]:
        """Returns an array of all registered and discovered tool instances."""
        return sorted(self._tools.values(), key=lambda t: t.display_name)

    def get_function_declarations(self) -> List[Schema]:
        """Retrieves the list of tool schemas (FunctionDeclaration array)."""
        return [tool.schema for tool in self.get_all_tools()]

    def get_tool_specs(self) -> List[Dict[str, Any]]:
        """
        Retrieves the list of tool specifications in the format expected by OpenAI.
        """
        return [
            {"type": "function", "function": schema}
            for schema in self.get_function_declarations()
        ]
