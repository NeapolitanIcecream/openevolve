"""
Defines the base classes and interfaces for tools.
"""

import abc
import json
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Union

# Using a TypeDict for schema for now, can be replaced with a more specific
# dataclass or class if needed, mirroring FunctionDeclaration from @google/genai.
Schema = Dict[str, Any]


class Icon(str, Enum):
    """Enum for tool icons, matching the TypeScript version."""

    FILE_SEARCH = "fileSearch"
    FOLDER = "folder"
    GLOBE = "globe"
    HAMMER = "hammer"
    LIGHT_BULB = "lightBulb"
    PENCIL = "pencil"
    REGEX = "regex"
    TERMINAL = "terminal"


@dataclass
class FileDiff:
    """Represents a diff for a file."""

    file_diff: str
    file_name: str
    original_content: str | None
    new_content: str


ToolResultDisplay = Union[str, FileDiff]


@dataclass
class ToolResult:
    """
    Represents the result of a tool execution.
    """

    llm_content: Union[str, List[Union[str, Dict[str, Any]]]]
    return_display: ToolResultDisplay
    summary: str | None = None


@dataclass
class ToolLocation:
    """Represents a file path and optional line number."""

    path: str
    line: int | None = None


class Tool(abc.ABC):
    """
    Abstract Base Class for all tools.
    """

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """The internal name of the tool."""
        pass

    @property
    @abc.abstractmethod
    def display_name(self) -> str:
        """The user-friendly display name of the tool."""
        pass

    @property
    @abc.abstractmethod
    def description(self) -> str:
        """Description of what the tool does."""
        pass

    @property
    @abc.abstractmethod
    def icon(self) -> Icon:
        """The icon for the tool."""
        pass

    @property
    @abc.abstractmethod
    def schema(self) -> Schema:
        """Function declaration schema."""
        pass

    @property
    @abc.abstractmethod
    def is_output_markdown(self) -> bool:
        """Whether the tool's output should be rendered as markdown."""
        pass

    @property
    @abc.abstractmethod
    def can_update_output(self) -> bool:
        """Whether the tool supports live (streaming) output."""
        pass

    @abc.abstractmethod
    def validate_tool_params(self, params: Dict[str, Any]) -> str | None:
        """Validates the parameters for the tool."""
        pass

    @abc.abstractmethod
    def get_description(self, params: Dict[str, Any]) -> str:
        """Gets a pre-execution description of the tool operation."""
        pass

    @abc.abstractmethod
    def tool_locations(self, params: Dict[str, Any]) -> List[ToolLocation]:
        """Determines what file system paths the tool will affect."""
        pass

    @abc.abstractmethod
    async def should_confirm_execute(self, params: Dict[str, Any]) -> Any:
        """Determines if the tool should prompt for confirmation."""
        pass

    @abc.abstractmethod
    async def execute(self, params: Dict[str, Any]) -> ToolResult:
        """Executes the tool with the given parameters."""
        pass


class BaseTool(Tool):
    """
    Base implementation for tools with common functionality.
    """

    def __init__(
        self,
        name: str,
        display_name: str,
        description: str,
        icon: Icon,
        parameter_schema: Schema,
        is_output_markdown: bool = True,
        can_update_output: bool = False,
    ):
        self._name = name
        self._display_name = display_name
        self._description = description
        self._icon = icon
        self._parameter_schema = parameter_schema
        self._is_output_markdown = is_output_markdown
        self._can_update_output = can_update_output

    @property
    def name(self) -> str:
        return self._name

    @property
    def display_name(self) -> str:
        return self._display_name

    @property
    def description(self) -> str:
        return self._description

    @property
    def icon(self) -> Icon:
        return self._icon

    @property
    def schema(self) -> Schema:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self._parameter_schema,
        }

    @property
    def is_output_markdown(self) -> bool:
        return self._is_output_markdown

    @property
    def can_update_output(self) -> bool:
        return self._can_update_output

    def validate_tool_params(self, params: Dict[str, Any]) -> str | None:
        # Placeholder implementation.
        return None

    def get_description(self, params: Dict[str, Any]) -> str:
        # Default implementation.
        return json.dumps(params)

    def tool_locations(self, params: Dict[str, Any]) -> List[ToolLocation]:
        # Default implementation.
        return []

    async def should_confirm_execute(self, params: Dict[str, Any]) -> Any:
        # Default implementation.
        return False

    @abc.abstractmethod
    async def execute(self, params: Dict[str, Any]) -> ToolResult:
        """Executes the tool with the given parameters."""
        pass
