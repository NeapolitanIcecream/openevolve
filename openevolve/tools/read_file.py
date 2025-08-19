"""
Implementation of the ReadFile tool.
"""

import os
from typing import Any, Dict, List

from .base import BaseTool, Icon, ToolLocation, ToolResult
from .config import Config
from .file_service import FileService
from .file_utils import process_single_file_content, is_within_root


class ReadFileTool(BaseTool):
    """
    Tool to read the content of a specified file.
    """

    def __init__(self, config: Config):
        super().__init__(
            name="read_file",
            display_name="ReadFile",
            description="Reads and returns the content of a specified file from the local filesystem. Handles text, images (PNG, JPG, GIF, WEBP, SVG, BMP), and PDF files. For text files, it can read specific line ranges.",
            icon=Icon.FILE_SEARCH,
            parameter_schema={
                "type": "object",
                "properties": {
                    "absolute_path": {
                        "description": "The absolute path to the file to read (e.g., '/home/user/project/file.txt'). Relative paths are not supported. You must provide an absolute path.",
                        "type": "string",
                    },
                    "offset": {
                        "description": "Optional: For text files, the 0-based line number to start reading from. Requires 'limit' to be set. Use for paginating through large files.",
                        "type": "integer",
                    },
                    "limit": {
                        "description": "Optional: For text files, maximum number of lines to read. Use with 'offset' to paginate through large files. If omitted, reads the entire file (if feasible, up to a default limit).",
                        "type": "integer",
                    },
                },
                "required": ["absolute_path"],
            },
        )
        self.config = config
        self.file_service = FileService(config)

    def validate_tool_params(self, params: Dict[str, Any]) -> str | None:
        file_path = params.get("absolute_path")
        if not file_path or not isinstance(file_path, str):
            return "Missing or invalid `absolute_path` parameter."

        if not os.path.isabs(file_path):
            return f"File path must be absolute, but was relative: {file_path}. You must provide an absolute path."

        if not is_within_root(file_path, self.config.root_dir):
            return (
                f"File path must be within the root directory ({self.config.root_dir}): {file_path}"
            )

        if params.get("offset") is not None and params["offset"] < 0:
            return "Offset must be a non-negative number"

        if params.get("limit") is not None and params["limit"] <= 0:
            return "Limit must be a positive number"

        return None

    def get_description(self, params: Dict[str, Any]) -> str:
        if not params or not params.get("absolute_path"):
            return "Path unavailable"
        relative_path = os.path.relpath(params["absolute_path"], self.config.root_dir)
        # A shortenPath equivalent can be implemented if necessary
        return relative_path

    def tool_locations(self, params: Dict[str, Any]) -> List[ToolLocation]:
        if not params or not params.get("absolute_path"):
            return []
        return [ToolLocation(path=params["absolute_path"], line=params.get("offset"))]

    async def execute(self, params: Dict[str, Any]) -> ToolResult:
        validation_error = self.validate_tool_params(params)
        if validation_error:
            return ToolResult(
                llm_content=f"Error: Invalid parameters provided. Reason: {validation_error}",
                return_display=validation_error,
            )

        result = process_single_file_content(
            params["absolute_path"],
            self.config.root_dir,
            params.get("offset"),
            params.get("limit"),
        )

        if result.get("error"):
            return ToolResult(llm_content=str(result["error"]), return_display=result["returnDisplay"])

        return ToolResult(llm_content=str(result["llmContent"]), return_display=result["returnDisplay"])
