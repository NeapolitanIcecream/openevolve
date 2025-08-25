"""
Defines the WriteFileTool for fully overwriting or creating files.
"""

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

# Note: This tool uses the `diff-match-patch` library (already used by EditTool).
# Prefer using uv to install: `uv add diff-match-patch` (or `pip install diff-match-patch`)
import diff_match_patch as dmp_module

from .base import BaseTool, FileDiff, Icon, ToolLocation, ToolResult, ToolResultDisplay


@dataclass
class WriteFileParams:
    file_path: str
    content: str
    encoding: str = "utf-8"


class WriteFileTool(BaseTool):
    """
    A tool for writing full content to a file, creating it if it does not exist.
    It does not attempt to preserve or replace parts of the file; it simply writes
    the provided content in full. When overwriting an existing file, a textual diff
    is returned for display, similar to EditTool.
    """

    def __init__(self, root_dir: str):
        super().__init__(
            name="write_file",
            display_name="Write File",
            description=(
                "Writes the provided full content to a file, creating directories and the file if missing."
            ),
            icon=Icon.PENCIL,
            parameter_schema={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Absolute path to the file to write.",
                    },
                    "content": {
                        "type": "string",
                        "description": "Full file content to write.",
                    },
                    "encoding": {
                        "type": "string",
                        "description": "Text encoding used for reading (if exists) and writing.",
                        "default": "utf-8",
                    },
                },
                "required": ["file_path", "content"],
            },
            is_output_markdown=False,
            can_update_output=False,
        )
        self.root_dir = root_dir

    def validate_tool_params(self, params: Dict[str, Any]) -> Optional[str]:
        file_path = params.get("file_path")
        if not file_path or not isinstance(file_path, str):
            return "Missing or invalid `file_path` parameter."

        if not os.path.isabs(file_path):
            return f"File path must be absolute: {file_path}"

        if not file_path.startswith(self.root_dir):
            return f"File path must be within the root directory ({self.root_dir}): {file_path}"

        content = params.get("content")
        if content is None or not isinstance(content, str):
            return "Missing or invalid `content` parameter."

        return None

    def tool_locations(self, params: Dict[str, Any]) -> List[ToolLocation]:
        return [ToolLocation(path=params["file_path"])]

    def get_description(self, params: Dict[str, Any]) -> str:
        relative_path = os.path.relpath(params["file_path"], self.root_dir)
        return f"Write {relative_path}"

    async def should_confirm_execute(self, params: Dict[str, Any]) -> Any:
        # Per requirements, user interaction is not implemented.
        return False

    async def execute(self, params: Dict[str, Any]) -> ToolResult:
        validation_error = self.validate_tool_params(params)
        if validation_error:
            return ToolResult(
                llm_content=validation_error, return_display=f"Error: {validation_error}"
            )

        tool_params = WriteFileParams(**params)

        try:
            # Read current content if the file exists, for diff display
            current_content: Optional[str] = None
            is_new_file = False
            try:
                with open(tool_params.file_path, "r", encoding=tool_params.encoding) as f:
                    # Normalize line endings for diff clarity
                    current_content = f.read().replace("\r\n", "\n")
            except FileNotFoundError:
                is_new_file = True

            # Ensure parent directory exists
            dir_name = os.path.dirname(tool_params.file_path)
            if not os.path.exists(dir_name):
                os.makedirs(dir_name, exist_ok=True)

            # Write new content
            with open(tool_params.file_path, "w", encoding=tool_params.encoding) as f:
                f.write(tool_params.content)

            if is_new_file:
                display_result: ToolResultDisplay = (
                    f"Created {os.path.relpath(tool_params.file_path, self.root_dir)}"
                )
                llm_message = f"Created new file: {tool_params.file_path}"
            else:
                dmp = dmp_module.diff_match_patch()
                diff = dmp.patch_make(current_content or "", tool_params.content)
                patch_text = dmp.patch_toText(diff)
                display_result = FileDiff(
                    file_diff=patch_text,
                    file_name=os.path.basename(tool_params.file_path),
                    original_content=current_content,
                    new_content=tool_params.content,
                )
                llm_message = f"Successfully wrote file: {tool_params.file_path} (overwritten)."

            return ToolResult(llm_content=llm_message, return_display=display_result)

        except Exception as e:
            return ToolResult(
                llm_content=f"Error executing write_file: {e}",
                return_display=f"Error writing file: {e}",
            )


