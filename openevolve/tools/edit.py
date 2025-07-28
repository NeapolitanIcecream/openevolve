"""
Defines the EditTool for replacing text in files.
"""

import asyncio
import os
from dataclasses import dataclass
from typing import Any, Dict, List, NamedTuple, Optional

# Note: This tool requires the `diff-match-patch` library.
# Please ensure it is installed: pip install diff-match-patch
import diff_match_patch as dmp_module

from .base import BaseTool, FileDiff, Icon, ToolLocation, ToolResult, ToolResultDisplay
from .edit_corrector import CorrectedEditResult, ensure_correct_edit
from .file_service import FileService, file_service
from openevolve.llm.base import LLMInterface

# --- Data Structures --- #


@dataclass
class EditToolParams:
    file_path: str
    old_string: str
    new_string: str
    expected_replacements: int = 1
    modified_by_user: bool = False


class CalculatedEdit(NamedTuple):
    current_content: Optional[str]
    new_content: str
    occurrences: int
    is_new_file: bool
    error: Optional[Dict[str, str]] = None


# --- Tool Implementation --- #


class EditTool(BaseTool):
    """
    A tool for editing files by replacing a specific string.
    """

    def __init__(self, root_dir: str, llm_client: LLMInterface):
        super().__init__(
            name="edit",
            display_name="Edit",
            description="Edits a file by replacing a string. To create a new file, leave old_string empty.",
            icon=Icon.PENCIL,
            parameter_schema={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "The path to the file to edit.",
                    },
                    "old_string": {
                        "type": "string",
                        "description": "The exact string to be replaced. If empty, a new file will be created.",
                    },
                    "new_string": {
                        "type": "string",
                        "description": "The string to replace the old_string with.",
                    },
                    "expected_replacements": {
                        "type": "integer",
                        "description": "The number of occurrences of old_string that are expected to be replaced.",
                        "default": 1,
                    },
                },
                "required": ["file_path", "old_string", "new_string"],
            },
            is_output_markdown=False,
            can_update_output=False,
        )
        self.root_dir = root_dir
        self.llm_client = llm_client

    def validate_tool_params(self, params: Dict[str, Any]) -> Optional[str]:
        file_path = params.get("file_path")
        if not file_path or not isinstance(file_path, str):
            return "Missing or invalid `file_path` parameter."

        if not os.path.isabs(file_path):
            return f"File path must be absolute: {file_path}"

        if not file_path.startswith(self.root_dir):
            return f"File path must be within the root directory ({self.root_dir}): {file_path}"

        return None

    def tool_locations(self, params: Dict[str, Any]) -> List[ToolLocation]:
        return [ToolLocation(path=params["file_path"])]

    def get_description(self, params: Dict[str, Any]) -> str:
        relative_path = os.path.relpath(params["file_path"], self.root_dir)
        if not params.get("old_string"):
            return f"Create {relative_path}"
        return f"Edit {relative_path}"

    async def should_confirm_execute(self, params: Dict[str, Any]) -> Any:
        # Per requirements, user interaction is not implemented.
        return False

    def _apply_replacement(
        self, current_content: Optional[str], old: str, new: str, is_new_file: bool
    ) -> str:
        if is_new_file:
            return new
        if current_content is None:
            return "" if old else new
        if not old:
            return current_content
        return current_content.replace(old, new)

    async def _calculate_edit(
        self, params: EditToolParams, abort_signal: asyncio.Event
    ) -> CalculatedEdit:
        current_content: Optional[str] = None
        file_exists = False
        is_new_file = False
        error: Optional[Dict[str, str]] = None

        try:
            with open(params.file_path, "r", encoding="utf-8") as f:
                current_content = f.read().replace("\r\n", "\n")
            file_exists = True
        except FileNotFoundError:
            file_exists = False
        except Exception as e:
            raise e  # Rethrow other FS errors

        if not params.old_string and not file_exists:
            is_new_file = True
            final_old_string = params.old_string
            final_new_string = params.new_string
            occurrences = 0
        elif not file_exists:
            error = {
                "display": "File not found. Cannot apply edit.",
                "raw": f"File not found: {params.file_path}",
            }
            final_old_string = params.old_string
            final_new_string = params.new_string
            occurrences = 0
        elif current_content is not None:
            corrected = await ensure_correct_edit(
                params.file_path, current_content, params.__dict__, self.llm_client, abort_signal
            )
            final_old_string = corrected.params.old_string
            final_new_string = corrected.params.new_string
            occurrences = corrected.occurrences

            if not params.old_string:
                error = {
                    "display": "File already exists. Cannot create.",
                    "raw": f"File already exists: {params.file_path}",
                }
            elif occurrences == 0:
                error = {
                    "display": "Failed to edit, could not find the string to replace.",
                    "raw": "The exact text in old_string was not found.",
                }
            elif occurrences != params.expected_replacements:
                error = {
                    "display": f"Expected {params.expected_replacements} occurrences but found {occurrences}.",
                    "raw": f"Expected {params.expected_replacements} but found {occurrences} for old_string in {params.file_path}",
                }
            elif final_old_string == final_new_string:
                error = {
                    "display": "No changes to apply. The old and new strings are identical.",
                    "raw": "No changes to apply. The old_string and new_string are identical.",
                }
        else:
            error = {"display": "Failed to read file content.", "raw": "Could not read file."}
            final_old_string = params.old_string
            final_new_string = params.new_string
            occurrences = 0

        new_content = self._apply_replacement(
            current_content, final_old_string, final_new_string, is_new_file
        )

        return CalculatedEdit(
            current_content=current_content,
            new_content=new_content,
            occurrences=occurrences,
            is_new_file=is_new_file,
            error=error,
        )

    async def execute(self, params: Dict[str, Any]) -> ToolResult:
        validation_error = self.validate_tool_params(params)
        if validation_error:
            return ToolResult(
                llm_content=validation_error, return_display=f"Error: {validation_error}"
            )

        tool_params = EditToolParams(**params)
        abort_signal = asyncio.Event()

        try:
            edit_data = await self._calculate_edit(tool_params, abort_signal)
        except Exception as e:
            return ToolResult(
                llm_content=f"Error preparing edit: {e}", return_display=f"Error: {e}"
            )

        if edit_data.error:
            return ToolResult(
                llm_content=edit_data.error["raw"],
                return_display=f"Error: {edit_data.error['display']}",
            )

        try:
            dir_name = os.path.dirname(tool_params.file_path)
            if not os.path.exists(dir_name):
                os.makedirs(dir_name, exist_ok=True)

            with open(tool_params.file_path, "w", encoding="utf-8") as f:
                f.write(edit_data.new_content)

            if edit_data.is_new_file:
                display_result: ToolResultDisplay = (
                    f"Created {os.path.relpath(tool_params.file_path, self.root_dir)}"
                )
                llm_message = f"Created new file: {tool_params.file_path}"
            else:
                dmp = dmp_module.diff_match_patch()
                diff = dmp.patch_make(edit_data.current_content or "", edit_data.new_content)
                patch_text = dmp.patch_toText(diff)
                display_result = FileDiff(
                    file_diff=patch_text,
                    file_name=os.path.basename(tool_params.file_path),
                    original_content=edit_data.current_content,
                    new_content=edit_data.new_content,
                )
                llm_message = f"Successfully modified file: {tool_params.file_path} ({edit_data.occurrences} replacements)."

            if tool_params.modified_by_user:
                llm_message += (
                    f" User modified the `new_string` content to be: {tool_params.new_string}."
                )

            return ToolResult(llm_content=llm_message, return_display=display_result)

        except Exception as e:
            return ToolResult(
                llm_content=f"Error executing edit: {e}", return_display=f"Error writing file: {e}"
            )
