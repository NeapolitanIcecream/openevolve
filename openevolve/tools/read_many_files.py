"""
Implementation of the ReadManyFiles tool.
"""

import glob
import os
from typing import Any, Dict, List, Set, Union
import pathspec

from .base import BaseTool, Icon, ToolResult
from .config import Config, DEFAULT_FILE_FILTERING_OPTIONS
from .file_service import FileService
from .file_utils import (
    DEFAULT_ENCODING,
    detect_file_type,
    process_single_file_content,
)
from .schemas import sanitize_parameters

DEFAULT_EXCLUDES: List[str] = [
    "**/node_modules/**",
    "**/.git/**",
    "**/.vscode/**",
    "**/.idea/**",
    "**/dist/**",
    "**/build/**",
    "**/coverage/**",
    "**/__pycache__/**",
    "**/*.pyc",
    "**/*.pyo",
    "**/*.bin",
    "**/*.exe",
    "**/*.dll",
    "**/*.so",
    "**/*.dylib",
    "**/*.class",
    "**/*.jar",
    "**/*.war",
    "**/*.zip",
    "**/*.tar",
    "**/*.gz",
    "**/*.bz2",
    "**/*.rar",
    "**/*.7z",
    "**/*.doc",
    "**/*.docx",
    "**/*.xls",
    "**/*.xlsx",
    "**/*.ppt",
    "**/*.pptx",
    "**/*.odt",
    "**/*.ods",
    "**/*.odp",
    "**/.DS_Store",
    "**/.env",
    "**/GEMINI.md",
]

DEFAULT_OUTPUT_SEPARATOR_FORMAT = "--- {filePath} ---"


class ReadManyFilesTool(BaseTool):
    """Tool to read and concatenate multiple files."""

    def __init__(self, config: Config):
        super().__init__(
            name="read_many_files",
            display_name="ReadManyFiles",
            description="Reads content from multiple files...",  # Truncated for brevity
            icon=Icon.FILE_SEARCH,
            parameter_schema={
                "type": "object",
                "properties": {
                    "paths": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "An array of glob patterns or paths relative to the target directory.",
                    },
                    "include": {"type": "array", "items": {"type": "string"}, "default": []},
                    "exclude": {"type": "array", "items": {"type": "string"}, "default": []},
                    "useDefaultExcludes": {"type": "boolean", "default": True},
                    "file_filtering_options": {
                        "type": "object",
                        "properties": {
                            "respect_git_ignore": {"type": "boolean"},
                            "respect_gemini_ignore": {"type": "boolean"},
                        },
                    },
                },
                "required": ["paths"],
            },
        )
        self.config = config
        self.file_service = FileService(config)

    def validate_tool_params(self, params: Dict[str, Any]) -> str | None:
        # In a real scenario, we would use a more robust schema validation library
        if not isinstance(params.get("paths"), list):
            return "`paths` parameter must be a list of strings."
        if not all(isinstance(p, str) for p in params["paths"]):
            return "All items in `paths` must be strings."
        # Add more validation as needed for other params
        return None

    def get_description(self, params: Dict[str, Any]) -> str:
        all_patterns = params.get("paths", []) + params.get("include", [])
        path_desc = f"using patterns: `{', '.join(all_patterns)}`"

        exclude_patterns = params.get("exclude", [])
        if params.get("useDefaultExcludes", True):
            exclude_patterns.extend(DEFAULT_EXCLUDES)

        exclude_desc = f"Excluding: {len(exclude_patterns)} patterns"
        return f"Read and concatenate files {path_desc}. {exclude_desc}."

    async def execute(self, params: Dict[str, Any]) -> ToolResult:
        input_patterns = params.get("paths", [])
        include = params.get("include", [])
        exclude = params.get("exclude", [])
        use_default_excludes = params.get("useDefaultExcludes", True)

        ff_opts = params.get("file_filtering_options", {})
        respect_git_ignore = ff_opts.get(
            "respect_git_ignore", DEFAULT_FILE_FILTERING_OPTIONS.respect_git_ignore
        )
        respect_gemini_ignore = ff_opts.get(
            "respect_gemini_ignore", DEFAULT_FILE_FILTERING_OPTIONS.respect_gemini_ignore
        )

        effective_excludes = (DEFAULT_EXCLUDES + exclude) if use_default_excludes else exclude
        search_patterns = input_patterns + include

        if not search_patterns:
            return ToolResult(
                llm_content="No search paths provided.",
                return_display="No search paths specified.",
            )

        all_found_files: Set[str] = set()
        for pattern in search_patterns:
            full_pattern = os.path.join(self.config.root_dir, pattern)
            found = glob.glob(full_pattern, recursive=True)
            all_found_files.update([os.path.abspath(p) for p in found if os.path.isfile(p)])

        # Filter based on exclude patterns
        spec = pathspec.PathSpec.from_lines("gitwildmatch", effective_excludes)
        included_files = [
            p
            for p in all_found_files
            if not spec.match_file(os.path.relpath(p, self.config.root_dir))
        ]

        # Filter based on ignore files
        final_filtered_entries = self.file_service.filter_files(
            included_files, respect_git_ignore, respect_gemini_ignore
        )

        content_parts = []
        processed_files_relative_paths = []
        skipped_files = []

        for file_path in sorted(final_filtered_entries):
            relative_path = os.path.relpath(file_path, self.config.root_dir)
            file_type = detect_file_type(file_path)

            if file_type in ["image", "pdf"]:
                file_extension = os.path.splitext(file_path)[1].lower()
                file_name_without_extension = os.path.basename(file_path).replace(file_extension, "")
                requested_explicitly = any(
                    pattern.lower().endswith(file_extension)
                    or file_name_without_extension in pattern
                    for pattern in search_patterns
                )
                if not requested_explicitly:
                    skipped_files.append(
                        {
                            "path": relative_path,
                            "reason": "asset file (image/pdf) was not explicitly requested by name or extension",
                        }
                    )
                    continue

            result = process_single_file_content(file_path, self.config.root_dir)

            if result.get("error"):
                skipped_files.append({"path": relative_path, "reason": result["error"]})
            else:
                if isinstance(result["llmContent"], dict):  # Binary file part
                    content_parts.append(result["llmContent"])
                else:  # Text file
                    separator = DEFAULT_OUTPUT_SEPARATOR_FORMAT.replace("{filePath}", file_path)
                    content_parts.append(f'{separator}\n\n{result["llmContent"]}\n\n')
                processed_files_relative_paths.append(relative_path)

        # Build display message
        display_message = f"### ReadManyFiles Result (Target Dir: `{self.config.root_dir}`)\n\n"
        if processed_files_relative_paths:
            display_message += f"Successfully read and concatenated content from **{len(processed_files_relative_paths)} file(s)**.\n"
            if len(processed_files_relative_paths) <= 10:
                display_message += "\n**Processed Files:**\n"
                for p in processed_files_relative_paths:
                    display_message += f"- `{p}`\n"
            else:
                display_message += "\n**Processed Files (first 10 shown):**\n"
                for p in processed_files_relative_paths[:10]:
                    display_message += f"- `{p}`\n"
                display_message += f"- ...and {len(processed_files_relative_paths) - 10} more.\n"

        if skipped_files:
            if not processed_files_relative_paths:
                display_message += "No files were read and concatenated based on the criteria.\n"
            if len(skipped_files) <= 5:
                display_message += f"\n**Skipped {len(skipped_files)} item(s):**\n"
            else:
                display_message += f"\n**Skipped {len(skipped_files)} item(s) (first 5 shown):**\n"
            for f in skipped_files[:5]:
                display_message += f"- `{f['path']}` (Reason: {f['reason']})\n"
            if len(skipped_files) > 5:
                display_message += f"- ...and {len(skipped_files) - 5} more.\n"

        if not processed_files_relative_paths and not skipped_files:
            display_message += "No files were read and concatenated based on the criteria.\n"

        if not content_parts:
            content_parts.append("No files matching the criteria were found or all were skipped.")

        return ToolResult(llm_content=content_parts, return_display=display_message.strip())
