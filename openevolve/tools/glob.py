"""
Glob tool: expand glob patterns to list matching files and/or directories under the project root.
"""

import glob as _glob
import os
from typing import Any, Dict, List, Set, Tuple

import pathspec

from .base import BaseTool, Icon, ToolResult
from .config import Config, DEFAULT_FILE_FILTERING_OPTIONS
from .file_service import FileService
from .read_many_files import DEFAULT_EXCLUDES


class GlobTool(BaseTool):
    """Tool to list filesystem entries matching glob patterns."""

    def __init__(self, config: Config):
        super().__init__(
            name="glob",
            display_name="Glob",
            description=(
                "Expands glob patterns to list matching files and/or directories under the project root. "
                "Supports include/exclude patterns and optional .gitignore filtering."
            ),
            icon=Icon.FILE_SEARCH,
            parameter_schema={
                "type": "object",
                "properties": {
                    "paths": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Glob patterns or relative paths from the project root (e.g., 'src/**/*.py').",
                    },
                    "include": {
                        "type": "array",
                        "items": {"type": "string"},
                        "default": [],
                        "description": "Additional include patterns (applied like 'paths').",
                    },
                    "exclude": {
                        "type": "array",
                        "items": {"type": "string"},
                        "default": [],
                        "description": "Exclude patterns (gitwildmatch).",
                    },
                    "useDefaultExcludes": {"type": "boolean", "default": True},
                    "file_filtering_options": {
                        "type": "object",
                        "properties": {"respect_git_ignore": {"type": "boolean"}},
                    },
                    "includeFiles": {"type": "boolean", "default": True},
                    "includeDirs": {"type": "boolean", "default": True},
                    "relative": {
                        "type": "boolean",
                        "default": True,
                        "description": "Return paths relative to the project root when true; absolute paths when false.",
                    },
                },
                "required": ["paths"],
            },
        )
        self.config = config
        self.file_service = FileService(config)

    def validate_tool_params(self, params: Dict[str, Any]) -> str | None:
        if not isinstance(params.get("paths"), list) or not all(
            isinstance(p, str) for p in params.get("paths", [])
        ):
            return "`paths` must be an array of strings."

        if params.get("includeFiles") is False and params.get("includeDirs") is False:
            return "At least one of `includeFiles` or `includeDirs` must be true."

        return None

    def _collect_matches(self, patterns: List[str]) -> Tuple[Set[str], Set[str]]:
        files: Set[str] = set()
        dirs: Set[str] = set()
        for pattern in patterns:
            full_pattern = os.path.join(self.config.root_dir, pattern)
            for path in _glob.glob(full_pattern, recursive=True):
                abs_path = os.path.abspath(path)
                if os.path.isdir(abs_path):
                    dirs.add(abs_path)
                elif os.path.isfile(abs_path):
                    files.add(abs_path)
        return files, dirs

    def _apply_excludes_and_ignores(
        self,
        files: Set[str],
        dirs: Set[str],
        exclude_patterns: List[str],
        respect_git_ignore: bool,
    ) -> Tuple[List[str], List[str]]:
        spec = pathspec.PathSpec.from_lines("gitwildmatch", exclude_patterns)

        # Filter files via helper to respect .gitignore
        filtered_files = [
            f
            for f in self.file_service.filter_files(list(files), respect_git_ignore, False)
            if not spec.match_file(os.path.relpath(f, self.config.root_dir))
        ]

        # Filter directories manually (by pattern and gitignore)
        filtered_dirs: List[str] = []
        for d in dirs:
            rel = os.path.relpath(d, self.config.root_dir)
            if spec.match_file(rel):
                continue
            if respect_git_ignore and self.file_service.gitignore_spec.match_file(rel):
                continue
            filtered_dirs.append(d)

        return filtered_files, filtered_dirs

    def get_description(self, params: Dict[str, Any]) -> str:
        patterns = params.get("paths", []) + params.get("include", [])
        return f"Glob match for {len(patterns)} pattern(s)"

    async def execute(self, params: Dict[str, Any]) -> ToolResult:
        validation_error = self.validate_tool_params(params)
        if validation_error:
            return ToolResult(llm_content=validation_error, return_display=validation_error)

        input_patterns = params.get("paths", [])
        include = params.get("include", [])
        exclude = params.get("exclude", [])
        use_default_excludes = params.get("useDefaultExcludes", True)
        include_files = params.get("includeFiles", True)
        include_dirs = params.get("includeDirs", True)
        return_relative = params.get("relative", True)

        ff_opts = params.get("file_filtering_options", {})
        respect_git_ignore = ff_opts.get(
            "respect_git_ignore", DEFAULT_FILE_FILTERING_OPTIONS.respect_git_ignore
        )

        effective_excludes = (DEFAULT_EXCLUDES + exclude) if use_default_excludes else exclude
        patterns = input_patterns + include
        if not patterns:
            return ToolResult(llm_content="[]", return_display="No patterns provided.")

        files, dirs = self._collect_matches(patterns)
        filtered_files, filtered_dirs = self._apply_excludes_and_ignores(
            files, dirs, effective_excludes, respect_git_ignore
        )

        result_paths: List[str] = []
        if include_files:
            result_paths.extend(filtered_files)
        if include_dirs:
            result_paths.extend(filtered_dirs)

        # Sort and map to desired path form
        result_paths_sorted = sorted(result_paths)
        if return_relative:
            result_paths_sorted = [
                os.path.relpath(p, self.config.root_dir) for p in result_paths_sorted
            ]

        # Build user-facing display
        display_lines: List[str] = []
        display_lines.append(
            f"### Glob Result (root: `{self.config.root_dir}`)\n\nFound {len(result_paths_sorted)} item(s)."
        )
        preview = result_paths_sorted[:20]
        if preview:
            display_lines.append("\n**First results:**")
            for p in preview:
                display_lines.append(f"- `{p}`")
            if len(result_paths_sorted) > len(preview):
                display_lines.append(f"- ...and {len(result_paths_sorted) - len(preview)} more.")

        llm_body = "\n".join(result_paths_sorted) if result_paths_sorted else ""
        return ToolResult(llm_content=llm_body, return_display="\n".join(display_lines).strip())


