"""
LS tool: list directory entries under the project root with filtering and formatting options.
"""

import os
from typing import Any, Dict, List, Optional

import pathspec

from .base import BaseTool, Icon, ToolLocation, ToolResult
from .config import Config, DEFAULT_FILE_FILTERING_OPTIONS
from .file_service import FileService
from .read_many_files import DEFAULT_EXCLUDES


class LSTool(BaseTool):
    """Tool to list directory entries."""

    def __init__(self, config: Config):
        super().__init__(
            name="ls",
            display_name="List",
            description=(
                "List files and directories at a given relative path under the project root. "
                "Supports recursion depth, include/exclude patterns, and .gitignore filtering."
            ),
            icon=Icon.FOLDER,
            parameter_schema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "default": ".",
                        "description": "Relative directory path from project root to list.",
                    },
                    "recursive": {"type": "boolean", "default": False},
                    "maxDepth": {"type": "integer", "default": 5},
                    "include": {"type": "array", "items": {"type": "string"}, "default": []},
                    "exclude": {"type": "array", "items": {"type": "string"}, "default": []},
                    "useDefaultExcludes": {"type": "boolean", "default": True},
                    "file_filtering_options": {
                        "type": "object",
                        "properties": {"respect_git_ignore": {"type": "boolean"}},
                    },
                    "includeFiles": {"type": "boolean", "default": True},
                    "includeDirs": {"type": "boolean", "default": True},
                    "relative": {"type": "boolean", "default": True},
                },
            },
        )
        self.config = config
        self.file_service = FileService(config)

    def validate_tool_params(self, params: Dict[str, Any]) -> str | None:
        if params.get("includeFiles") is False and params.get("includeDirs") is False:
            return "At least one of `includeFiles` or `includeDirs` must be true."
        return None

    def get_description(self, params: Dict[str, Any]) -> str:
        return f"List entries in `{params.get('path', '.')}`"

    def tool_locations(self, params: Dict[str, Any]) -> List[ToolLocation]:
        root_relative = params.get("path", ".")
        abs_path = os.path.join(self.config.root_dir, root_relative)
        return [ToolLocation(path=abs_path)]

    def _should_include(self, rel_path: str, spec: pathspec.PathSpec) -> bool:
        return not spec.match_file(rel_path)

    async def execute(self, params: Dict[str, Any]) -> ToolResult:
        root_relative = params.get("path", ".")
        recursive = params.get("recursive", False)
        max_depth = int(params.get("maxDepth", 5))
        include_patterns: List[str] = params.get("include", [])
        exclude_patterns: List[str] = params.get("exclude", [])
        use_default_excludes = params.get("useDefaultExcludes", True)
        include_files = params.get("includeFiles", True)
        include_dirs = params.get("includeDirs", True)
        return_relative = params.get("relative", True)

        ff_opts = params.get("file_filtering_options", {})
        respect_git_ignore = ff_opts.get(
            "respect_git_ignore", DEFAULT_FILE_FILTERING_OPTIONS.respect_git_ignore
        )

        abs_root = os.path.abspath(os.path.join(self.config.root_dir, root_relative))
        if not os.path.exists(abs_root):
            msg = f"Directory does not exist: {root_relative}"
            return ToolResult(llm_content=msg, return_display=msg)

        effective_excludes = (
            DEFAULT_EXCLUDES + exclude_patterns if use_default_excludes else exclude_patterns
        )
        exclude_spec = pathspec.PathSpec.from_lines("gitwildmatch", effective_excludes)

        results: List[str] = []

        def maybe_add(path: str):
            rel = os.path.relpath(path, self.config.root_dir)
            if not self._should_include(rel, exclude_spec):
                return
            if respect_git_ignore and self.file_service.gitignore_spec.match_file(rel):
                return
            results.append(path)

        if not recursive:
            for entry in sorted(os.listdir(abs_root)):
                p = os.path.join(abs_root, entry)
                if os.path.isdir(p) and include_dirs:
                    maybe_add(p)
                elif os.path.isfile(p) and include_files:
                    maybe_add(p)
        else:
            start_depth = abs_root.rstrip(os.sep).count(os.sep)
            for dirpath, dirnames, filenames in os.walk(abs_root):
                current_depth = dirpath.rstrip(os.sep).count(os.sep) - start_depth
                if current_depth > max_depth:
                    # Prune traversal
                    dirnames[:] = []
                    continue
                if include_dirs:
                    maybe_add(dirpath)
                if include_files:
                    for fn in filenames:
                        maybe_add(os.path.join(dirpath, fn))

        # Apply include patterns last (if provided)
        if include_patterns:
            include_spec = pathspec.PathSpec.from_lines("gitwildmatch", include_patterns)
            results = [
                p for p in results if include_spec.match_file(os.path.relpath(p, self.config.root_dir))
            ]

        # Convert to desired path form
        if return_relative:
            results = [os.path.relpath(p, self.config.root_dir) for p in results]

        results = sorted(set(results))

        display_lines: List[str] = []
        display_lines.append(
            f"### LS Result (root: `{self.config.root_dir}`)\n\nFound {len(results)} item(s)."
        )
        for p in results[:50]:
            display_lines.append(f"- `{p}`")
        if len(results) > 50:
            display_lines.append(f"- ...and {len(results) - 50} more.")

        return ToolResult(llm_content="\n".join(results), return_display="\n".join(display_lines).strip())


