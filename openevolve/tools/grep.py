"""
Grep tool: search for a regex pattern across files under the project root.
Respects include/exclude patterns and optionally .gitignore.

Optimized strategy:
- Prefer `git grep` when available (fast, respects repo state and can include
  untracked via flag). Falls back to system `grep` if `git` is unavailable.
- If neither is suitable (or for multiline searches), falls back to Python
  implementation.

All outputs are normalized to the same line-oriented format for LLM-friendly
consumption:
- Match line: "path:lineNumber:lineText"
- Context (before): "path:lineNumber-<lineText>"
- Context (after): "path:lineNumber+<lineText>"
"""

import glob
import os
import re
import shutil
import subprocess
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple
from typing import cast

import pathspec

from .base import BaseTool, Icon, ToolResult
from .config import Config, DEFAULT_FILE_FILTERING_OPTIONS
from .file_service import FileService
from .file_utils import DEFAULT_ENCODING, detect_file_type
from .read_many_files import DEFAULT_EXCLUDES


def _unique_preserve_order(items: Iterable[str]) -> List[str]:
    seen: Set[str] = set()
    result: List[str] = []
    for it in items:
        if it not in seen:
            seen.add(it)
            result.append(it)
    return result


class GrepTool(BaseTool):
    """Tool to search for a regex pattern across files."""

    def __init__(self, config: Config):
        super().__init__(
            name="grep",
            display_name="Grep",
            description=(
                "Search for a regular expression across files. Supports include/exclude patterns, "
                ".gitignore filtering, case-insensitive and multiline modes. Outputs either matching lines, "
                "files with matches, or counts per file."
            ),
            icon=Icon.REGEX,
            parameter_schema={
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Regular expression to search for."},
                    "paths": {
                        "type": "array",
                        "items": {"type": "string"},
                        "default": [],
                        "description": "Glob patterns or relative paths to limit search scope.",
                    },
                    "include": {"type": "array", "items": {"type": "string"}, "default": []},
                    "exclude": {"type": "array", "items": {"type": "string"}, "default": []},
                    "useDefaultExcludes": {"type": "boolean", "default": True},
                    "file_filtering_options": {
                        "type": "object",
                        "properties": {"respect_git_ignore": {"type": "boolean"}},
                    },
                    "caseInsensitive": {"type": "boolean", "default": False},
                    "multiline": {"type": "boolean", "default": False},
                    "output_mode": {
                        "type": "string",
                        "enum": ["content", "files_with_matches", "count"],
                        "default": "content",
                    },
                    "before": {"type": "integer", "default": 0, "description": "Lines of context before each match."},
                    "after": {"type": "integer", "default": 0, "description": "Lines of context after each match."},
                    "relative": {"type": "boolean", "default": True},
                },
                "required": ["pattern"],
            },
        )
        self.config = config
        self.file_service = FileService(config)

    def validate_tool_params(self, params: Dict[str, Any]) -> str | None:
        pattern = params.get("pattern")
        if not pattern or not isinstance(pattern, str):
            return "`pattern` must be a non-empty string."
        before = params.get("before", 0)
        after = params.get("after", 0)
        if before < 0 or after < 0:
            return "`before` and `after` must be non-negative integers."
        return None

    def _expand_files(self, patterns: List[str]) -> Set[str]:
        if not patterns:
            # default to everything under root
            patterns = ["**/*"]
        result: Set[str] = set()
        for pattern in patterns:
            full_pattern = os.path.join(self.config.root_dir, pattern)
            for p in glob.glob(full_pattern, recursive=True):
                if os.path.isfile(p):
                    result.add(os.path.abspath(p))
        return result

    def _filter_candidates(
        self,
        files: Set[str],
        exclude: List[str],
        respect_git_ignore: bool,
        use_default_excludes: bool,
    ) -> List[str]:
        patterns = (DEFAULT_EXCLUDES + exclude) if use_default_excludes else exclude
        spec = pathspec.PathSpec.from_lines("gitwildmatch", patterns)
        not_excluded = [
            f
            for f in files
            if not spec.match_file(os.path.relpath(f, self.config.root_dir))
        ]
        filtered = self.file_service.filter_files(not_excluded, respect_git_ignore)
        return sorted(filtered)

    def _compile_regex(self, pattern: str, case_insensitive: bool, multiline: bool) -> re.Pattern:
        flags = 0
        if case_insensitive:
            flags |= re.IGNORECASE
        if multiline:
            flags |= re.DOTALL
        return re.compile(pattern, flags)

    def _is_command_available(self, command: str) -> bool:
        return shutil.which(command) is not None

    def _chunk_list(self, items: List[str], chunk_size: int = 1000) -> List[List[str]]:
        if not items:
            return []
        return [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]

    def _run_git_grep(
        self,
        pattern: str,
        files: List[str],
        case_insensitive: bool,
    ) -> Tuple[bool, List[str]]:
        """Run git grep and return success flag and raw lines.

        - Returns (True, lines) if command executed (even if zero matches).
        - Returns (False, []) if command is unavailable or errored unexpectedly.
        """
        if not self._is_command_available("git"):
            return False, []

        # Convert absolute paths to repo-relative paths expected by git
        rel_files = [os.path.relpath(p, self.config.root_dir) for p in files]
        args_base = [
            "git",
            "grep",
            "-n",  # line numbers
            "-I",  # ignore binary
            "-H",  # always print filename
            "-E",  # extended regex
            "--no-color",
            "--untracked",  # include untracked files in working tree
        ]
        if case_insensitive:
            args_base.append("-i")

        # Some shells have argv limits; chunk file lists
        all_lines: List[str] = []
        if not rel_files:
            # No files to scan
            return True, []
        for chunk in self._chunk_list(rel_files, 1000):
            cmd = args_base + [pattern, "--"] + chunk
            try:
                proc = subprocess.run(
                    cmd,
                    cwd=self.config.root_dir,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
            except Exception:
                return False, []
            # git grep exit codes: 0=match found, 1=no matches, 2+=error
            if proc.returncode == 0:
                if proc.stdout:
                    all_lines.extend([ln for ln in proc.stdout.splitlines() if ln.strip()])
            elif proc.returncode == 1:
                # no matches in this chunk; continue
                continue
            else:
                # unexpected error; abort to fallback
                return False, []
        return True, all_lines

    def _run_system_grep(
        self,
        pattern: str,
        files: List[str],
        case_insensitive: bool,
    ) -> Tuple[bool, List[str]]:
        """Run system grep over provided files.

        - Returns (True, lines) if command executed (even if zero matches).
        - Returns (False, []) if command is unavailable or errored unexpectedly.
        """
        if not self._is_command_available("grep"):
            return False, []

        args_base = [
            "grep",
            "-n",  # line numbers
            "-H",  # always print filename
            "-I",  # ignore binary files
            "-s",  # suppress error messages
            "-E",  # extended regex
        ]
        if case_insensitive:
            args_base.append("-i")

        all_lines: List[str] = []
        if not files:
            return True, []
        for chunk in self._chunk_list(files, 1000):
            cmd = args_base + [pattern] + chunk
            try:
                proc = subprocess.run(
                    cmd,
                    cwd=self.config.root_dir,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
            except Exception:
                return False, []

            # grep exit codes: 0=match found, 1=no matches, 2=error
            if proc.returncode == 0:
                if proc.stdout:
                    all_lines.extend([ln for ln in proc.stdout.splitlines() if ln.strip()])
            elif proc.returncode == 1:
                continue
            else:
                return False, []

        return True, all_lines

    def _parse_grep_match_lines(self, raw_lines: List[str]) -> List[Tuple[str, int, str]]:
        """Parse 'path:line:content' output from grep-like tools.

        Lines that do not conform are skipped gracefully.
        """
        results: List[Tuple[str, int, str]] = []
        for ln in raw_lines:
            # Only trust lines that have at least two ':' separators
            # Split at the first two ':' to preserve colons in content
            first = ln.find(":")
            if first == -1:
                continue
            second = ln.find(":", first + 1)
            if second == -1:
                continue
            path = ln[:first]
            line_str = ln[first + 1 : second]
            text = ln[second + 1 :]
            try:
                line_no = int(line_str)
            except ValueError:
                continue
            results.append((path, line_no, text))
        return results

    def _build_context_lines(
        self,
        matches: List[Tuple[str, int, str]],
        before: int,
        after: int,
        return_relative: bool,
    ) -> List[str]:
        """Given match triples, expand with before/after contexts from files.

        Output lines follow the unified format described in module docstring.
        """
        # Group matches per file
        matches_by_file: Dict[str, List[Tuple[int, str]]] = {}
        for path, line_no, line_text in matches:
            matches_by_file.setdefault(path, []).append((line_no, line_text))

        all_lines: List[str] = []
        for path, entries in matches_by_file.items():
            abs_path = path
            if not os.path.isabs(abs_path):
                abs_path = os.path.join(self.config.root_dir, path)
            try:
                if detect_file_type(abs_path) != "text":
                    continue
                with open(abs_path, "r", encoding=DEFAULT_ENCODING) as f:
                    file_lines = f.readlines()
            except Exception:
                continue

            # Normalize display path
            path_display = (
                os.path.relpath(abs_path, self.config.root_dir)
                if return_relative
                else abs_path
            )

            # Build a map of which line numbers to output and their roles
            # Prefer 'match' over context when overlapping
            roles: Dict[int, str] = {}
            texts: Dict[int, str] = {}

            for line_no, match_text in entries:
                roles[line_no] = "match"
                texts[line_no] = match_text
                if before > 0:
                    start = max(1, line_no - before)
                    for ln in range(start, line_no):
                        if ln not in roles:
                            roles[ln] = "before"
                            # strip newline for context
                            if 1 <= ln <= len(file_lines):
                                texts[ln] = file_lines[ln - 1].rstrip("\n")
                if after > 0:
                    end = min(len(file_lines), line_no + after)
                    for ln in range(line_no + 1, end + 1):
                        if ln not in roles:
                            roles[ln] = "after"
                            if 1 <= ln <= len(file_lines):
                                texts[ln] = file_lines[ln - 1].rstrip("\n")

            # Emit lines in ascending order as they appear in file
            for ln in sorted(roles.keys()):
                txt = texts.get(ln, "")
                role = roles[ln]
                if role == "match":
                    all_lines.append(f"{path_display}:{ln}:{txt}")
                elif role == "before":
                    all_lines.append(f"{path_display}:{ln}-{txt} ")
                else:  # after
                    all_lines.append(f"{path_display}:{ln}+{txt} ")

        return all_lines

    def _scan_file_line_mode(
        self, file_path: str, regex: re.Pattern, before: int, after: int
    ) -> List[str]:
        try:
            if detect_file_type(file_path) != "text":
                return []
            with open(file_path, "r", encoding=DEFAULT_ENCODING) as f:
                lines = f.readlines()
        except Exception:
            return []

        matches: List[str] = []
        last_appended_end = -1
        context_after_remaining = 0
        for idx, line in enumerate(lines):
            if regex.search(line):
                # Append context before
                start_context = max(0, idx - before)
                if start_context <= last_appended_end:
                    start_context = last_appended_end + 1
                for j in range(start_context, idx):
                    matches.append(f"{file_path}:{j+1}-{lines[j].rstrip()} ")
                # Append the match line
                matches.append(f"{file_path}:{idx+1}:{line.rstrip()}")
                last_appended_end = idx
                context_after_remaining = after
            elif context_after_remaining > 0:
                matches.append(f"{file_path}:{idx+1}+{line.rstrip()} ")
                last_appended_end = idx
                context_after_remaining -= 1
        return matches

    def _scan_file_multiline(
        self, file_path: str, regex: re.Pattern, relative: bool
    ) -> Tuple[int, List[str]]:
        try:
            if detect_file_type(file_path) != "text":
                return 0, []
            with open(file_path, "r", encoding=DEFAULT_ENCODING) as f:
                content = f.read()
        except Exception:
            return 0, []

        results: List[str] = []
        count = 0
        for m in regex.finditer(content):
            count += 1
            # Compute line number by counting newlines up to match start
            start_pos = m.start()
            line_number = content.count("\n", 0, start_pos) + 1
            matched_line_start = content.rfind("\n", 0, start_pos) + 1
            matched_line_end = content.find("\n", m.end())
            if matched_line_end == -1:
                matched_line_end = len(content)
            line_text = content[matched_line_start:matched_line_end]
            path_display = file_path
            if relative:
                path_display = os.path.relpath(file_path, self.config.root_dir)
            results.append(f"{path_display}:{line_number}:{line_text}")
        return count, results

    async def execute(self, params: Dict[str, Any]) -> ToolResult:
        validation_error = self.validate_tool_params(params)
        if validation_error:
            return ToolResult(llm_content=validation_error, return_display=validation_error)

        pattern = cast(str, params.get("pattern"))
        input_patterns = params.get("paths", [])
        include = params.get("include", [])
        exclude = params.get("exclude", [])
        use_default_excludes = params.get("useDefaultExcludes", True)
        ff_opts = params.get("file_filtering_options", {})
        respect_git_ignore = ff_opts.get(
            "respect_git_ignore", DEFAULT_FILE_FILTERING_OPTIONS.respect_git_ignore
        )
        case_insensitive = params.get("caseInsensitive", False)
        multiline = params.get("multiline", False)
        output_mode = params.get("output_mode", "content")
        before = max(0, int(params.get("before", 0)))
        after = max(0, int(params.get("after", 0)))
        return_relative = params.get("relative", True)

        patterns = input_patterns + include
        candidate_files = self._expand_files(patterns)
        filtered_files = self._filter_candidates(
            candidate_files, exclude, respect_git_ignore, use_default_excludes
        )

        regex = self._compile_regex(pattern, case_insensitive, multiline)

        if output_mode == "files_with_matches":
            matched_files: List[str] = []
            if multiline:
                # Use Python multiline scanner
                for file_path in filtered_files:
                    count, _ = self._scan_file_multiline(file_path, regex, return_relative)
                    if count > 0:
                        matched_files.append(file_path)
            else:
                # Prefer external grep
                success, raw_lines = self._run_git_grep(pattern, filtered_files, case_insensitive)
                if not success:
                    success, raw_lines = self._run_system_grep(pattern, filtered_files, case_insensitive)
                if success:
                    triples = self._parse_grep_match_lines(raw_lines)
                    paths = [t[0] for t in triples]
                    # Normalize to absolute for dedupe, then conditionally make relative
                    abs_paths = [p if os.path.isabs(p) else os.path.join(self.config.root_dir, p) for p in paths]
                    matched_files = _unique_preserve_order(abs_paths)
                else:
                    # Fallback to Python line-mode scanner
                    for file_path in filtered_files:
                        lines = self._scan_file_line_mode(file_path, regex, 0, 0)
                        if lines:
                            matched_files.append(file_path)
            # Normalize display paths
            if return_relative:
                matched_files = [
                    os.path.relpath(p, self.config.root_dir) for p in matched_files
                ]
            else:
                matched_files = [
                    p if os.path.isabs(p) else os.path.join(self.config.root_dir, p)
                    for p in matched_files
                ]
            display = [
                f"### Grep Result (files)\n\nFound {len(matched_files)} file(s) with matches."
            ]
            for p in matched_files[:50]:
                display.append(f"- `{p}`")
            if len(matched_files) > 50:
                display.append(f"- ...and {len(matched_files) - 50} more.")
            llm_body = "\n".join(matched_files)
            return ToolResult(llm_content=llm_body, return_display="\n".join(display).strip())

        if output_mode == "count":
            count_lines: List[str] = []
            total = 0
            if multiline:
                for file_path in filtered_files:
                    cnt, _ = self._scan_file_multiline(file_path, regex, return_relative)
                    total += cnt
                    p = os.path.relpath(file_path, self.config.root_dir) if return_relative else file_path
                    count_lines.append(f"{p}:{cnt}")
            else:
                # Prefer external grep to collect counts efficiently
                success, raw_lines = self._run_git_grep(pattern, filtered_files, case_insensitive)
                if not success:
                    success, raw_lines = self._run_system_grep(pattern, filtered_files, case_insensitive)
                if success:
                    triples = self._parse_grep_match_lines(raw_lines)
                    # Initialize all files with zero counts to preserve previous behavior
                    per_file: Dict[str, int] = {
                        (fp if os.path.isabs(fp) else os.path.abspath(fp)): 0 for fp in filtered_files
                    }
                    for p, _ln, _txt in triples:
                        abs_p = p if os.path.isabs(p) else os.path.join(self.config.root_dir, p)
                        per_file[abs_p] = per_file.get(abs_p, 0) + 1
                    for abs_p in sorted(per_file.keys()):
                        cnt = per_file[abs_p]
                        total += cnt
                        disp = os.path.relpath(abs_p, self.config.root_dir) if return_relative else abs_p
                        count_lines.append(f"{disp}:{cnt}")
                else:
                    for file_path in filtered_files:
                        matches = self._scan_file_line_mode(file_path, regex, 0, 0)
                        file_count = len([m for m in matches if ":" in m and not m.endswith(" ")])
                        total += file_count
                        p = os.path.relpath(file_path, self.config.root_dir) if return_relative else file_path
                        count_lines.append(f"{p}:{file_count}")
            display = f"### Grep Count\n\nTotal matches: {total} across {len(filtered_files)} file(s)."
            return ToolResult(llm_content="\n".join(count_lines), return_display=display)

        # Default: output matching lines with context
        lines_out: List[str] = []

        if multiline:
            # Python fallback for multiline to ensure correctness
            for file_path in filtered_files:
                path_display = (
                    os.path.relpath(file_path, self.config.root_dir)
                    if return_relative
                    else file_path
                )
                _, matches = self._scan_file_multiline(file_path, regex, return_relative)
                if before == 0 and after == 0:
                    lines_out.extend(matches)
                else:
                    # Convert parsed line format back to triples for context expansion
                    triples: List[Tuple[str, int, str]] = []
                    for m in matches:
                        # m is already in path:line:text, possibly relative
                        first = m.find(":")
                        if first == -1:
                            continue
                        second = m.find(":", first + 1)
                        if second == -1:
                            continue
                        p = m[:first]
                        line_str = m[first + 1 : second]
                        text = m[second + 1 :]
                        try:
                            ln = int(line_str)
                        except ValueError:
                            continue
                        triples.append((p if not return_relative else os.path.join(self.config.root_dir, p), ln, text))
                    lines_out.extend(
                        self._build_context_lines(triples, before, after, return_relative)
                    )
        else:
            # Try external search first (git grep -> system grep), then fall back
            success, raw_lines = self._run_git_grep(pattern, filtered_files, case_insensitive)
            if not success:
                success, raw_lines = self._run_system_grep(pattern, filtered_files, case_insensitive)

            if success:
                triples = self._parse_grep_match_lines(raw_lines)
                if output_mode == "files_with_matches":
                    matched_files = _unique_preserve_order(
                        [t[0] for t in triples]
                    )
                    if return_relative:
                        matched_files = [
                            os.path.relpath(p if os.path.isabs(p) else os.path.join(self.config.root_dir, p), self.config.root_dir)
                            for p in matched_files
                        ]
                    display = [
                        f"### Grep Result (files)\n\nFound {len(matched_files)} file(s) with matches."
                    ]
                    for p in matched_files[:50]:
                        display.append(f"- `{p}`")
                    if len(matched_files) > 50:
                        display.append(f"- ...and {len(matched_files) - 50} more.")
                    llm_body = "\n".join(matched_files)
                    return ToolResult(llm_content=llm_body, return_display="\n".join(display).strip())

                if before == 0 and after == 0:
                    # Only emit matched lines, normalizing path display
                    for p, ln, txt in triples:
                        abs_p = p if os.path.isabs(p) else os.path.join(self.config.root_dir, p)
                        path_display = (
                            os.path.relpath(abs_p, self.config.root_dir)
                            if return_relative
                            else abs_p
                        )
                        lines_out.append(f"{path_display}:{ln}:{txt}")
                else:
                    lines_out.extend(
                        self._build_context_lines(triples, before, after, return_relative)
                    )
            else:
                # Fallback to Python line-mode scan
                for file_path in filtered_files:
                    path_display = (
                        os.path.relpath(file_path, self.config.root_dir)
                        if return_relative
                        else file_path
                    )
                    for line in self._scan_file_line_mode(file_path, regex, before, after):
                        if return_relative and line.startswith(file_path):
                            line = line.replace(file_path, path_display, 1)
                        lines_out.append(line)

        preview = lines_out[:200]
        display_lines = [
            f"### Grep Result (root: `{self.config.root_dir}`)\n\nFound {len(lines_out)} matching line(s) across {len(filtered_files)} file(s)."
        ]
        if preview:
            display_lines.append("\n**First results:**")
            for l in preview:
                display_lines.append(f"- `{l}`")
            if len(lines_out) > len(preview):
                display_lines.append(f"- ...and {len(lines_out) - len(preview)} more.")

        return ToolResult(llm_content="\n".join(lines_out), return_display="\n".join(display_lines).strip())


