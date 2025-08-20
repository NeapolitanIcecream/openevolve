"""
Single-file linter tool using ruff.

Constraints:
- Accept exactly one file via absolute path, must be within project root.
- No repository-wide scanning.
- Return minimal diagnostics to the model to save tokens (no stats/severity/linter fields).

Outputs:
- llm_content:
  - output_mode == "diagnostics" or "summary": newline-delimited diagnostics lines
    formatted as: "path:line:col:code:message".
  - output_mode == "json": JSON string { "diagnostics": [ { path, line, col, code, message } ] }.
- return_display: human-friendly markdown summary; may include counts, but
  those are not included in llm_content to reduce tokens sent to the model.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
from typing import Any, Dict, List, Tuple

from .base import BaseTool, Icon, ToolLocation, ToolResult
from .config import Config
from .file_utils import detect_file_type, is_within_root


class LintFileTool(BaseTool):
    """Run ruff against a single file and return cleaned diagnostics."""

    def __init__(self, config: Config):
        super().__init__(
            name="lint_file",
            display_name="LintFile",
            description=(
                "Run ruff lint on a single Python file. Returns cleaned diagnostics only "
                "(path, line, col, code, message) to minimize tokens."
            ),
            icon=Icon.HAMMER,
            parameter_schema={
                "type": "object",
                "properties": {
                    "absolute_path": {
                        "type": "string",
                        "description": "Absolute path to a Python file within project root.",
                    },
                    "output_mode": {
                        "type": "string",
                        "enum": ["summary", "diagnostics", "json"],
                        "default": "summary",
                    },
                    "relative": {"type": "boolean", "default": True},
                    "maxDiagnostics": {"type": "integer", "default": 500},
                    "timeoutSeconds": {"type": "integer", "default": 60},
                    "exitZero": {
                        "type": "boolean",
                        "default": True,
                        "description": "Treat non-zero ruff exit as analysis result, not execution failure.",
                    },
                },
                "required": ["absolute_path"],
            },
        )
        self.config = config

    def validate_tool_params(self, params: Dict[str, Any]) -> str | None:
        file_path = params.get("absolute_path")
        if not file_path or not isinstance(file_path, str):
            return "Missing or invalid `absolute_path` parameter."
        if not os.path.isabs(file_path):
            return f"File path must be absolute, but was relative: {file_path}."
        if not is_within_root(file_path, self.config.root_dir):
            return (
                f"File path must be within the root directory ({self.config.root_dir}): {file_path}"
            )
        if not os.path.exists(file_path) or not os.path.isfile(file_path):
            return f"File does not exist or is not a file: {file_path}"
        if detect_file_type(file_path) != "text":
            return "Only text files are supported."
        _, ext = os.path.splitext(file_path)
        if ext.lower() not in [".py", ".pyi"]:
            return "Only Python files (.py, .pyi) are supported by the linter."
        max_diags = int(params.get("maxDiagnostics", 500))
        if max_diags <= 0:
            return "`maxDiagnostics` must be a positive integer."
        timeout_s = int(params.get("timeoutSeconds", 60))
        if timeout_s <= 0:
            return "`timeoutSeconds` must be a positive integer."
        return None

    def get_description(self, params: Dict[str, Any]) -> str:
        try:
            rel = os.path.relpath(params.get("absolute_path", ""), self.config.root_dir)
        except Exception:
            rel = "(invalid path)"
        return rel

    def tool_locations(self, params: Dict[str, Any]) -> List[ToolLocation]:
        p = params.get("absolute_path")
        return [ToolLocation(path=p)] if p else []

    async def should_confirm_execute(self, params: Dict[str, Any]) -> Any:
        return False

    async def _run_ruff_with_timeout(
        self, cmd: List[str], timeout_seconds: int
    ) -> Tuple[int | None, str, str, bool]:
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=self.config.root_dir,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except Exception as e:
            return None, "", str(e), False

        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(), timeout=timeout_seconds
            )
        except asyncio.TimeoutError:
            try:
                proc.kill()
            except Exception:
                pass
            return None, "", "Timed out while running ruff.", True

        return proc.returncode, stdout_bytes.decode() if stdout_bytes else "", (
            stderr_bytes.decode() if stderr_bytes else ""
        ), False

    def _parse_ruff_json_stdout(self, stdout: str) -> List[Dict[str, Any]]:
        # ruff may output a JSON list or JSONL. Try list first, then line-by-line.
        stdout = stdout.strip()
        if not stdout:
            return []
        diagnostics: List[Dict[str, Any]] = []
        try:
            data = json.loads(stdout)
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        diagnostics.append(item)
                return diagnostics
        except json.JSONDecodeError:
            pass

        # Fallback: parse JSON per line (JSONL)
        for ln in stdout.splitlines():
            ln = ln.strip()
            if not ln:
                continue
            try:
                obj = json.loads(ln)
                if isinstance(obj, dict):
                    diagnostics.append(obj)
            except json.JSONDecodeError:
                continue
        return diagnostics

    def _normalize_diagnostics(
        self,
        raw: List[Dict[str, Any]],
        rel_path: str,
        return_relative: bool,
    ) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        for it in raw:
            try:
                filename = it.get("filename") or rel_path
                loc = it.get("location") or {}
                row = int(loc.get("row", 0))
                col = int(loc.get("column", 0))
                code = str(it.get("code") or "")
                message = str(it.get("message") or "")
                if not filename or row <= 0:
                    continue
                # ruff filename is usually relative to cwd; normalize to requested form
                abs_file = (
                    filename
                    if os.path.isabs(filename)
                    else os.path.join(self.config.root_dir, filename)
                )
                path_display = (
                    os.path.relpath(abs_file, self.config.root_dir)
                    if return_relative
                    else os.path.abspath(abs_file)
                )
                results.append(
                    {
                        "path": path_display,
                        "line": row,
                        "col": col,
                        "code": code,
                        "message": message,
                    }
                )
            except Exception:
                continue
        # Sort by (path, line, col, code) for stability
        results.sort(key=lambda d: (d.get("path", ""), d.get("line", 0), d.get("col", 0), d.get("code", "")))
        return results

    def _build_diagnostics_text(self, diags: List[Dict[str, Any]]) -> str:
        lines: List[str] = []
        for d in diags:
            path = d.get("path", "")
            line = d.get("line", 0)
            col = d.get("col", 0)
            code = d.get("code", "")
            message = d.get("message", "")
            lines.append(f"{path}:{line}:{col}:{code}:{message}")
        return "\n".join(lines)

    async def execute(self, params: Dict[str, Any]) -> ToolResult:
        validation_error = self.validate_tool_params(params)
        if validation_error:
            return ToolResult(llm_content=validation_error, return_display=validation_error)

        file_path: str = params["absolute_path"]
        return_relative = bool(params.get("relative", True))
        output_mode = params.get("output_mode", "summary")
        max_diagnostics = int(params.get("maxDiagnostics", 500))
        timeout_seconds = int(params.get("timeoutSeconds", 60))
        exit_zero = bool(params.get("exitZero", True))

        if shutil.which("ruff") is None:
            msg = (
                "Ruff is not installed or not found in PATH. Please install it (e.g., `uv add ruff` or `pip install ruff`)."
            )
            return ToolResult(llm_content="", return_display=msg)

        rel_path = os.path.relpath(file_path, self.config.root_dir)

        # Try modern ruff option first; fall back to legacy flag if needed.
        candidate_cmds: List[List[str]] = [
            ["ruff", "check", "--output-format", "json", "--quiet", rel_path],
            ["ruff", "check", "--format", "json", "--quiet", rel_path],
        ]

        raw_items: List[Dict[str, Any]] = []
        last_stderr = ""
        timed_out = False
        last_returncode: int | None = 0

        for cmd in candidate_cmds:
            returncode, stdout_str, stderr_str, did_timeout = await self._run_ruff_with_timeout(
                cmd, timeout_seconds
            )
            last_returncode = returncode
            last_stderr = stderr_str
            timed_out = did_timeout

            if did_timeout:
                # No need to try more variants if timed out
                break

            # Try to parse JSON regardless of exit code, unless stdout is empty
            items = self._parse_ruff_json_stdout(stdout_str)
            if items:
                raw_items = items
                break
            # If stdout empty, try next variant

        if timed_out:
            msg = f"Ruff lint timed out after {timeout_seconds}s for `{rel_path}`."
            return ToolResult(llm_content="", return_display=msg)

        # If still nothing and non-zero exit without output
        if not raw_items and (last_returncode is None or (last_returncode != 0 and not exit_zero)):
            stderr_preview = (last_stderr or "").strip()
            if len(stderr_preview) > 1000:
                stderr_preview = stderr_preview[:1000] + "..."
            msg = f"Failed to run ruff on `{rel_path}`. Exit code: {last_returncode}. Stderr: {stderr_preview}"
            return ToolResult(llm_content="", return_display=msg)

        # Normalize and trim diagnostics
        normalized = self._normalize_diagnostics(raw_items, rel_path, return_relative)
        if max_diagnostics and len(normalized) > max_diagnostics:
            normalized = normalized[:max_diagnostics]

        diagnostics_text = self._build_diagnostics_text(normalized)

        # Build display summary (for humans) without bloating llm_content
        if normalized:
            display_lines = [
                f"### LintFile Result (root: `{self.config.root_dir}`)\n",
                f"File: `{os.path.relpath(file_path, self.config.root_dir)}`",
                f"Issues: {len(normalized)}",
            ]
            # Show up to 10 examples inline
            shown = min(10, len(normalized))
            if shown:
                display_lines.append("\n**Examples:**")
                for d in normalized[:shown]:
                    display_lines.append(
                        f"- `{d['path']}:{d['line']}:{d['col']}:{d['code']}:{d['message']}`"
                    )
            if len(normalized) > shown:
                display_lines.append(f"- ...and {len(normalized) - shown} more.")
            return_display = "\n".join(display_lines).strip()
        else:
            return_display = (
                f"### LintFile Result (root: `{self.config.root_dir}`)\n\n"
                f"File: `{os.path.relpath(file_path, self.config.root_dir)}`\nNo issues found."
            )

        # Determine llm_content by output_mode, ensuring no stats are included
        if output_mode == "json":
            llm_content = json.dumps({"diagnostics": normalized})
        else:
            # For both summary and diagnostics, provide the minimal line form
            llm_content = diagnostics_text

        return ToolResult(llm_content=llm_content, return_display=return_display)


