"""
Submission tool for commit-based evolution.

This tool triggers repository-level evaluation and returns metrics along with a
commit_message authored by the agent. The tool output is intended to be stored
in conversation history and consumed by the host to complete the iteration.
"""

from __future__ import annotations

import json
import os
import hashlib
from typing import Any, Dict, List

from .base import BaseTool, Icon, ToolLocation, ToolResult
from .config import Config as ToolConfig
from openevolve.utils.git_utils import diff_worktree
from openevolve.utils.diff_utils import clean_diff


class SubmitTool(BaseTool):
    """A repository-root based submission tool.

    Requires a human-written commit_message describing the changes.
    """

    def __init__(self, evaluator, config: ToolConfig):
        super().__init__(
            name="submit",
            display_name="Submit",
            description=(
                "Submit current work for evaluation and commit. "
                "Provide a concise, human-readable 'commit_message' describing the changes."
            ),
            icon=Icon.HAMMER,
            parameter_schema={
                "type": "object",
                "properties": {
                    "program_id": {
                        "type": "string",
                        "description": "Optional ID for logging.",
                    },
                    "commit_message": {
                        "type": "string",
                        "description": "Required. Describe the current commit's changes in one or a few sentences.",
                    },
                },
                "required": ["commit_message"],
            },
            is_output_markdown=False,
            can_update_output=False,
        )
        self.evaluator = evaluator
        self.config = config

    def validate_tool_params(self, params: Dict[str, Any]) -> str | None:
        msg = params.get("commit_message", "")
        if not isinstance(msg, str) or not msg.strip():
            return "commit_message is required and must be a non-empty string."
        return None

    def tool_locations(self, params: Dict[str, Any]) -> List[ToolLocation]:
        # Repository-level submission; does not target a specific file
        return []

    async def execute(self, params: Dict[str, Any]) -> ToolResult:
        error = self.validate_tool_params(params)
        if error:
            return ToolResult(llm_content=error, return_display=error)

        program_id = params.get("program_id", "")
        commit_message = params.get("commit_message", "").strip()

        # Pre-evaluation exact deduplication (skip high-cost evaluation for duplicates)
        try:
            root_dir = self.config.root_dir
            base_ref = self.config.other_config.get("base_ref", "HEAD")
            dedup_exact_enabled = bool(self.config.other_config.get("dedup_exact_enabled", True))
            existing_keys = set(self.config.other_config.get("dedup_keys", []))

            if dedup_exact_enabled:
                raw_diff = diff_worktree(root_dir, base_ref)
                prompt_diff, hash_diff = clean_diff(raw_diff)
                text = hash_diff or prompt_diff or ""
                if text:
                    key = hashlib.sha1(text.encode("utf-8")).hexdigest()
                    if key in existing_keys:
                        # Mark this iteration to be skipped by the host after tool loop
                        skip_marker = os.path.join(root_dir, ".openevolve_skip_commit")
                        try:
                            with open(skip_marker, "w") as f:
                                f.write("duplicate-pre-eval\n")
                        except Exception:
                            pass
                        payload = {
                            "metrics": {},
                            "commit_message": commit_message,
                            "skipped_eval": True,
                            "reason": "duplicate_pre_eval",
                        }
                        payload_str = json.dumps(payload)
                        return ToolResult(llm_content=payload_str, return_display=payload_str)
        except Exception:
            # Dedup is best-effort; continue to evaluation on failure
            pass

        # Evaluate using the repository root
        try:
            metrics = await self.evaluator.evaluate_repo(self.config.root_dir, program_id=program_id)
        except Exception as e:
            msg = f"Evaluation failed: {e}"
            return ToolResult(llm_content=msg, return_display=msg)

        payload = {"metrics": metrics, "commit_message": commit_message}
        payload_str = json.dumps(payload)
        return ToolResult(llm_content=payload_str, return_display=payload_str)


