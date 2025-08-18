"""
Evaluation tool for commit-based evolution.

This tool bridges the worker's Evaluator and the tool-calling loop.
It evaluates a single entry file by reading its content and invoking
the evaluator on the code string, returning a JSON string of metrics.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List

from .base import BaseTool, Icon, ToolLocation, ToolResult
from .config import Config as ToolConfig


class EvaluateTool(BaseTool):
    """A repository-root based evaluation tool.

    Notes: 'entry_file_path' is deprecated; evaluation always runs against the current
    working tree root. Optional parameter:
      - program_id (string, optional): Identifier for the evaluation run, used for logs/artifacts tracking.
    """

    def __init__(self, evaluator, config: ToolConfig):
        super().__init__(
            name="evaluate",
            display_name="Evaluate",
            description=(
                "Evaluate the current repository state. "
                "If 'entry_file_path' is provided, it will be read and passed to the evaluator; "
                "otherwise the evaluator will run in repo-level mode with the current working tree."
            ),
            icon=Icon.HAMMER,
            parameter_schema={
                "type": "object",
                "properties": {
                    "program_id": {
                        "type": "string",
                        "description": "Optional ID for logging and artifact tracking.",
                    },
                },
                "required": [],
            },
            is_output_markdown=False,
            can_update_output=False,
        )
        self.evaluator = evaluator
        self.config = config

    def validate_tool_params(self, params: Dict[str, Any]) -> str | None:
        # 'entry_file_path' is deprecated; keep empty validation
        return None

    def tool_locations(self, params: Dict[str, Any]) -> List[ToolLocation]:
        # Repository-level evaluation; does not target a specific file
        return []

    async def execute(self, params: Dict[str, Any]) -> ToolResult:
        error = self.validate_tool_params(params)
        if error:
            return ToolResult(llm_content=error, return_display=error)

        program_id = params.get("program_id", "")

        # Always evaluate using the repository root
        try:
            metrics = await self.evaluator.evaluate_repo(self.config.root_dir, program_id=program_id)
        except Exception as e:
            msg = f"Evaluation failed: {e}"
            return ToolResult(llm_content=msg, return_display=msg)

        # Return metrics as a JSON string for the model to parse
        metrics_json = json.dumps(metrics)
        return ToolResult(llm_content=metrics_json, return_display=metrics_json)


