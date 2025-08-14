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
    """一个基于仓库根目录的评估工具。

    说明：entry_file_path 已弃用，评估总是基于当前工作树根目录执行。
    支持可选参数：
      - program_id (string, optional): 评估运行的标识，用于日志/制品追踪。
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
        # entry_file_path 已弃用，保留空验证
        return None

    def tool_locations(self, params: Dict[str, Any]) -> List[ToolLocation]:
        # 仓库级评估，不定位到具体文件
        return []

    async def execute(self, params: Dict[str, Any]) -> ToolResult:
        error = self.validate_tool_params(params)
        if error:
            return ToolResult(llm_content=error, return_display=error)

        program_id = params.get("program_id", "")

        # 始终使用仓库根目录进行评估
        try:
            metrics = await self.evaluator.evaluate_repo(self.config.root_dir, program_id=program_id)
        except Exception as e:
            msg = f"Evaluation failed: {e}"
            return ToolResult(llm_content=msg, return_display=msg)

        # Return metrics as a JSON string for the model to parse
        metrics_json = json.dumps(metrics)
        return ToolResult(llm_content=metrics_json, return_display=metrics_json)


