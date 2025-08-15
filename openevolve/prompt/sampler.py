"""
Prompt sampler for commit-based evolution (simplified, code-agnostic)
"""

import logging
from typing import Any, Dict, List, Optional, Union

from openevolve.config import PromptConfig

logger = logging.getLogger(__name__)
_PROMPT_SAMPLER_LOGGED = False


class PromptSampler:
    """构造基于提交的提示词（不依赖代码片段）。

    职责：
    - 会话级（稳定、KV-cache 友好）的 system message 构建
    - 迭代级（小而精的增量）iteration context 构建
    """

    def __init__(self, config: PromptConfig):
        self.config = config
        global _PROMPT_SAMPLER_LOGGED
        if not _PROMPT_SAMPLER_LOGGED:
            logger.info("Initialized commit-based prompt sampler")
            _PROMPT_SAMPLER_LOGGED = True

    def set_templates(self, system_template: Optional[str] = None, user_template: Optional[str] = None) -> None:
        # 保留接口以兼容外部调用，但 commit-based 版本不使用模板系统
        logger.info(f"(ignored) set_templates: system={system_template}, user={user_template}")

    def build_system_message(self) -> str:
        """返回本会话使用的系统提示（稳定、可缓存）。

        优先使用 `PromptConfig.system_message`，若为空或为占位值则回退到安全且工具导向的默认指令。
        """
        configured = (self.config.system_message or "").strip()
        if configured and configured != "system_message":
            return configured
        return (
            "You are an expert software agent operating inside a git worktree. "
            "Use the tools to read files, make minimal safe edits, and finally call 'evaluate' once to finish an iteration. "
            "Always use absolute paths under the provided root, avoid destructive changes, and keep edits consistent."
        )

    def build_iteration_context(
        self,
        evolution_target: Optional[str],
        parent_prompt_diff: Optional[str],
        inspiration_diffs: List[str],
        parent_metrics: Dict[str, Any],
        artifacts: Optional[Dict[str, Union[str, bytes]]] = None,
        max_inspirations: int = 2,
    ) -> str:
        """构造单次迭代的上下文文本（追加到会话历史）。

        仅包含必要增量：目标、父代与灵感的 diff 摘要、父代评估指标，以及可选的上次执行工件摘要。
        """
        parts: List[str] = []
        if evolution_target:
            parts.append(f"Goal: {evolution_target}")

        if parent_prompt_diff:
            parts.append("Parent changes (root→parent):\n" + parent_prompt_diff)

        if inspiration_diffs:
            use = inspiration_diffs[: max(0, max_inspirations)]
            for diff in use:
                if not diff:
                    continue
                parts.append("Inspiration (root→commit):\n" + diff)

        if parent_metrics:
            parts.append("Last evaluation (parent):\n" + self._format_metrics(parent_metrics))

        if self.config.include_artifacts and artifacts:
            rendered = self._render_artifacts(artifacts)
            if rendered:
                parts.append(rendered)

        return "\n\n".join(parts).strip()

    def build_commit_prompt(
        self,
        evolution_target: Optional[str],
        parent_prompt_diff: Optional[str],
        inspiration_diffs: List[str],
        parent_metrics: Dict[str, Any],
        artifacts: Optional[Dict[str, Union[str, bytes]]] = None,
        max_inspirations: int = 2,
    ) -> Dict[str, str]:
        parts: List[str] = []
        if evolution_target:
            parts.append(f"Goal:\n{evolution_target}\n")

        if parent_prompt_diff:
            parts.append("Parent changes (root→parent):\n```diff\n" + parent_prompt_diff + "\n```\n")

        if inspiration_diffs:
            use = inspiration_diffs[: max(0, max_inspirations)]
            for i, diff in enumerate(use, start=1):
                if not diff:
                    continue
                parts.append(f"Inspiration {i} (root→commit):\n```diff\n{diff}\n```\n")

        if parent_metrics:
            parts.append("Last evaluation (parent):\n" + self._format_metrics(parent_metrics) + "\n")

        if self.config.include_artifacts and artifacts:
            rendered = self._render_artifacts(artifacts)
            if rendered:
                parts.append(rendered)

        parts.append(
            "Instructions:\n"
            "- Use tools to read/modify files under the workspace root.\n"
            "- Make minimal, safe edits.\n"
            "- When the changes are ready, call 'evaluate' exactly once to finish the iteration.\n"
            "- Do NOT output code blocks; use tools instead.\n"
        )

        user_message = "\n\n".join(parts).strip()
        system_message = self.config.system_message or "You are an expert repo-level coding agent."
        return {"system": system_message, "user": user_message}

    def _format_metrics(self, metrics: Dict[str, Any]) -> str:
        lines: List[str] = []
        for k, v in metrics.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                try:
                    lines.append(f"- {k}: {v:.4f}")
                except Exception:
                    lines.append(f"- {k}: {v}")
            else:
                lines.append(f"- {k}: {v}")
        return "\n".join(lines)

    def _render_artifacts(self, artifacts: Dict[str, Union[str, bytes]]) -> str:
        if not artifacts:
            return ""
        sections: List[str] = []
        for key, value in artifacts.items():
            content = self._safe_decode_artifact(value)
            sections.append(f"### {key}\n```\n{content}\n```")
        if sections:
            return "## Last Execution Output\n\n" + "\n\n".join(sections)
        return ""

    def _safe_decode_artifact(self, value: Union[str, bytes]) -> str:
        if isinstance(value, str):
            return value
        if isinstance(value, bytes):
            try:
                return value.decode("utf-8", errors="replace")
            except Exception:
                return f"<binary data: {len(value)} bytes>"
        return str(value)

    
