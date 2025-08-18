"""
Prompt sampler for commit-based evolution (simplified, code-agnostic)
"""

import logging
from typing import Any, Dict, List, Optional, Union

from openevolve.config import PromptConfig

logger = logging.getLogger(__name__)
_PROMPT_SAMPLER_LOGGED = False


class PromptSampler:
    """Construct commit-based prompts (code-agnostic).

    Responsibilities:
    - Build session-level system message (stable, KV-cache friendly)
    - Build iteration-level context (small, precise increments)
    """

    def __init__(self, config: PromptConfig):
        self.config = config
        global _PROMPT_SAMPLER_LOGGED
        if not _PROMPT_SAMPLER_LOGGED:
            logger.info("Initialized commit-based prompt sampler")
            _PROMPT_SAMPLER_LOGGED = True

    def build_system_message(self) -> str:
        """Return the system prompt used for this session (stable, cacheable).

        Prefer `PromptConfig.system_message`; if empty or a placeholder, fall back to a safe,
        tool-oriented default instruction.
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
        """Construct per-iteration context text (appended to session history).

        Includes only necessary increments: target, diffs for parent and inspirations, parent metrics,
        and optional summary of last execution artifacts.
        """
        parts: List[str] = []
        if evolution_target:
            parts.append(f"Goal: {evolution_target}")

        if parent_prompt_diff:
            parts.append("Parent changes (root→parent):\n" + parent_prompt_diff)

        if parent_metrics:
            parts.append("Last evaluation (parent):\n" + self._format_metrics(parent_metrics))

        if inspiration_diffs:
            use = inspiration_diffs[: max(0, max_inspirations)]
            for diff in use:
                if not diff:
                    continue
                parts.append("Inspiration (root→commit):\n" + diff)

        if self.config.include_artifacts and artifacts:
            rendered = self._render_artifacts(artifacts)
            if rendered:
                parts.append(rendered)

        return "\n\n".join(parts).strip()

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

    
