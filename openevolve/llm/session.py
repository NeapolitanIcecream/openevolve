"""
Unified conversation session for OpenAI-compatible chat history.
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional


def _now_ms() -> int:
    return int(time.time() * 1000)


class ConversationSession:
    """Centralized session that holds OpenAI-style chat messages.

    Messages do not include the system entry. The system message is stored separately
    and passed to the client on each API call.
    """

    def __init__(self, system_message: str) -> None:
        self.system_message: str = system_message
        self.messages: List[Dict[str, Any]] = []
        self.iteration_boundaries: List[int] = []

    def start_iteration(self, iteration: int, parent_commit: str, iteration_context: str) -> None:
        self.messages.append(
            {
                "role": "user",
                "content": f"Iteration {iteration}. Parent commit: {parent_commit}.\n" + iteration_context,
                "created": _now_ms(),
            }
        )

    def record_assistant_tool_calls(self, assistant_tool_calls: List[Dict[str, Any]]) -> None:
        self.messages.append(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": assistant_tool_calls,
                "created": _now_ms(),
            }
        )

    def append_tool_result(self, tool_call_id: str, name: str, content: str) -> None:
        self.messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "name": name,
                "content": content,
                "created": _now_ms(),
            }
        )

    def append_assistant_text(self, content: str) -> None:
        self.messages.append({"role": "assistant", "content": content, "created": _now_ms()})

    def mark_evaluated(self) -> None:
        self.iteration_boundaries.append(len(self.messages))

    def to_openai_messages(self) -> List[Dict[str, Any]]:
        return list(self.messages)

    def get_history(self) -> List[Dict[str, Any]]:
        return list(self.messages)

    async def compress_if_needed(self, prompt_cfg: Any, llm_client: Any) -> None:
        max_tokens = getattr(prompt_cfg, "session_max_tokens", 120000)
        compress_threshold = getattr(prompt_cfg, "session_compress_threshold", 80000)
        recent_limit = getattr(prompt_cfg, "recent_history_tokens", 30000)

        def _estimate_chars(msgs: List[Dict[str, Any]]) -> int:
            try:
                return sum(len(json.dumps(m, ensure_ascii=False)) for m in msgs)
            except Exception:
                return sum(len(str(m)) for m in msgs)

        total_chars = _estimate_chars(self.messages)
        if total_chars <= compress_threshold:
            return

        boundaries = self.iteration_boundaries[:]
        if not boundaries:
            return

        keep_start_index = boundaries[0]
        for i in range(len(boundaries) - 1, -1, -1):
            start = boundaries[i]
            tail = self.messages[start:]
            kept_chars = _estimate_chars(tail)
            if kept_chars >= recent_limit:
                keep_start_index = start
                break
            keep_start_index = start

        head = self.messages[:keep_start_index]

        summary_messages: List[Dict[str, Any]] = []
        summary_messages.append(
            {
                "role": "user",
                "content": (
                    "Please summarize the following prior iterations into a concise state snapshot, "
                    "preserving key decisions, file paths, and pending tasks. Return only the snapshot "
                    "wrapped in <state_snapshot>...</state_snapshot>.\n\n" + json.dumps(head, ensure_ascii=False)
                ),
            }
        )

        try:
            out = await llm_client.invoke(messages=summary_messages, system_message=None, max_tokens=1024)
            summary_text = out.content or ""
        except Exception:
            summary_text = "<state_snapshot>(unavailable)</state_snapshot>"

        if "<state_snapshot>" not in summary_text:
            summary_text = f"<state_snapshot>{summary_text}</state_snapshot>"

        tail = self.messages[keep_start_index:]
        self.messages.clear()
        self.messages.append({"role": "system", "content": self.system_message, "created": _now_ms()})
        self.messages.append({"role": "user", "content": summary_text, "created": _now_ms()})
        self.messages.append({"role": "assistant", "content": "Got it. Thanks for the additional context.", "created": _now_ms()})
        self.messages.extend(tail)


