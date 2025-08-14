"""
Base LLM interface
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str


@dataclass
class LLMResult:
    content: Optional[str] = None
    tool_calls: List[ToolCall] = field(default_factory=list)
    json: Optional[Any] = None
    raw: Optional[Any] = None


class LLMInterface(ABC):
    """Abstract base class for LLM interfaces with a unified invoke API"""

    @abstractmethod
    async def invoke(
        self,
        *,
        messages: List[Dict[str, Any]],
        system_message: Optional[str] = None,
        response_format: Optional[Dict[str, Any]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
        seed: Optional[int] = None,
        timeout: Optional[float] = None,
        retries: Optional[int] = None,
        retry_delay: Optional[float] = None,
    ) -> LLMResult:
        """Unified entry for OpenAI-style chat.completions.

        - messages: OpenAI Chat messages without the system message (unless caller wants it inside messages)
        - system_message: optional system message injected as the first message
        - response_format: optional OpenAI response_format (supports JSON Schema)
        - tools/tool_choice: optional tool call configuration
        - temperature/top_p/max_tokens/seed/timeout/retries/retry_delay: generation and runtime controls
        """
        raise NotImplementedError

    @abstractmethod
    async def get_history(self) -> List[Dict[str, Any]]:
        """Get the conversation history (OpenAI Chat format messages)."""
        raise NotImplementedError

    @abstractmethod
    def attach_session(self, session: Any) -> None:
        """Attach a conversation session to this client for history access."""
        raise NotImplementedError

    @abstractmethod
    def detach_session(self) -> None:
        """Detach any previously attached conversation session."""
        raise NotImplementedError
