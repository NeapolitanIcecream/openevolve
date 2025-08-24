"""
Base LLM interface
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, TYPE_CHECKING, Literal, TypedDict

if TYPE_CHECKING:
    # Imported only for type checking to avoid runtime cycles
    from openevolve.llm.session import ConversationSession
    from openevolve.config import PromptConfig


# --- Shared OpenAI-style chat and tool types ---

class FunctionCall(TypedDict):
    name: str
    arguments: str


class AssistantToolCall(TypedDict):
    id: str
    type: Literal["function"]
    function: FunctionCall


class ChatMessage(TypedDict, total=False):
    role: Literal["system", "user", "assistant", "tool"]
    content: Optional[str]
    name: str
    tool_call_id: str
    tool_calls: List[AssistantToolCall]
    created: int


class ToolFunctionDef(TypedDict, total=False):
    name: str
    description: Optional[str]
    parameters: Dict[str, object]


class ToolSpec(TypedDict):
    type: Literal["function"]
    function: ToolFunctionDef


class TokenUsage(TypedDict, total=False):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cached_tokens: int


class IterationRunResult(TypedDict, total=False):
    metrics: Dict[str, object]
    did_evaluate: bool
    commit_message: Optional[str]


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str


@dataclass
class LLMResult:
    content: Optional[str] = None
    tool_calls: List[ToolCall] = field(default_factory=list)
    json: Optional[object] = None
    raw: Optional[object] = None
    # Optional usage stats for token accounting
    usage: Optional[TokenUsage] = None


class LLMInterface(ABC):
    """Abstract base class for LLM interfaces with a unified invoke API"""

    @abstractmethod
    async def invoke(
        self,
        *,
        messages: List[ChatMessage],
        system_message: Optional[str] = None,
        response_format: Optional[Dict[str, object]] = None,
        tools: Optional[List[ToolSpec]] = None,
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
    async def get_history(self) -> List[ChatMessage]:
        """Get the conversation history (OpenAI Chat format messages)."""
        raise NotImplementedError

    @abstractmethod
    def attach_session(self, session: "ConversationSession") -> None:
        """Attach a conversation session to this client for history access."""
        raise NotImplementedError

    @abstractmethod
    def detach_session(self) -> None:
        """Detach any previously attached conversation session."""
        raise NotImplementedError

    @abstractmethod
    async def run_iteration_with_tools(
        self,
        iteration: int,
        parent_commit: str,
        iteration_context: str,
        *,
        prompt_cfg: Optional["PromptConfig"] = None,
        compression_client: Optional["LLMInterface"] = None,
        max_steps: int = 30,
    ) -> IterationRunResult:
        """High-level tool loop for one iteration."""
        raise NotImplementedError
