"""
Base LLM interface
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class LLMInterface(ABC):
    """Abstract base class for LLM interfaces"""

    @abstractmethod
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from a prompt"""
        pass

    @abstractmethod
    async def generate_with_context(
        self, system_message: str, messages: List[Dict[str, str]], **kwargs
    ) -> str:
        """Generate text using a system message and conversational context"""
        pass

    @abstractmethod
    async def generate_json(
        self, prompt: str, json_schema: Dict[str, Any], **kwargs
    ) -> Dict[str, Any]:
        """Generate text from a prompt and parse it as JSON"""
        pass

    @abstractmethod
    async def get_history(self) -> List[Dict[str, Any]]:
        """Get the conversation history"""
        pass
