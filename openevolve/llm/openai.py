"""
OpenAI API interface for LLMs
"""

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional, Union

import openai
import json

from openevolve.config import LLMModelConfig
from openevolve.llm.base import LLMInterface
from openevolve.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


class OpenAILLM(LLMInterface):
    """LLM interface using OpenAI-compatible APIs"""
    _initialized_models: set[str] = set()

    def __init__(
        self,
        model_cfg: Optional[LLMModelConfig] = None,
        tool_registry: Optional[ToolRegistry] = None,
    ):
        if not model_cfg:
            raise ValueError("LLMModelConfig is required")
        self.model = model_cfg.name
        self.system_message = model_cfg.system_message
        self.temperature = model_cfg.temperature
        self.top_p = model_cfg.top_p
        self.max_tokens = model_cfg.max_tokens
        self.timeout = model_cfg.timeout
        self.retries = model_cfg.retries
        self.retry_delay = model_cfg.retry_delay
        self.api_base = model_cfg.api_base
        self.api_key = model_cfg.api_key
        self.random_seed = getattr(model_cfg, "random_seed", None)
        self.tool_registry = tool_registry
        self.history: List[Dict[str, Any]] = []

        # Set up API client
        self.client = openai.OpenAI(
            api_key=self.api_key,
            base_url=self.api_base,
        )

        # Only log unique models to reduce duplication
        if self.model not in OpenAILLM._initialized_models:
            logger.info(f"Initialized OpenAI LLM with model: {self.model}")
            OpenAILLM._initialized_models.add(self.model)

    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from a prompt"""
        return await self.generate_with_context(
            system_message=self.system_message or "",
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )

    async def generate_with_context(
        self, system_message: str, messages: List[Dict[str, str]], **kwargs
    ) -> str:
        """Generate text using a system message and conversational context"""
        # Prepare messages with system message
        formatted_messages = [{"role": "system", "content": system_message}]
        formatted_messages.extend(messages)
        self.history.append({"role": "user", "parts": formatted_messages})

        # Set up generation parameters
        if self.api_base == "https://api.openai.com/v1" and str(self.model).lower().startswith("o"):
            # For o-series models
            params = {
                "model": self.model,
                "messages": formatted_messages,
                "max_completion_tokens": kwargs.get("max_tokens", self.max_tokens),
            }
        else:
            params = {
                "model": self.model,
                "messages": formatted_messages,
                "temperature": kwargs.get("temperature", self.temperature),
                "top_p": kwargs.get("top_p", self.top_p),
                "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            }

        if self.tool_registry:
            params["tools"] = self.tool_registry.get_tool_specs()
            params["tool_choice"] = "auto"
            
        # Add seed parameter for reproducibility if configured
        # Skip seed parameter for Google AI Studio endpoint as it doesn't support it
        seed = kwargs.get("seed", self.random_seed)
        if seed is not None:
            if self.api_base == "https://generativelanguage.googleapis.com/v1beta/openai/":
                logger.warning(
                    "Skipping seed parameter as Google AI Studio endpoint doesn't support it. "
                    "Reproducibility may be limited."
                )
            else:
                params["seed"] = seed

        # Attempt the API call with retries
        retries = kwargs.get("retries", self.retries)
        retry_delay = kwargs.get("retry_delay", self.retry_delay)
        timeout = kwargs.get("timeout", self.timeout)

        for attempt in range(retries + 1):
            try:
                response = await asyncio.wait_for(self._call_api(params), timeout=timeout)
                self.history.append({"role": "model", "parts": [{"text": response}]})
                return response
            except asyncio.TimeoutError:
                if attempt < retries:
                    logger.warning(f"Timeout on attempt {attempt + 1}/{retries + 1}. Retrying...")
                    await asyncio.sleep(retry_delay)
                else:
                    logger.error(f"All {retries + 1} attempts failed with timeout")
                    raise
            except Exception as e:
                if attempt < retries:
                    logger.warning(
                        f"Error on attempt {attempt + 1}/{retries + 1}: {str(e)}. Retrying..."
                    )
                    await asyncio.sleep(retry_delay)
                else:
                    logger.error(f"All {retries + 1} attempts failed with error: {str(e)}")
                    raise
        raise RuntimeError("All retry attempts failed.")

    async def _call_api(self, params: Dict[str, Any]) -> str:
        """Make the actual API call"""
        # Use asyncio to run the blocking API call in a thread pool
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None, lambda: self.client.chat.completions.create(**params)
        )
        # Logging of system prompt, user message and response content
        logger = logging.getLogger(__name__)
        logger.debug(f"API parameters: {params}")
        if response.choices[0].message.tool_calls:
            logger.debug(f"API response: {response.choices[0].message.tool_calls[0].function.arguments}")
            return response.choices[0].message.tool_calls[0].function.arguments
        logger.debug(f"API response: {response.choices[0].message.content}")
        return response.choices[0].message.content

    async def generate_json(
        self, prompt: str, json_schema: Dict[str, Any], **kwargs
    ) -> Dict[str, Any]:
        """Generate text from a prompt and parse it as JSON"""
        messages = [{"role": "user", "content": prompt}]
        params = {
            "model": self.model,
            "messages": messages,
            # Use the newer structured output format based on ResponseFormatJSONSchema
            # The OpenAI Python SDK expects the following structure:
            # {
            #     "type": "json_schema",
            #     "json_schema": {
            #         "name": "response",
            #         "description": "Schema for JSON response",
            #         "schema": { ... actual JSON schema ... },
            #         "strict": bool (optional)
            #     }
            # }
            # To maintain backward-compatibility with existing callers that pass in a bare
            # JSON Schema (i.e. without the wrapper fields), we automatically wrap the
            # provided schema if it doesn’t appear to already be in the expected
            # ResponseFormatJSONSchema format.
            "response_format": {
                "type": "json_schema",
                "json_schema": (
                    json_schema
                    if isinstance(json_schema, dict) and "schema" in json_schema
                    else {
                        "name": "response",
                        "schema": json_schema,
                    }
                ),
            },
            "temperature": kwargs.get("temperature", self.temperature),
            "top_p": kwargs.get("top_p", self.top_p),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
        }

        retries = kwargs.get("retries", self.retries)
        retry_delay = kwargs.get("retry_delay", self.retry_delay)
        timeout = kwargs.get("timeout", self.timeout)

        for attempt in range(retries + 1):
            try:
                response_str = await asyncio.wait_for(self._call_api(params), timeout=timeout)
                return json.loads(response_str)
            except (asyncio.TimeoutError, json.JSONDecodeError) as e:
                if attempt < retries:
                    logger.warning(
                        f"Error on attempt {attempt + 1}/{retries + 1}: {str(e)}. Retrying..."
                    )
                    await asyncio.sleep(retry_delay)
                else:
                    logger.error(
                        f"All {retries + 1} attempts failed with error: {str(e)}"
                    )
                    raise
        raise RuntimeError("All retry attempts failed.")

    async def get_history(self) -> List[Dict[str, Any]]:
        """Get the conversation history"""
        return self.history
