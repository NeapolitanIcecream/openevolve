"""
OpenAI API interface for LLMs
"""

import asyncio
import logging
import json
from typing import Any, Dict, List, Optional, cast, TypeVar

import openai

from openevolve.config import LLMModelConfig, PromptConfig
from openevolve.llm.base import (
    LLMInterface,
    LLMResult,
    ToolCall,
    ChatMessage,
    ToolSpec,
    TokenUsage,
    IterationRunResult,
)
from openevolve.llm.session import ConversationSession
from openevolve.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


class OpenAILLM(LLMInterface):
    """LLM interface using OpenAI-compatible APIs with a unified invoke API"""
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
        # External session (single source of conversation history)
        self._session: Optional[ConversationSession] = None

        # Usage cumulative stats
        self._usage_cumulative: Dict[str, int] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "cached_tokens": 0,
            "calls": 0,
        }

        # Set up API client
        self.client = openai.OpenAI(
            api_key=self.api_key,
            base_url=self.api_base,
        )

        # Only log unique models to reduce duplication
        if isinstance(self.model, str) and self.model not in OpenAILLM._initialized_models:
            logger.info(f"Initialized OpenAI LLM with model: {self.model}")
            OpenAILLM._initialized_models.add(self.model)

    # --- Unified invoke API ---
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
        # Prepare messages
        formatted_messages: List[Dict[str, Any]] = []
        if system_message is None and self.system_message:
            formatted_messages.append({"role": "system", "content": self.system_message})
        elif system_message is not None:
            formatted_messages.append({"role": "system", "content": system_message})
        formatted_messages.extend(self._sanitize_messages(messages))

        # Build params
        params: Dict[str, Any] = {
            "model": self.model,
            "messages": formatted_messages,
            "temperature": self._fallback(temperature, self.temperature),
            "top_p": self._fallback(top_p, self.top_p),
        }
        if max_tokens is not None:
            params["max_tokens"] = max_tokens
        elif self.max_tokens is not None:
            params["max_tokens"] = self.max_tokens

        if response_format is not None:
            # Accept raw ResponseFormatJSONSchema or wrap bare schema
            if isinstance(response_format, dict) and "type" in response_format:
                params["response_format"] = response_format
            else:
                params["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "response",
                        "schema": response_format,
                    },
                }

        if tools is not None:
            params["tools"] = tools
            if tool_choice is not None:
                params["tool_choice"] = tool_choice

        use_seed = seed if seed is not None else self.random_seed
        if use_seed is not None:
            params["seed"] = use_seed

        # Attempt the API call with retries; if expecting JSON, treat JSON decode failure as retryable
        max_retries = self._fallback(retries, self.retries) or 0
        wait = self._fallback(retry_delay, self.retry_delay) or 0
        timeout_s = self._fallback(timeout, self.timeout)

        for attempt in range(max_retries + 1):
            try:
                response = await asyncio.wait_for(self._call_api(params), timeout=timeout_s)
                choice = response.choices[0].message
                content_text: Optional[str] = getattr(choice, "content", None)
                tool_calls_out: List[ToolCall] = []
                if getattr(choice, "tool_calls", None):
                    for tc in choice.tool_calls:
                        try:
                            tool_calls_out.append(
                                ToolCall(
                                    id=tc.id,
                                    name=tc.function.name,
                                    arguments=tc.function.arguments,
                                )
                            )
                        except Exception:
                            continue

                result = LLMResult(content=content_text, tool_calls=tool_calls_out, raw=response)

                # Extract usage and log token stats and cache percentage
                usage_info = self._extract_usage(response)
                if usage_info:
                    result.usage = usage_info
                    self._accumulate_usage(usage_info)
                    try:
                        prompt_tokens = usage_info.get("prompt_tokens", 0)
                        completion_tokens = usage_info.get("completion_tokens", 0)
                        total_tokens = usage_info.get("total_tokens", 0)
                        cached_tokens = usage_info.get("cached_tokens", 0)
                        percent_cached = (cached_tokens / prompt_tokens * 100.0) if prompt_tokens else 0.0
                        logger.info(
                            "Tokens used: prompt=%d, completion=%d, total=%d | cached=%d (%.1f%%)",
                            prompt_tokens,
                            completion_tokens,
                            total_tokens,
                            cached_tokens,
                            percent_cached,
                        )
                        # Also log cumulative usage briefly
                        cum = self.get_usage_stats()
                        logger.debug(
                            "Cumulative tokens: prompt=%d, completion=%d, total=%d | cached=%d (%.1f%%) in %d calls",
                            cum.get("prompt_tokens", 0),
                            cum.get("completion_tokens", 0),
                            cum.get("total_tokens", 0),
                            cum.get("cached_tokens", 0),
                            cum.get("percent_cached", 0.0),
                            cum.get("calls", 0),
                        )
                    except Exception:
                        # Avoid breaking flow on logging failure
                        pass

                # If JSON expected, parse strictly
                if params.get("response_format") is not None and not tool_calls_out:
                    if not content_text:
                        raise ValueError("Empty content for JSON response")
                    try:
                        result.json = json.loads(content_text)
                    except json.JSONDecodeError as e:
                        raise e

                logger.debug(f"API parameters: {self._redact(params)}")
                logger.debug(
                    f"API response: content_len={len(content_text or '')}, tool_calls={len(tool_calls_out)}"
                )
                return result
            except (asyncio.TimeoutError, json.JSONDecodeError) as e:
                if attempt < max_retries:
                    logger.warning(
                        f"OpenAI invoke attempt {attempt + 1}/{max_retries + 1} failed: {e}. Retrying..."
                    )
                    await asyncio.sleep(wait)
                else:
                    logger.error(
                        f"OpenAI invoke exhausted {max_retries + 1} attempts. Last error: {e}"
                    )
                    raise
            except Exception as e:
                if attempt < max_retries:
                    logger.warning(
                        f"OpenAI invoke attempt {attempt + 1}/{max_retries + 1} failed: {e}. Retrying..."
                    )
                    await asyncio.sleep(wait)
                else:
                    logger.error(
                        f"OpenAI invoke exhausted {max_retries + 1} attempts. Last error: {e}"
                    )
                    raise

        raise RuntimeError("All retry attempts failed")

    async def _call_api(self, params: Dict[str, Any]):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.client.chat.completions.create(**params))

    def _redact(self, params: Dict[str, Any]) -> Dict[str, Any]:
        # Avoid logging large messages and API key
        out = dict(params)
        out.pop("api_key", None)
        msgs = out.get("messages")
        if isinstance(msgs, list) and len(msgs) > 3:
            out["messages"] = msgs[:1] + ["...", msgs[-1]]
        return out

    T = TypeVar("T")

    def _fallback(self, v: Optional[T], default: Optional[T]) -> Optional[T]:
        return v if v is not None else default

    def _extract_usage(self, response: Any) -> TokenUsage:
        """Extract usage info from OpenAI response robustly.

        Returns keys: prompt_tokens, completion_tokens, total_tokens, cached_tokens
        """
        try:
            raw_usage: Any = getattr(response, "usage", None)
            if raw_usage is None and hasattr(response, "to_dict"):
                # Fallback for some client variants
                try:
                    raw_dict: Any = response.to_dict()
                    raw_usage = raw_dict.get("usage") if isinstance(raw_dict, dict) else None
                except Exception:
                    raw_usage = None
            if raw_usage is None and hasattr(response, "model_dump"):
                try:
                    raw_usage = response.model_dump().get("usage")
                except Exception:
                    raw_usage = None

            # Normalize attribute/dict style access
            def _get(obj: Any, key: str, default: Any = 0) -> Any:
                if obj is None:
                    return default
                if isinstance(obj, dict):
                    return obj.get(key, default)
                return getattr(obj, key, default)

            prompt_tokens = int(_get(raw_usage, "prompt_tokens", 0) or 0)
            completion_tokens = int(_get(raw_usage, "completion_tokens", 0) or 0)
            total_tokens = int(_get(raw_usage, "total_tokens", prompt_tokens + completion_tokens) or 0)

            # cached_tokens often lives in prompt_tokens_details.cached_tokens
            prompt_tokens_details: Any = _get(raw_usage, "prompt_tokens_details", None)
            cached_tokens = int(_get(prompt_tokens_details, "cached_tokens", 0) or 0)
        except Exception:
            # Provide at least zeroed structure to avoid KeyErrors upstream
            prompt_tokens = 0
            completion_tokens = 0
            total_tokens = 0
            cached_tokens = 0

        return cast(
            TokenUsage,
            {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "cached_tokens": cached_tokens,
            },
        )

    def _accumulate_usage(self, usage: TokenUsage) -> None:
        try:
            self._usage_cumulative["prompt_tokens"] += int(usage.get("prompt_tokens", 0) or 0)
            self._usage_cumulative["completion_tokens"] += int(usage.get("completion_tokens", 0) or 0)
            self._usage_cumulative["total_tokens"] += int(usage.get("total_tokens", 0) or 0)
            self._usage_cumulative["cached_tokens"] += int(usage.get("cached_tokens", 0) or 0)
            self._usage_cumulative["calls"] += 1
        except Exception:
            pass

    def get_usage_stats(self) -> Dict[str, Any]:
        try:
            prompt = self._usage_cumulative.get("prompt_tokens", 0) or 0
            cached = self._usage_cumulative.get("cached_tokens", 0) or 0
            percent_cached = (cached / prompt * 100.0) if prompt else 0.0
        except Exception:
            percent_cached = 0.0
        return {
            "prompt_tokens": int(self._usage_cumulative.get("prompt_tokens", 0) or 0),
            "completion_tokens": int(self._usage_cumulative.get("completion_tokens", 0) or 0),
            "total_tokens": int(self._usage_cumulative.get("total_tokens", 0) or 0),
            "cached_tokens": int(self._usage_cumulative.get("cached_tokens", 0) or 0),
            "calls": int(self._usage_cumulative.get("calls", 0) or 0),
            "percent_cached": percent_cached,
        }

    # --- Session attachment API ---
    def attach_session(self, session: ConversationSession) -> None:
        self._session = session

    def detach_session(self) -> None:
        self._session = None

    async def get_history(self) -> List[ChatMessage]:
        return self._session.get_history() if self._session else []

    # --- Helpers ---
    def _sanitize_messages(self, messages: List[ChatMessage]) -> List[Dict[str, Any]]:
        """Filter message fields to those accepted by OpenAI Chat Completions."""
        allowed_top = {"role", "content", "name", "tool_call_id", "tool_calls"}
        allowed_fn = {"name", "arguments"}
        sanitized: List[Dict[str, Any]] = []
        for m in messages:
            role = m.get("role")
            out: Dict[str, Any] = {k: v for k, v in m.items() if k in allowed_top}
            # Normalize assistant.tool_calls
            if role == "assistant" and isinstance(m.get("tool_calls"), list):
                tcs: List[Dict[str, Any]] = []
                for tc in m.get("tool_calls", []):
                    try:
                        fn = tc.get("function") or {}
                        tcs.append(
                            {
                                "id": tc.get("id"),
                                "type": "function",
                                "function": {k: fn.get(k) for k in allowed_fn if k in fn},
                            }
                        )
                    except Exception:
                        continue
                out["tool_calls"] = tcs
            sanitized.append(out)
        return sanitized

    # --- High-level tool loop API ---
    async def run_iteration_with_tools(
        self,
        iteration: int,
        parent_commit: str,
        iteration_context: str,
        *,
        prompt_cfg: Optional[PromptConfig] = None,
        compression_client: Optional[LLMInterface] = None,
        max_steps: int = 30,
    ) -> IterationRunResult:
        if not self._session:
            raise RuntimeError("No session attached to LLM client")
        if not self.tool_registry:
            raise RuntimeError("No ToolRegistry attached to LLM client")

        # Start iteration context as a user message
        self._session.start_iteration(iteration, parent_commit, iteration_context)

        metrics: Dict[str, object] = {}
        did_evaluate = False
        commit_message: Optional[str] = None

        for _ in range(max_steps):
            tools = cast(List[ToolSpec], self.tool_registry.get_tool_specs())
            out = await self.invoke(
                messages=self._session.messages,
                system_message=self._session.system_message,
                tools=tools,
                tool_choice="auto",
                max_tokens=self.max_tokens,
            )

            tool_calls = out.tool_calls
            content = out.content

            if tool_calls:
                # Record assistant tool calls (structured)
                assistant_tool_calls = []
                for tc in tool_calls:
                    assistant_tool_calls.append(
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                "arguments": tc.arguments or "{}",
                            },
                        }
                    )
                self._session.record_assistant_tool_calls(assistant_tool_calls)

                # Execute tool calls sequentially and append tool results
                for tc in tool_calls:
                    tool_name = tc.name or ""
                    raw_args = tc.arguments or "{}"
                    try:
                        params = json.loads(raw_args)
                    except Exception:
                        params = {}

                    tool = self.tool_registry.get_tool(tool_name)
                    if not tool:
                        self._session.append_tool_result(tc.id, tool_name, f"Error: Unknown tool '{tool_name}'")
                        continue

                    executed_ok = True
                    result = None
                    try:
                        result = await tool.execute(params)
                        tool_output_str = result.llm_content if hasattr(result, "llm_content") else "(no output)"
                    except Exception as e:
                        executed_ok = False
                        tool_output_str = f"Error executing tool '{tool_name}': {e}"

                    self._session.append_tool_result(tc.id, tool_name, tool_output_str)

                    if tool_name == "submit":
                        # Parse JSON from tool output: {"metrics": {...}, "commit_message": "..."}
                        try:
                            if executed_ok and isinstance(tool_output_str, str):
                                parsed = json.loads(tool_output_str)
                                if isinstance(parsed, dict):
                                    maybe_metrics = parsed.get("metrics")
                                    if isinstance(maybe_metrics, dict):
                                        metrics = maybe_metrics
                                    maybe_msg = parsed.get("commit_message")
                                    if isinstance(maybe_msg, str) and maybe_msg.strip():
                                        commit_message = maybe_msg.strip()
                        except Exception:
                            # Do not break flow on parse errors
                            pass

                        self._session.mark_evaluated()
                        did_evaluate = True
                        break

                if metrics:
                    break
                else:
                    continue

            if content:
                self._session.append_assistant_text(content)
                continue
            else:
                break

        # Optional compression after iteration completes
        try:
            if hasattr(self._session, "compress_if_needed"):
                await self._session.compress_if_needed(prompt_cfg, compression_client or self)
        except Exception as e:
            logger.debug(f"Session compression skipped or failed: {e}")

        return {"metrics": metrics, "did_evaluate": did_evaluate, "commit_message": commit_message}
