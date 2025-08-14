"""
Model ensemble for LLMs
"""

import logging
import random
from typing import Dict, List, Optional, Any

from openevolve.llm.base import LLMInterface, LLMResult
from openevolve.llm.session import ConversationSession
from openevolve.llm.openai import OpenAILLM
from openevolve.config import LLMModelConfig
from openevolve.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


class LLMEnsemble(LLMInterface):
    """Ensemble of LLMs using unified invoke API"""
    _ensemble_logged: bool = False

    def __init__(self, models_cfg: List[LLMModelConfig], tool_registry: Optional[ToolRegistry] = None):
        self.models_cfg = models_cfg

        # Initialize models from the configuration
        self.models = [OpenAILLM(model_cfg, tool_registry=tool_registry) for model_cfg in models_cfg]

        # Extract and normalize model weights
        self.weights = [model.weight for model in models_cfg]
        total = sum(self.weights)
        self.weights = [w / total for w in self.weights]

        # Set up random state for deterministic model selection
        self.random_state = random.Random()
        # Initialize with seed from first model's config if available
        if (
            models_cfg
            and hasattr(models_cfg[0], "random_seed")
            and models_cfg[0].random_seed is not None
        ):
            self.random_state.seed(models_cfg[0].random_seed)
            logger.debug(
                f"LLMEnsemble: Set random seed to {models_cfg[0].random_seed} for deterministic model selection"
            )

        # Only log if we have multiple models or this is the first ensemble
        if len(models_cfg) > 1 or not LLMEnsemble._ensemble_logged:
            logger.info(
                f"Initialized LLM ensemble with models: "
                + ", ".join(
                    f"{model.name} (weight: {weight:.2f})"
                    for model, weight in zip(models_cfg, self.weights)
                )
            )
            LLMEnsemble._ensemble_logged = True

        # Shared session across ensemble members
        self._session: Optional[ConversationSession] = None

    def _sample_model(self) -> LLMInterface:
        index = self.random_state.choices(range(len(self.models)), weights=self.weights, k=1)[0]
        sampled_model = self.models[index]
        logger.info(f"Sampled model: {vars(sampled_model)['model']}")
        return sampled_model

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
        model = self._sample_model()
        return await model.invoke(
            messages=messages,
            system_message=system_message,
            response_format=response_format,
            tools=tools,
            tool_choice=tool_choice,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            seed=seed,
            timeout=timeout,
            retries=retries,
            retry_delay=retry_delay,
        )

    async def get_history(self) -> List[Dict[str, Any]]:
        return self._session.get_history() if self._session else []

    def attach_session(self, session: ConversationSession) -> None:
        self._session = session
        for m in self.models:
            try:
                m.attach_session(session)  # type: ignore[attr-defined]
            except Exception:
                pass

    def detach_session(self) -> None:
        for m in self.models:
            try:
                m.detach_session()  # type: ignore[attr-defined]
            except Exception:
                pass
        self._session = None
