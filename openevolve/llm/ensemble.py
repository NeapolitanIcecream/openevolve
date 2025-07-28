"""
Model ensemble for LLMs
"""

import asyncio
import logging
import random
from typing import Dict, List, Optional, Tuple, Any

from openevolve.llm.base import LLMInterface
from openevolve.llm.openai import OpenAILLM
from openevolve.config import LLMModelConfig
from openevolve.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


class LLMEnsemble(LLMInterface):
    """Ensemble of LLMs"""
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

        # Centralized history buffer to keep chronological order across models
        self._history: List[Dict[str, Any]] = []

    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate text using a randomly selected model based on weights"""
        model = self._sample_model()
        # Capture history before and after to maintain global order
        prev_len = len(await model.get_history())
        response = await model.generate(prompt, **kwargs)
        new_entries = (await model.get_history())[prev_len:]
        self._history.extend(new_entries)
        return response

    async def generate_with_context(
        self, system_message: str, messages: List[Dict[str, str]], **kwargs
    ) -> str:
        """Generate text using a system message and conversational context"""
        model = self._sample_model()
        prev_len = len(await model.get_history())
        response = await model.generate_with_context(system_message, messages, **kwargs)
        new_entries = (await model.get_history())[prev_len:]
        self._history.extend(new_entries)
        return response

    def _sample_model(self) -> LLMInterface:
        """Sample a model from the ensemble based on weights"""
        index = self.random_state.choices(range(len(self.models)), weights=self.weights, k=1)[0]
        sampled_model = self.models[index]
        logger.info(f"Sampled model: {vars(sampled_model)['model']}")
        return sampled_model

    async def generate_multiple(self, prompt: str, n: int, **kwargs) -> List[str]:
        """Generate multiple texts in parallel"""
        tasks = [self.generate(prompt, **kwargs) for _ in range(n)]
        return await asyncio.gather(*tasks)

    async def parallel_generate(self, prompts: List[str], **kwargs) -> List[str]:
        """Generate responses for multiple prompts in parallel"""
        tasks = [self.generate(prompt, **kwargs) for prompt in prompts]
        return await asyncio.gather(*tasks)

    async def generate_all_with_context(
        self, system_message: str, messages: List[Dict[str, str]], **kwargs
    ) -> List[str]:
        """Generate text using a all available models and average their returned metrics"""
        responses = []
        for model in self.models:
            responses.append(await model.generate_with_context(system_message, messages, **kwargs))
        return responses

    async def generate_json(
        self, prompt: str, json_schema: Dict[str, Any], **kwargs
    ) -> Dict[str, Any]:
        """Generate json using a randomly selected model based on weights"""
        model = self._sample_model()
        prev_len = len(await model.get_history())
        response = await model.generate_json(prompt, json_schema, **kwargs)
        new_entries = (await model.get_history())[prev_len:]
        self._history.extend(new_entries)
        return response

    async def get_history(self) -> List[Dict[str, Any]]:
        """Get the aggregated conversation history in chronological order"""
        return self._history
