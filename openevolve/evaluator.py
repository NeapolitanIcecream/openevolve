"""
Simplified repository-level evaluator for commit-based evolution.

It loads a user-provided evaluation script and executes its `evaluate` function
against the current worktree directory. The evaluation script is expected to
return a dictionary of metrics. Timeouts and retries are handled here.
"""

from __future__ import annotations

import asyncio
import importlib.util
import logging
import os
import sys
import time
import traceback
from typing import Any, Dict, Optional

from openevolve.config import EvaluatorConfig

logger = logging.getLogger(__name__)


class Evaluator:
    """Repo-level evaluator that calls a user-provided script on a worktree."""

    def __init__(self, config: EvaluatorConfig, evaluation_file: str, database: Optional[Any] = None):
        self.config = config
        self.evaluation_file = evaluation_file
        self.database = database
        self._load_evaluation_function()

    def _load_evaluation_function(self) -> None:
        if not os.path.exists(self.evaluation_file):
            raise ValueError(f"Evaluation file {self.evaluation_file} not found")

        try:
            eval_dir = os.path.dirname(os.path.abspath(self.evaluation_file))
            if eval_dir not in sys.path:
                sys.path.insert(0, eval_dir)
                logger.debug(f"Added {eval_dir} to Python path for evaluator imports")

            spec = importlib.util.spec_from_file_location("evaluation_module", self.evaluation_file)
            if spec is None or spec.loader is None:
                raise ImportError(f"Failed to load spec from {self.evaluation_file}")

            module = importlib.util.module_from_spec(spec)
            sys.modules["evaluation_module"] = module
            spec.loader.exec_module(module)

            # Prefer evaluate(repo_root), fallback to evaluate() with cwd set by caller
            if not hasattr(module, "evaluate"):
                raise AttributeError(
                    f"Evaluation file {self.evaluation_file} must define an 'evaluate' function"
                )

            self._evaluate_func = module.evaluate
            logger.info(f"Loaded evaluation function from {self.evaluation_file}")
        except Exception as e:
            logger.error(f"Error loading evaluation function: {e}")
            raise

    async def evaluate_repo(self, repo_root: str, program_id: str = "") -> Dict[str, Any]:
        """Run the evaluation function on the given repo root with timeout and retries."""
        last_exception: Optional[Exception] = None
        start_time = time.time()

        for attempt in range(self.config.max_retries + 1):
            try:
                async def _run():
                    loop = asyncio.get_event_loop()

                    def _call():
                        # Attempt call styles: evaluate(repo_root) -> dict, or evaluate() -> dict
                        try:
                            return self._evaluate_func(repo_root)
                        except TypeError:
                            # Maybe no parameters
                            return self._evaluate_func()

                    return await loop.run_in_executor(None, _call)

                result = await asyncio.wait_for(_run(), timeout=self.config.timeout)

                if not isinstance(result, dict):
                    logger.warning(
                        f"Evaluator returned non-dict result of type {type(result)}; coercing to empty dict"
                    )
                    result = {}

                elapsed = time.time() - start_time
                logger.info(
                    f"Evaluated repo at {repo_root} in {elapsed:.2f}s with metrics keys: {list(result.keys())}"
                )
                return result

            except asyncio.TimeoutError:
                logger.warning(
                    f"Evaluation timed out after {self.config.timeout}s (attempt {attempt + 1})"
                )
                return {"timeout": True}
            except Exception as e:
                last_exception = e
                logger.warning(
                    f"Evaluation attempt {attempt + 1}/{self.config.max_retries + 1} failed: {e}"
                )
                if attempt < self.config.max_retries:
                    await asyncio.sleep(1.0)

        logger.error(f"All evaluation attempts failed. Last error: {last_exception}")
        return {"error": str(last_exception) if last_exception else "unknown"}

