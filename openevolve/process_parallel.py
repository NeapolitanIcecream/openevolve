"""
Process-based parallel controller for true parallelism (commit-based evolution)

This controller allocates dedicated git worktrees to worker processes so that
each iteration can safely modify the repository state in parallel and create a
commit. The database derives diffs and MinHash signatures from commit hashes.
"""

import asyncio
import logging
import multiprocessing as mp
import os
import subprocess
import time
import hashlib
import uuid
from concurrent.futures import ProcessPoolExecutor, Future
from dataclasses import dataclass, asdict
from typing import Any, Callable, Dict, List, Optional, cast, TYPE_CHECKING

from openevolve.config import Config, LLMModelConfig
from openevolve.database import Program, ProgramDatabase
from openevolve.evaluator import Evaluator
from openevolve.llm.openai import OpenAILLM
from openevolve.llm.session import ConversationSession
from openevolve.llm.ensemble import LLMEnsemble
from openevolve.llm.base import IterationRunResult
from openevolve.prompt.sampler import PromptSampler
from openevolve.utils.git_utils import (
    configure_repo_defaults,
    create_worktree,
    ensure_clean_worktree,
    checkout_branch_at,
    create_commit_from_worktree,
)

logger = logging.getLogger(__name__)


# --- Worker-scoped globals (typed) ---
if TYPE_CHECKING:
    from openevolve.tools.registry import ToolRegistry
    from openevolve.llm.base import LLMInterface

_worker_config: Optional[Config] = None
_worker_evaluation_file: str = ""
_worker_evaluator: Optional[Evaluator] = None
_worker_llm_ensemble: Optional[LLMEnsemble] = None
_worker_prompt_sampler: Optional[PromptSampler] = None
_worker_registry: Optional["ToolRegistry"] = None
_worker_llm: Optional["LLMInterface"] = None
_worker_session: Optional[ConversationSession] = None


@dataclass
class SerializableResult:
    """Result that can be pickled and sent between processes"""
    child_program_dict: Optional[Dict[str, Any]] = None
    parent_id: Optional[str] = None
    iteration_time: float = 0.0
    iteration: int = 0
    error: Optional[str] = None


class WorktreePool:
    """Pre-create and manage a fixed number of git worktrees for parallel workers."""

    def __init__(
        self,
        repo_path: str,
        base_commit: str,
        pool_size: int,
        *,
        base_dir: Optional[str] = None,
        git_user_name: str = "OpenEvolve",
        git_user_email: str = "openevolve@example.com",
    ) -> None:
        self.repo_path = os.path.abspath(repo_path)
        self.base_commit = base_commit
        self.pool_size = max(1, pool_size)
        self.base_dir = base_dir or os.path.join(self.repo_path, ".openevolve", "worktrees")
        os.makedirs(self.base_dir, exist_ok=True)

        # Reduce lock contention and ensure commits work without relying on global config
        configure_repo_defaults(self.repo_path, git_user_name, git_user_email)

        self.slots: List[str] = []
        self._in_use: Dict[str, bool] = {}

        for i in range(self.pool_size):
            wt_dir = os.path.join(self.base_dir, f"wk_{i}")
            if not os.path.exists(os.path.join(wt_dir, ".git")):
                self._create_worktree(wt_dir)
            self.slots.append(wt_dir)
            self._in_use[wt_dir] = False

        logger.info(
            f"WorktreePool initialized with {len(self.slots)} worktrees under {self.base_dir}"
        )

    def _create_worktree(self, wt_dir: str) -> None:
        os.makedirs(os.path.dirname(wt_dir), exist_ok=True)
        create_worktree(self.repo_path, wt_dir, self.base_commit, detach=True)
        logger.debug(f"Created worktree {wt_dir} at {self.base_commit}")

    def acquire(self) -> Optional[str]:
        for wt in self.slots:
            if not self._in_use[wt]:
                self._in_use[wt] = True
                return wt
        return None

    def release(self, wt_dir: str) -> None:
        if wt_dir in self._in_use:
            self._in_use[wt_dir] = False
        else:
            logger.warning(f"Release called on unknown worktree {wt_dir}")

def _worker_init(config_dict: Dict[str, Any], evaluation_file: str) -> None:
    """Initialize worker process with necessary components (commit-based)."""
    global _worker_config
    global _worker_evaluation_file
    global _worker_evaluator
    global _worker_llm_ensemble
    global _worker_prompt_sampler
    global _worker_registry
    global _worker_llm
    global _worker_session
    
    # Store config for later use
    # Reconstruct Config object from nested dictionaries
    from openevolve.config import Config, DatabaseConfig, EvaluatorConfig, LLMConfig, PromptConfig, LLMModelConfig
    
    # Reconstruct model objects and defaults
    llm_block: Dict[str, Any] = cast(Dict[str, Any], config_dict['llm'])
    models_data: List[Dict[str, Any]] = cast(List[Dict[str, Any]], llm_block.get('models', []))
    evaluator_models_data: List[Dict[str, Any]] = cast(List[Dict[str, Any]], llm_block.get('evaluator_models', []))
    models = [LLMModelConfig(**m) for m in models_data]
    evaluator_models = [LLMModelConfig(**m) for m in evaluator_models_data]

    defaults_dict: Dict[str, Any] = cast(Dict[str, Any], llm_block.get('defaults', {}))
    defaults_cfg = LLMModelConfig(**defaults_dict) if isinstance(defaults_dict, dict) else LLMModelConfig()

    # Dedicated models (optional)
    write_tool_model_obj = None
    compression_model_obj = None
    if isinstance(llm_block.get('write_tool_model'), dict):
        write_tool_model_obj = LLMModelConfig(**cast(Dict[str, Any], llm_block.get('write_tool_model')))
    if isinstance(llm_block.get('compression_model'), dict):
        compression_model_obj = LLMModelConfig(**cast(Dict[str, Any], llm_block.get('compression_model')))

    llm_config = LLMConfig(
        defaults=defaults_cfg,
        models=models,
        evaluator_models=evaluator_models,
        write_tool_model=write_tool_model_obj,
        compression_model=compression_model_obj,
        tool_loop_max_steps=llm_block.get('tool_loop_max_steps', 30),
    )
    
    # Create other configs
    prompt_config = PromptConfig(**cast(Dict[str, Any], config_dict['prompt']))
    database_config = DatabaseConfig(**cast(Dict[str, Any], config_dict['database']))
    evaluator_config = EvaluatorConfig(**cast(Dict[str, Any], config_dict['evaluator']))
    
    _worker_config = Config(
        llm=llm_config,
        prompt=prompt_config,
        database=database_config,
        evaluator=evaluator_config,
        **{k: v for k, v in config_dict.items() if k not in ['llm', 'prompt', 'database', 'evaluator']}
    )
    _worker_evaluation_file = evaluation_file
    
    # These will be lazily initialized on first use
    _worker_evaluator = None
    _worker_llm_ensemble = None
    _worker_prompt_sampler = None
    _worker_registry = None
    _worker_llm = None

    # Unified session management (use PromptSampler to generate a stable system prompt)
    _worker_prompt_sampler = PromptSampler(prompt_config)
    _worker_session = ConversationSession(system_message=_worker_prompt_sampler.build_system_message())

    # Initialize repo-level evaluator for this worker
    try:
        _worker_evaluator = Evaluator(evaluator_config, evaluation_file)
    except Exception as e:
        logger.warning(f"Failed to initialize Evaluator in worker: {e}")


def _lazy_init_worker_components():
    """Lazily initialize expensive components on first use"""
    global _worker_evaluator
    global _worker_llm_ensemble
    global _worker_prompt_sampler
    global _worker_registry
    global _worker_llm
    global _worker_session
    global _worker_config
    
    assert _worker_config is not None, "Worker config not initialized"
    cfg: Config = _worker_config
    
    if _worker_llm_ensemble is None:
        _worker_llm_ensemble = LLMEnsemble(cfg.llm.models)

    if _worker_prompt_sampler is None:
        from openevolve.prompt.sampler import PromptSampler
        _worker_prompt_sampler = PromptSampler(cfg.prompt)

    if _worker_evaluator is None:
        from openevolve.evaluator import Evaluator
        _worker_evaluator = Evaluator(
            cfg.evaluator,
            _worker_evaluation_file,
            database=None,
        )

    # Singleton ToolRegistry and LLM (internal history disabled)
    if _worker_registry is None:
        from openevolve.tools.registry import ToolRegistry
        _worker_registry = ToolRegistry(config={"root_dir": cfg.database.git_repo_path}, evaluator=_worker_evaluator)

    if _worker_llm is None:
        # Decide the main working client: single model -> OpenAILLM; multiple -> Ensemble
        if len(cfg.llm.models) <= 1:
            _worker_llm = OpenAILLM(cfg.llm.models[0], tool_registry=_worker_registry)
        else:
            _worker_llm = LLMEnsemble(cfg.llm.models, tool_registry=_worker_registry)
        assert _worker_session is not None
        _worker_llm.attach_session(_worker_session)
        _worker_registry.set_llm_client(_worker_llm)

        # Prepare dedicated write tool client if configured
        write_cfg = None
        if getattr(cfg.llm, "write_tool_model", None) is not None:
            write_cfg = cfg.llm.write_tool_model
        # Build write LLM client
        if write_cfg is None:
            # Fallback: use the main working client (ensemble or single)
            write_llm_client = _worker_llm
        else:
            write_llm_client = OpenAILLM(write_cfg, tool_registry=_worker_registry)
            assert _worker_session is not None
            write_llm_client.attach_session(_worker_session)
        _worker_registry.set_write_llm_client(write_llm_client)


def _run_iteration_worker(
    iteration: int,
    db_snapshot: Dict[str, object],
    parent_id: str,
    inspiration_ids: List[str],
    worktree_dir: str,
    branch_name: str,
    parent_commit: str,
    iteration_context: str,
) -> SerializableResult:
    """Run a single iteration in a worker process (commit-based)."""
    try:
        # Lazy initialization
        _lazy_init_worker_components()

        # Reconstruct programs from snapshot
        prog_map = cast(Dict[str, Dict[str, Any]], db_snapshot.get("programs", {}))
        programs = {pid: Program.from_dict(prog_dict) for pid, prog_dict in prog_map.items()}
        
        parent = programs[parent_id]
        
        # Build exact-dedup keys from snapshot for CURRENT ISLAND only
        dedup_keys: set[str] = set()
        try:
            islands: List[List[str]] = cast(List[List[str]], db_snapshot.get("islands", []))
            current_island_idx: int = int(cast(int, db_snapshot.get("current_island", 0)))
            island_member_ids = set(islands[current_island_idx]) if 0 <= current_island_idx < len(islands) else set()
            for pid in island_member_ids:
                p = programs.get(pid)
                if not p:
                    continue
                text = (p.hash_diff or p.prompt_diff or "")
                if text:
                    try:
                        dedup_keys.add(hashlib.sha1(text.encode("utf-8")).hexdigest())
                    except Exception:
                        pass
        except Exception:
            # Fallback to empty set if snapshot malformed
            dedup_keys = set()

        # Start timer
        iteration_start = time.time()

        # Ensure clean working tree
        ensure_clean_worktree(worktree_dir)

        # Checkout unique branch at parent commit
        proc = checkout_branch_at(worktree_dir, branch_name, parent_commit, force=True)
        if proc.returncode != 0:
            return SerializableResult(error=f"git checkout failed: {proc.stderr}", iteration=iteration)

        # ---- LLM + Tools loop (multi-iteration session) ----
        # Bind this iteration's root directory to the registry (update root_dir)
        assert _worker_registry is not None, "ToolRegistry is not initialized"
        assert _worker_llm is not None, "LLM is not initialized"
        _worker_registry.tool_config.root_dir = worktree_dir
        _worker_registry.config["root_dir"] = worktree_dir
        # Inject pre-eval dedup context for Submit tool
        try:
            # Enable exact dedup by default; near-dup not used at pre-eval stage
            from openevolve.config import DatabaseConfig  # type: ignore
            assert _worker_config is not None
            dedup_exact_enabled = bool(getattr(_worker_config.database, 'dedup_exact_enabled', True))
        except Exception:
            dedup_exact_enabled = True
        # Use root_commit for dedup consistency with DB's diff baseline
        try:
            assert _worker_config is not None
            base_ref_for_dedup = _worker_config.database.root_commit
        except Exception:
            base_ref_for_dedup = parent_commit
        _worker_registry.tool_config.other_config["base_ref"] = base_ref_for_dedup
        _worker_registry.tool_config.other_config["dedup_exact_enabled"] = dedup_exact_enabled
        _worker_registry.tool_config.other_config["dedup_keys"] = list(dedup_keys)

        # Let the LLM layer run the full tool loop and maintain history
        # Build compression client if configured
        compression_client = None
        comp_cfg: Optional[LLMModelConfig] = None
        assert _worker_config is not None
        if getattr(_worker_config.llm, "compression_model", None) is not None:
            comp_cfg = _worker_config.llm.compression_model
        if comp_cfg is not None:
            compression_client = OpenAILLM(comp_cfg, tool_registry=None)

        run_out: IterationRunResult = asyncio.run(
            _worker_llm.run_iteration_with_tools(
                iteration=iteration,
                parent_commit=parent_commit,
                iteration_context=iteration_context,
                prompt_cfg=_worker_config.prompt,
                compression_client=compression_client,
                max_steps=getattr(_worker_config.llm, 'tool_loop_max_steps', 30),
            )
        )
        metrics: Dict[str, object] = cast(Dict[str, object], run_out.get("metrics") or {})
        did_evaluate: bool = bool(run_out.get("did_evaluate"))
        provided_commit_message: Optional[str] = run_out.get("commit_message")

        # Enforce 'submit' before committing only if required by the configuration
        require_eval = getattr(_worker_config.evaluator, 'require_evaluate_before_commit', True)
        if require_eval and not did_evaluate:
            return SerializableResult(error="Iteration ended without calling submit tool", iteration=iteration)

        # Always use the template; include agent-provided commit_message field
        max_metrics = getattr(_worker_config.database, 'commit_message_max_metrics', 6)
        metrics_list = [
            f"{k}={v:.4f}" if isinstance(v, (int, float)) else f"{k}={v}"
            for k, v in (metrics or {}).items()
        ][: max(0, max_metrics)]
        commit_metrics = " ".join(metrics_list)
        template = getattr(
            _worker_config.database,
            'commit_message_template',
            'OpenEvolve iteration {iteration} {commit_message} {metrics}'
        )
        commit_msg = template.format(
            iteration=iteration,
            metrics=commit_metrics,
            commit_message=(provided_commit_message or '').strip(),
        ).strip()
        # If a pre-eval dedup marker exists, skip committing
        skip_marker = os.path.join(worktree_dir, ".openevolve_skip_commit")
        if os.path.exists(skip_marker):
            return SerializableResult(error="Duplicate detected pre-eval; skipping commit", iteration=iteration)

        child_hash = create_commit_from_worktree(worktree_dir, commit_msg)

        # Create child program (DB will compute diffs/signatures)
        # Keep only numeric metrics for Program schema
        typed_metrics: Dict[str, float] = {
            k: float(v) for k, v in metrics.items() if isinstance(v, (int, float)) and not isinstance(v, bool)
        }

        child_program = Program(
            id=str(uuid.uuid4()),
            commit_hash=child_hash,
            parent_id=parent.id,
            generation=parent.generation + 1,
            metrics=typed_metrics,
            iteration_found=iteration,
            language=getattr(_worker_config, "language", "python") or "python",
            metadata={
                "branch": branch_name,
                "parent_commit": parent_commit,
            },
        )
        
        iteration_time = time.time() - iteration_start

        # Session compression is already executed on demand in the LLM layer
        
        return SerializableResult(
            child_program_dict=child_program.to_dict(),
            parent_id=parent.id,
            iteration_time=iteration_time,
            iteration=iteration,
        )
        
    except Exception as e:
        logger.exception(f"Error in worker iteration {iteration}")
        return SerializableResult(
            error=str(e),
            iteration=iteration
        )


class ProcessParallelController:
    """Controller for process-based parallel evolution"""
    
    def __init__(self, config: Config, evaluation_file: str, database: ProgramDatabase):
        self.config = config
        self.evaluation_file = evaluation_file
        self.database = database
        
        self.executor: Optional[ProcessPoolExecutor] = None
        self.shutdown_event = mp.Event()
        self.worktree_pool: Optional[WorktreePool] = None
        
        # Number of worker processes
        self.num_workers = config.evaluator.parallel_evaluations
        self._worktree_assignments: Dict[int, str] = {}
        
        logger.info(f"Initialized process parallel controller with {self.num_workers} workers")
    
    def _serialize_config(self, config: Config) -> Dict[str, object]:
        """Serialize config object to a dictionary that can be pickled"""
        # Manual serialization to handle nested objects properly
        def _maybe_asdict_model(m: Optional[LLMModelConfig]) -> Optional[Dict[str, object]]:
            if m is None:
                return None
            return asdict(m)  # dataclass -> Dict[str, object]

        return {
            'llm': {
                'defaults': asdict(config.llm.defaults),
                'models': [asdict(m) for m in config.llm.models],
                'evaluator_models': [asdict(m) for m in config.llm.evaluator_models],
                'write_tool_model': _maybe_asdict_model(getattr(config.llm, 'write_tool_model', None)),
                'compression_model': _maybe_asdict_model(getattr(config.llm, 'compression_model', None)),
                'tool_loop_max_steps': getattr(config.llm, 'tool_loop_max_steps', 30),
            },
            'prompt': asdict(config.prompt),
            'database': asdict(config.database),
            'evaluator': asdict(config.evaluator),
            'max_iterations': config.max_iterations,
            'checkpoint_interval': config.checkpoint_interval,
            'log_level': config.log_level,
            'log_dir': config.log_dir,
            'random_seed': config.random_seed,
            'language': config.language,
        }
    
    def start(self) -> None:
        """Start the process pool"""
        # Convert config to dict for pickling
        # We need to be careful with nested dataclasses
        config_dict = self._serialize_config(self.config)
        
        # Initialize a worktree pool sized to the number of workers
        self.worktree_pool = WorktreePool(
            repo_path=self.config.database.git_repo_path,
            base_commit=self.config.database.root_commit,
            pool_size=max(1, self.num_workers),
            base_dir=self.config.database.worktree_base_dir,
            git_user_name=self.config.database.git_user_name,
            git_user_email=self.config.database.git_user_email,
        )
        
        # Create process pool with initializer
        self.executor = ProcessPoolExecutor(
            max_workers=self.num_workers,
            initializer=_worker_init,
            initargs=(config_dict, self.evaluation_file)
        )
        
        logger.info(f"Started process pool with {self.num_workers} processes")
    
    def stop(self) -> None:
        """Stop the process pool"""
        self.shutdown_event.set()
        
        if self.executor:
            self.executor.shutdown(wait=True)
            self.executor = None
        
        logger.info("Stopped process pool")
    
    def request_shutdown(self) -> None:
        """Request graceful shutdown"""
        logger.info("Graceful shutdown requested...")
        self.shutdown_event.set()
    
    def _create_database_snapshot(self) -> Dict[str, object]:
        """Create a serializable snapshot of the database state"""
        # Only include necessary data for workers
        snapshot: Dict[str, object] = {
            "programs": {
                pid: prog.to_dict() 
                for pid, prog in self.database.programs.items()
            },
            "islands": [
                list(island) for island in self.database.islands
            ],
            "current_island": self.database.current_island,
        }
        
        return snapshot
    
    async def run_evolution(
        self,
        start_iteration: int,
        max_iterations: int,
        target_score: Optional[float] = None,
        checkpoint_callback: Optional[Callable[[int], None]] = None,
    ):
        """Run evolution with process-based parallelism"""
        if not self.executor:
            raise RuntimeError("Process pool not started")
        if self.worktree_pool is None:
            raise RuntimeError("Worktree pool not initialized")
        wt_pool = self.worktree_pool
        
        total_iterations = start_iteration + max_iterations
        
        logger.info(
            f"Starting process-based evolution from iteration {start_iteration} "
            f"for {max_iterations} iterations (total: {total_iterations})"
        )
        
        # Track pending futures
        pending_futures: Dict[int, Future[SerializableResult]] = {}
        deferred_iterations: List[int] = []  # iterations that failed to submit due to capacity and should be retried
        batch_size = min(self.num_workers * 2, max_iterations)
        
        # Submit initial batch
        for i in range(start_iteration, min(start_iteration + batch_size, total_iterations)):
            future = self._submit_iteration(i)
            if future:
                pending_futures[i] = future
            else:
                # Keep the iteration number for later retry when a worktree frees up
                deferred_iterations.append(i)
        
        next_iteration = start_iteration + batch_size
        completed_iterations = 0
        stop_requested = False
        
        # Island management
        programs_per_island = max(1, getattr(self.config.database, 'island_programs_per_switch', 10))
        current_island_counter = 0
        
        # Process results as they complete
        while (
            pending_futures 
            and completed_iterations < max_iterations
            and not self.shutdown_event.is_set()
        ):
            # Find completed futures
            completed_iteration = None
            for iteration, future in list(pending_futures.items()):
                if future.done():
                    completed_iteration = iteration
                    break
            
            if completed_iteration is None:
                await asyncio.sleep(0.01)
                continue
            
            # Process completed result
            future = pending_futures.pop(completed_iteration)
            
            try:
                result: SerializableResult = future.result()
                
                if result.error:
                    logger.warning(f"Iteration {completed_iteration} error: {result.error}")
                elif result.child_program_dict:
                    # Reconstruct program from dict
                    child_program = Program(**result.child_program_dict)
                    
                    # Add to database
                    self.database.add(child_program, iteration=completed_iteration)
                    
                    # Prompt logging not used in commit-based scaffold
                    
                    # Island management
                    if completed_iteration > start_iteration and current_island_counter >= programs_per_island:
                        self.database.next_island()
                        current_island_counter = 0
                        logger.debug(f"Switched to island {self.database.current_island}")
                    
                    current_island_counter += 1
                    self.database.increment_island_generation()
                    
                    # Check migration
                    if self.database.should_migrate():
                        logger.info(f"Performing migration at iteration {completed_iteration}")
                        self.database.migrate_programs()
                        self.database.log_island_status()
                    
                    # Log progress
                    logger.info(
                        f"Iteration {completed_iteration}: "
                        f"Program {child_program.id} "
                        f"(parent: {result.parent_id}) "
                        f"completed in {result.iteration_time:.2f}s"
                    )
                    
                    if child_program.metrics:
                        metrics_str = ", ".join([f"{k}={v:.4f}" for k, v in child_program.metrics.items()])
                        logger.info(f"Metrics: {metrics_str}")
                        
                        # Check if this is the first program without combined_score
                        if not hasattr(self, '_warned_about_combined_score'):
                            self._warned_about_combined_score = False
                        
                        if "combined_score" not in child_program.metrics and not self._warned_about_combined_score:
                            from openevolve.utils.metrics_utils import safe_numeric_average
                            avg_score = safe_numeric_average(child_program.metrics)
                            logger.warning(
                                f"⚠️  No 'combined_score' metric found in evaluation results. "
                                f"Using average of all numeric metrics ({avg_score:.4f}) for evolution guidance. "
                                f"For better evolution results, please modify your evaluator to return a 'combined_score' "
                                f"metric that properly weights different aspects of program performance."
                            )
                            self._warned_about_combined_score = True
                    
                    # Check for new best
                    if self.database.best_program_id == child_program.id:
                        logger.info(
                            f"🌟 New best solution found at iteration {completed_iteration}: "
                            f"{child_program.id}"
                        )
                    
                    # Checkpoint callback
                    # Don't checkpoint at iteration 0 (that's just the initial program)
                    if completed_iteration > 0 and completed_iteration % self.config.checkpoint_interval == 0:
                        logger.info(f"Checkpoint interval reached at iteration {completed_iteration}")
                        self.database.log_island_status()
                        if checkpoint_callback:
                            checkpoint_callback(completed_iteration)
                    
                    # Check target score
                    if target_score is not None and child_program.metrics:
                        numeric_metrics = list(child_program.metrics.values())
                        if numeric_metrics:
                            avg_score = sum(numeric_metrics) / len(numeric_metrics)
                            if avg_score >= target_score:
                                logger.info(
                                    f"Target score {target_score} reached at iteration {completed_iteration}"
                                )
                                # Defer loop break until after we release resources
                                stop_requested = True
                
            except Exception as e:
                logger.error(f"Error processing result from iteration {completed_iteration}: {e}")
            
            # Release any worktree assigned to the completed iteration ASAP to improve throughput
            if completed_iteration in self._worktree_assignments:
                wt_dir = self._worktree_assignments.pop(completed_iteration)
                wt_pool.release(wt_dir)

            completed_iterations += 1

            # If early stop requested, break after releasing resources
            if stop_requested:
                break

            # Try to submit deferred iterations first (retry those that previously failed due to capacity)
            while deferred_iterations and not self.shutdown_event.is_set():
                retry_it = deferred_iterations[0]
                fut_retry = self._submit_iteration(retry_it)
                if fut_retry:
                    pending_futures[retry_it] = fut_retry
                    deferred_iterations.pop(0)
                else:
                    # No capacity yet; try later
                    break

            # Submit next iteration (append to deferred if capacity unavailable)
            if next_iteration < total_iterations and not self.shutdown_event.is_set():
                fut_next = self._submit_iteration(next_iteration)
                if fut_next:
                    pending_futures[next_iteration] = fut_next
                else:
                    deferred_iterations.append(next_iteration)
                next_iteration += 1
        
        # Cancel any remaining evaluations (shutdown or early stop or natural end)
        if pending_futures:
            if self.shutdown_event.is_set():
                logger.info("Shutdown requested, canceling remaining evaluations...")
            else:
                logger.info("Canceling remaining evaluations...")
            for future in pending_futures.values():
                future.cancel()

        # Release any worktrees still assigned to in-flight or canceled iterations
        for it, wt_dir in list(self._worktree_assignments.items()):
            try:
                wt_pool.release(wt_dir)
            except Exception:
                pass
            finally:
                self._worktree_assignments.pop(it, None)

        logger.info("Evolution completed")
        
        return self.database.get_best_program()
    
    def _submit_iteration(self, iteration: int) -> Optional[Future[SerializableResult]]:
        """Submit an iteration to the process pool (commit-based)."""
        worktree_dir: Optional[str] = None
        try:
            # Avoid scheduling when executor not available or shutting down
            if self.executor is None or self.shutdown_event.is_set():
                return None
            # Ensure worktree pool is available
            wt_pool = self.worktree_pool
            if wt_pool is None:
                logging.warning("Worktree pool not initialized; cannot submit iteration")
                return None

            # Sample parent and inspirations
            parent, inspirations = self.database.sample()
            
            # Create database snapshot
            db_snapshot = self._create_database_snapshot()
            
            # Acquire a dedicated worktree for this iteration
            worktree_dir = wt_pool.acquire()
            if not worktree_dir:
                logger.warning("No available worktree to schedule iteration; delaying submission")
                return None
            branch_name = f"oe/it_{iteration}_{uuid.uuid4().hex[:8]}"
            parent_commit = parent.commit_hash
            
            # Submit to process pool
            future = self.executor.submit(
                _run_iteration_worker,
                iteration,
                db_snapshot,
                parent.id,
                [insp.id for insp in inspirations],
                worktree_dir,
                branch_name,
                parent_commit,
                self._build_iteration_context(parent, inspirations),
            )
            
            self._worktree_assignments[iteration] = worktree_dir
            
            return future
            
        except Exception as e:
            logger.error(f"Error submitting iteration {iteration}: {e}")
            # Ensure we release any acquired worktree on failure
            try:
                if worktree_dir and self.worktree_pool is not None:
                    self.worktree_pool.release(worktree_dir)
            except Exception:
                pass
            return None

    def _build_iteration_context(self, parent: Program, inspirations: List[Program]) -> str:
        """Build per-iteration context using PromptSampler."""
        sampler = PromptSampler(self.config.prompt)
        target = getattr(self.config.database, "evolution_target", None)
        parent_diff = getattr(parent, "prompt_diff", None)
        inspiration_diffs: List[str] = []
        for insp in inspirations:
            if getattr(insp, "prompt_diff", None):
                inspiration_diffs.append(insp.prompt_diff or "")
        return sampler.build_iteration_context(
            evolution_target=target,
            parent_prompt_diff=parent_diff,
            inspiration_diffs=inspiration_diffs,
            parent_metrics=parent.metrics or {},
            max_inspirations=getattr(self.config.prompt, 'max_inspirations', 2),
        )