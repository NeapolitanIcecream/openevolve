"""
Configuration handling for OpenEvolve
"""

import os
from dataclasses import dataclass, field, asdict, fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml


@dataclass
class LLMModelConfig:
    """Configuration for a single LLM model"""

    # API configuration
    api_base: Optional[str] = None
    api_key: Optional[str] = None
    name: Optional[str] = None

    # Weight for model in ensemble
    weight: float = 1.0

    # Generation parameters
    system_message: Optional[str] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None

    # Request parameters
    timeout: Optional[int] = None
    retries: Optional[int] = None
    retry_delay: Optional[int] = None

    # Reproducibility
    random_seed: Optional[int] = None


@dataclass
class LLMConfig:
    """Configuration for LLM ensemble (composition-based)."""

    # Shared defaults applied to each model (unless model overrides)
    defaults: LLMModelConfig = field(
        default_factory=lambda: LLMModelConfig(
            api_base="https://api.openai.com/v1",
            system_message="system_message",
            temperature=0.7,
            top_p=0.95,
            max_tokens=4096,
            timeout=60,
            retries=3,
            retry_delay=5,
        )
    )

    # Ensemble configurations
    models: List[LLMModelConfig] = field(
        default_factory=lambda: [
            LLMModelConfig(name="gpt-4o-mini", weight=0.8),
            LLMModelConfig(name="gpt-4o", weight=0.2),
        ]
    )
    # TODO: The evaluator_models parameter is used to support "LLM as an Evaluator."
    # This feature is planned for low-priority support.
    evaluator_models: List[LLMModelConfig] = field(default_factory=lambda: [])

    # Optional dedicated models
    write_tool_model: Optional[LLMModelConfig] = None
    compression_model: Optional[LLMModelConfig] = None

    # Tool loop control
    tool_loop_max_steps: int = 30

    def __post_init__(self):
        # If no evaluator models are defined, use the same models as for evolution
        if not self.evaluator_models or len(self.evaluator_models) < 1:
            self.evaluator_models = self.models.copy()

        # Propagate defaults to models when fields are None
        self._propagate_defaults()

    def _propagate_defaults(self) -> None:
        shared = {
            "api_base": self.defaults.api_base,
            "api_key": self.defaults.api_key,
            "system_message": self.defaults.system_message,
            "temperature": self.defaults.temperature,
            "top_p": self.defaults.top_p,
            "max_tokens": self.defaults.max_tokens,
            "timeout": self.defaults.timeout,
            "retries": self.defaults.retries,
            "retry_delay": self.defaults.retry_delay,
            "random_seed": self.defaults.random_seed,
        }
        self.update_model_params(shared, overwrite=False)

    def update_model_params(self, args: Dict[str, Any], overwrite: bool = False) -> None:
        """Update parameters for defaults and all models.

        - Defaults always updated (overwrite True assumed for defaults)
        - Individual models updated only when attribute is None unless overwrite=True
        """
        # Update defaults
        for key, value in args.items():
            if value is not None:
                setattr(self.defaults, key, value)
        # Update models
        for model in self.models + self.evaluator_models:
            for key, value in args.items():
                if value is None:
                    continue
                if overwrite or getattr(model, key, None) is None:
                    setattr(model, key, value)
        # Update dedicated models if present
        for maybe_model in [self.write_tool_model, self.compression_model]:
            if maybe_model is None:
                continue
            for key, value in args.items():
                if value is None:
                    continue
                if overwrite or getattr(maybe_model, key, None) is None:
                    setattr(maybe_model, key, value)

    def apply_env_defaults(self) -> None:
        """Apply environment variables (api_base, api_key) into defaults and models."""
        api_key = os.environ.get("OPENAI_API_KEY")
        api_base = os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1")
        self.update_model_params({"api_key": api_key, "api_base": api_base})


@dataclass
class PromptConfig:
    """Configuration for prompt generation"""

    # TODO: Support external prompt templates loaded from directory
    template_dir: Optional[str] = None
    system_message: str = "system_message"
    # TODO: System prompt for LLM-as-Evaluator / feedback (not yet wired)
    evaluator_system_message: str = "evaluator_system_message"

    # Template stochasticity
    # TODO: Template randomization not implemented yet
    use_template_stochasticity: bool = True
    template_variations: Dict[str, List[str]] = field(default_factory=dict)

    # Meta-prompting
    # TODO: Meta prompting is not implemented; kept for future experiments
    use_meta_prompting: bool = False
    meta_prompt_weight: float = 0.1

    # Inspirations in prompt context
    max_inspirations: int = 2

    # --- Long-session management (KV-cache friendly) ---
    session_max_tokens: int = 120000
    session_compress_threshold: int = 80000
    recent_history_tokens: int = 30000


@dataclass
class DatabaseConfig:
    """Configuration for the program database"""

    # General settings
    db_path: Optional[str] = None  # Path to store database on disk
    in_memory: bool = True


    # Evolutionary parameters
    population_size: int = 1000
    archive_size: int = 100
    num_islands: int = 5

    # Selection parameters
    elite_selection_ratio: float = 0.1
    exploration_ratio: float = 0.2
    exploitation_ratio: float = 0.7

    # Evolution target description & similarity control
    evolution_target: Optional[str] = None
    signature_similarity_threshold: float = 0.8

    # Git evolution settings
    # Root commit used as baseline for diff (e.g. initial commit SHA or main branch)
    root_commit: str = "HEAD"
    # Path to the git repository (defaults to current working directory)
    git_repo_path: str = "."
    # Worktree pool settings
    worktree_base_dir: Optional[str] = None  # Defaults to <repo>/.openevolve/worktrees
    git_user_name: str = "OpenEvolve"
    git_user_email: str = "openevolve@example.com"

    # Feature map dimensions for MAP-Elites
    # Default to complexity and diversity for better exploration
    feature_dimensions: List[str] = field(default_factory=lambda: ["complexity", "diversity"])
    feature_bins: Union[int, Dict[str, int]] = 10  # Can be int (all dims) or dict (per-dim)
    diversity_reference_size: int = 20  # Size of reference set for diversity calculation

    # Migration parameters for island-based evolution
    migration_interval: int = 50  # Migrate every N generations
    migration_rate: float = 0.1  # Fraction of population to migrate

    # Sampling/inspiration
    num_inspirations: int = 5

    # Island switching cadence
    island_programs_per_switch: int = 10

    # Diversity estimation
    island_diversity_sample_size: int = 5
    island_diversity_max_comparisons: int = 6
    diversity_cache_size: int = 1000
    feature_scaling_method: str = "minmax"
    # --- Feature statistics (sliding window & robust scaling) ---
    feature_stats_window_size: int = 5000
    feature_stats_recompute_interval: int = 500
    feature_stats_min_samples: int = 50
    feature_stats_robust_low_q: float = 0.05
    feature_stats_robust_high_q: float = 0.95
    feature_scaling_method_per_dim: Dict[str, str] = field(default_factory=dict)

    # MinHash signature settings
    minhash_num_perm: int = 64
    minhash_shingle_len: int = 5

    # Commit message
    commit_message_template: str = "OpenEvolve iteration {iteration} {commit_message} {metrics}"
    commit_message_max_metrics: int = 6

    # Random seed for reproducible sampling
    random_seed: Optional[int] = 42

    # --- Deduplication and migration diversity control ---
    # Exact deduplication (strong): treat programs with identical normalized diffs as duplicates
    dedup_exact_enabled: bool = True
    # Near-duplicate filtering for migration (MinHash-based). Keeps diversity high across islands
    dedup_near_enabled: bool = False
    # Similarity threshold for near-dup detection (1.0 == identical signatures)
    dedup_near_similarity_threshold: float = 0.98
    # Size of target island reference set for migration-time near-dup filtering
    migration_diversity_topk: int = 20


@dataclass
class EvaluatorConfig:
    """Configuration for program evaluation"""

    # General settings
    timeout: int = 300  # Maximum evaluation time in seconds
    max_retries: int = 3

    # Resource limits for evaluation
    memory_limit_mb: Optional[int] = None
    cpu_limit: Optional[float] = None

    # Evaluation strategies
    cascade_evaluation: bool = True
    cascade_thresholds: List[float] = field(default_factory=lambda: [0.5, 0.75, 0.9])

    # Parallel evaluation
    parallel_evaluations: int = 1
    distributed: bool = False

    # LLM-based feedback
    use_llm_feedback: bool = False
    llm_feedback_weight: float = 0.1

    # Commit gating
    require_evaluate_before_commit: bool = True


@dataclass
class Config:
    """Master configuration for OpenEvolve"""

    # General settings
    max_iterations: int = 10000
    checkpoint_interval: int = 100
    log_level: str = "INFO"
    log_dir: Optional[str] = None
    random_seed: Optional[int] = 42
    language: Optional[str] = None

    # Component configurations
    llm: LLMConfig = field(default_factory=LLMConfig)
    prompt: PromptConfig = field(default_factory=PromptConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    evaluator: EvaluatorConfig = field(default_factory=EvaluatorConfig)

    # Evolution settings (legacy removed in commit-based mode)

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "Config":
        """Load configuration from a YAML file"""
        with open(path, "r") as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "Config":
        """Create configuration from a dictionary"""
        # Handle nested configurations
        config = Config()

        # Update top-level fields
        for key, value in config_dict.items():
            if key not in ["llm", "prompt", "database", "evaluator"] and hasattr(config, key):
                setattr(config, key, value)

        # Update nested configs
        if "llm" in config_dict:
            llm_dict = config_dict["llm"]
            # Build defaults
            defaults_cfg = llm_dict.get("defaults", {})
            defaults = LLMModelConfig(**defaults_cfg) if isinstance(defaults_cfg, dict) else LLMModelConfig()
            # Build model lists
            models = [LLMModelConfig(**m) for m in llm_dict.get("models", [])]
            evaluator_models = [LLMModelConfig(**m) for m in llm_dict.get("evaluator_models", [])]
            write_tool_model_obj = None
            compression_model_obj = None
            wtm_raw = llm_dict.get("write_tool_model")
            if isinstance(wtm_raw, dict):
                write_tool_model_obj = LLMModelConfig(**wtm_raw)
            cpm_raw = llm_dict.get("compression_model")
            if isinstance(cpm_raw, dict):
                compression_model_obj = LLMModelConfig(**cpm_raw)

            config.llm = LLMConfig(
                defaults=defaults,
                models=models or Config().llm.models,
                evaluator_models=evaluator_models,
                write_tool_model=write_tool_model_obj,
                compression_model=compression_model_obj,
                tool_loop_max_steps=llm_dict.get("tool_loop_max_steps", 30),
            )
        if "prompt" in config_dict:
            config.prompt = PromptConfig(**config_dict["prompt"])
        if "database" in config_dict:
            config.database = DatabaseConfig(**config_dict["database"])

        # Ensure database inherits the random seed if not explicitly set
        if config.database.random_seed is None and config.random_seed is not None:
            config.database.random_seed = config.random_seed
        if "evaluator" in config_dict:
            config.evaluator = EvaluatorConfig(**config_dict["evaluator"])

        return config

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to a dictionary"""
        return {
            # General settings
            "max_iterations": self.max_iterations,
            "checkpoint_interval": self.checkpoint_interval,
            "log_level": self.log_level,
            "log_dir": self.log_dir,
            "random_seed": self.random_seed,
            # Component configurations
            "llm": {
                "defaults": {
                    "api_base": self.llm.defaults.api_base,
                    "api_key": self.llm.defaults.api_key,
                    "system_message": self.llm.defaults.system_message,
                    "temperature": self.llm.defaults.temperature,
                    "top_p": self.llm.defaults.top_p,
                    "max_tokens": self.llm.defaults.max_tokens,
                    "timeout": self.llm.defaults.timeout,
                    "retries": self.llm.defaults.retries,
                    "retry_delay": self.llm.defaults.retry_delay,
                    "random_seed": self.llm.defaults.random_seed,
                },
                "models": [asdict(m) for m in self.llm.models],
                "evaluator_models": [asdict(m) for m in self.llm.evaluator_models],
                "write_tool_model": asdict(self.llm.write_tool_model) if self.llm.write_tool_model else None,
                "compression_model": asdict(self.llm.compression_model) if self.llm.compression_model else None,
                "tool_loop_max_steps": self.llm.tool_loop_max_steps,
            },
            "prompt": {
                "template_dir": self.prompt.template_dir,
                "system_message": self.prompt.system_message,
                "evaluator_system_message": self.prompt.evaluator_system_message,
                "use_template_stochasticity": self.prompt.use_template_stochasticity,
                "template_variations": self.prompt.template_variations,
                "session_max_tokens": self.prompt.session_max_tokens,
                "session_compress_threshold": self.prompt.session_compress_threshold,
                "recent_history_tokens": self.prompt.recent_history_tokens,
                "max_inspirations": self.prompt.max_inspirations,
                # Note: meta-prompting features not implemented
                # "use_meta_prompting": self.prompt.use_meta_prompting,
                # "meta_prompt_weight": self.prompt.meta_prompt_weight,
            },
            "database": {
                "db_path": self.database.db_path,
                "in_memory": self.database.in_memory,
                "population_size": self.database.population_size,
                "archive_size": self.database.archive_size,
                "num_islands": self.database.num_islands,
                "elite_selection_ratio": self.database.elite_selection_ratio,
                "exploration_ratio": self.database.exploration_ratio,
                "exploitation_ratio": self.database.exploitation_ratio,
                "root_commit": self.database.root_commit,
                "git_repo_path": self.database.git_repo_path,
                "worktree_base_dir": self.database.worktree_base_dir,
                "git_user_name": self.database.git_user_name,
                "git_user_email": self.database.git_user_email,
                "feature_dimensions": self.database.feature_dimensions,
                "feature_bins": self.database.feature_bins,
                "migration_interval": self.database.migration_interval,
                "migration_rate": self.database.migration_rate,
                "random_seed": self.database.random_seed,
                "evolution_target": self.database.evolution_target,
                "signature_similarity_threshold": self.database.signature_similarity_threshold,
                "num_inspirations": self.database.num_inspirations,
                "island_programs_per_switch": self.database.island_programs_per_switch,
                "island_diversity_sample_size": self.database.island_diversity_sample_size,
                "island_diversity_max_comparisons": self.database.island_diversity_max_comparisons,
                "diversity_cache_size": self.database.diversity_cache_size,
                "feature_scaling_method": self.database.feature_scaling_method,
                # Feature statistics (sliding window & robust scaling)
                "feature_stats_window_size": self.database.feature_stats_window_size,
                "feature_stats_recompute_interval": self.database.feature_stats_recompute_interval,
                "feature_stats_min_samples": self.database.feature_stats_min_samples,
                "feature_stats_robust_low_q": self.database.feature_stats_robust_low_q,
                "feature_stats_robust_high_q": self.database.feature_stats_robust_high_q,
                "feature_scaling_method_per_dim": self.database.feature_scaling_method_per_dim,
                "minhash_num_perm": self.database.minhash_num_perm,
                "minhash_shingle_len": self.database.minhash_shingle_len,
                "commit_message_template": self.database.commit_message_template,
                "commit_message_max_metrics": self.database.commit_message_max_metrics,
                # Deduplication and migration diversity
                "dedup_exact_enabled": self.database.dedup_exact_enabled,
                "dedup_near_enabled": self.database.dedup_near_enabled,
                "dedup_near_similarity_threshold": self.database.dedup_near_similarity_threshold,
                "migration_diversity_topk": self.database.migration_diversity_topk,
            },
            "evaluator": {
                "timeout": self.evaluator.timeout,
                "max_retries": self.evaluator.max_retries,
                # Note: resource limits not implemented
                # "memory_limit_mb": self.evaluator.memory_limit_mb,
                # "cpu_limit": self.evaluator.cpu_limit,
                "cascade_evaluation": self.evaluator.cascade_evaluation,
                "cascade_thresholds": self.evaluator.cascade_thresholds,
                "parallel_evaluations": self.evaluator.parallel_evaluations,
                # Note: distributed evaluation not implemented
                # "distributed": self.evaluator.distributed,
                "use_llm_feedback": self.evaluator.use_llm_feedback,
                "llm_feedback_weight": self.evaluator.llm_feedback_weight,
                "require_evaluate_before_commit": self.evaluator.require_evaluate_before_commit,
            },
        }

    def to_yaml(self, path: Union[str, Path]) -> None:
        """Save configuration to a YAML file"""
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)


def load_config(config_path: Optional[Union[str, Path]] = None) -> Config:
    """Load configuration from a YAML file or use defaults"""
    if config_path and os.path.exists(config_path):
        config = Config.from_yaml(config_path)
    else:
        config = Config()

        # Use environment variables if available
        config.llm.apply_env_defaults()

    # Make the system message available to the individual models, in case it is not provided from the prompt sampler
    config.llm.update_model_params({"system_message": config.prompt.system_message})

    return config
