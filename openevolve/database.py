"""
Program database for OpenEvolve
"""

import json
import logging
import os
import random
import time
import uuid
import hashlib
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Dict, List, Optional, Set, Tuple, cast, TypedDict, Deque
from collections import deque
import math
import functools

from openevolve.config import DatabaseConfig
from openevolve.utils.metrics_utils import safe_numeric_average
from openevolve.utils.diff_utils import clean_diff, minhash_signature, minhash_similarity
from openevolve.utils.git_utils import diff_between

logger = logging.getLogger(__name__)


# -------- Typed helpers --------


class DiversityCacheEntry(TypedDict):
    value: float
    timestamp: float


class FeatureStats(TypedDict):
    min: float
    max: float
 
@dataclass
class Program:
    """Represents a program in the database"""

    # Program identification
    id: str
    # Commit identifier (no longer store full code text)
    commit_hash: str
    # Added: commit information and its diff
    prompt_diff: Optional[str] = None  # Cleaned diff with filenames, for prompts
    hash_diff: Optional[str] = None    # Fully cleaned diff, used for MinHash
    minhash_signature: List[int] = field(default_factory=list)
    language: str = "python"

    # Evolution information
    parent_id: Optional[str] = None
    generation: int = 0
    timestamp: float = field(default_factory=time.time)
    iteration_found: int = 0  # Track which iteration this program was found

    # Performance metrics
    metrics: Dict[str, float] = field(default_factory=dict)

    # Derived features
    complexity: float = 0.0
    diversity: float = 0.0

    # Metadata
    metadata: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Program":
        """Create from dictionary representation"""
        # Get the valid field names for the Program dataclass
        valid_fields = {f.name for f in fields(cls)}

        # Filter the data to only include valid fields
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}

        # Log if we're filtering out any fields
        if len(filtered_data) != len(data):
            filtered_out = set(data.keys()) - set(filtered_data.keys())
            logger.debug(f"Filtered out unsupported fields when loading Program: {filtered_out}")

        return cls(**filtered_data)


class ProgramDatabase:
    """
    Database for storing and sampling programs during evolution

    The database implements a combination of MAP-Elites algorithm and
    island-based population model to maintain diversity during evolution.
    It also tracks the absolute best program separately to ensure it's never lost.
    """

    def __init__(self, config: DatabaseConfig):
        self.config = config

        # Persistence switch controlled by config.in_memory
        # When in_memory is True, we avoid automatic disk IO unless an explicit path is provided
        self.persistence_enabled: bool = not getattr(config, "in_memory", True)

        # In-memory program storage
        self.programs: Dict[str, Program] = {}

        # Feature grid for MAP-Elites
        self.feature_map: Dict[str, str] = {}

        # Handle both int and dict types for feature_bins
        if isinstance(config.feature_bins, int):
            self.feature_bins = max(
                config.feature_bins,
                int(pow(config.archive_size, 1 / len(config.feature_dimensions)) + 0.99),
            )
        else:
            # If dict, keep as is (we'll use feature_bins_per_dim instead)
            self.feature_bins = 10  # Default fallback for backward compatibility

        # Island populations
        self.islands: List[Set[str]] = [set() for _ in range(config.num_islands)]

        # Island management attributes
        self.current_island: int = 0
        self.island_generations: List[int] = [0] * config.num_islands
        self.last_migration_generation: int = 0
        self.migration_interval: int = getattr(config, "migration_interval", 10)  # Default to 10
        self.migration_rate: float = getattr(config, "migration_rate", 0.1)  # Default to 0.1

        # Archive of elite programs
        self.archive: Set[str] = set()

        # Track the absolute best program separately
        self.best_program_id: Optional[str] = None

        # Track best program per island for proper island-based evolution
        self.island_best_programs: List[Optional[str]] = [None] * config.num_islands

        # Track the last iteration number (for resuming)
        self.last_iteration: int = 0

        # Global exact-dedup set based on normalized diff hash (SHA1 of hash_diff/prompt_diff)
        self.seen_equiv_keys: Set[str] = set()
        # Optional mapping to the first program id seen for a given equivalence key
        self.key_to_program_id: Dict[str, str] = {}

        # Configure persistence and optionally load from disk
        if self.persistence_enabled:
            # Use default db path if not provided: <git_repo_path>/.openevolve/db
            if not getattr(self.config, "db_path", None):
                try:
                    repo_path = getattr(self.config, "git_repo_path", ".") or "."
                    repo_abs = os.path.abspath(repo_path)
                    default_db_path = os.path.join(repo_abs, ".openevolve", "db")
                    self.config.db_path = default_db_path
                    logger.info(
                        f"Persistence enabled and no db_path specified; using default path: {self.config.db_path}"
                    )
                except Exception as e:
                    logger.warning(f"Failed to determine default db_path: {e}")

            if self.config.db_path and os.path.exists(self.config.db_path):
                self.load(self.config.db_path)
        else:
            # In-memory mode: ignore db_path for automatic loading
            if getattr(self.config, "db_path", None):
                logger.info(
                    f"In-memory mode enabled; ignoring configured db_path for auto-load: {self.config.db_path}"
                )

 

        # Initialize instance RNG to avoid global random state pollution
        if config.random_seed is not None:
            self.rng = random.Random(config.random_seed)
            logger.debug(f"Database: Initialized RNG with seed {config.random_seed}")
        else:
            self.rng = random.Random()

        # Diversity caching infrastructure (use stable string key)
        self.diversity_cache: Dict[str, DiversityCacheEntry] = {}
        self.diversity_cache_size: int = getattr(config, "diversity_cache_size", 1000)
        self.diversity_reference_set: List[List[int]] = []  # Reference signatures
        self.diversity_reference_size: int = getattr(config, "diversity_reference_size", 20)
        # Reference set maintenance state
        self.diversity_reference_program_ids: List[str] = []
        self._divref_adds_since_build: int = 0
        self._divref_last_built_at_time: float = time.time()
        self._divref_sig_len: int = int(getattr(config, "minhash_num_perm", 64))

        # Feature scaling infrastructure
        self.feature_stats: Dict[str, FeatureStats] = {}
        self.feature_scaling_method: str = getattr(config, "feature_scaling_method", "minmax")

        # Sliding window + periodic recompute configuration
        self.feature_stats_window_size: int = int(
            getattr(config, "feature_stats_window_size", 5000)
        )
        self.feature_stats_recompute_interval: int = int(
            getattr(config, "feature_stats_recompute_interval", 500)
        )
        self.feature_stats_min_samples: int = int(
            getattr(config, "feature_stats_min_samples", 50)
        )
        # Rebuild trigger on removals (fraction of population removed in a single operation)
        self.feature_stats_rebuild_remove_ratio: float = float(
            getattr(config, "feature_stats_rebuild_remove_ratio", 0.1)
        )
        # Robust scaling quantiles (inclusive lower, upper)
        self.feature_stats_robust_low_q: float = float(
            getattr(config, "feature_stats_robust_low_q", 0.05)
        )
        self.feature_stats_robust_high_q: float = float(
            getattr(config, "feature_stats_robust_high_q", 0.95)
        )
        # Optional per-dimension scaling override
        self.feature_scaling_method_per_dim: Dict[str, str] = cast(
            Dict[str, str], getattr(config, "feature_scaling_method_per_dim", {})
        )

        # In-memory sliding window buffers and caches (not persisted)
        self._feature_value_buffer: Dict[str, Deque[float]] = {}
        self._feature_updates_since_recompute: Dict[str, int] = {}
        # Cache structure per feature: { "min", "max", "q_low", "q_high", "mean", "std" }
        self._feature_stats_cache: Dict[str, Dict[str, float]] = {}

        # Per-dimension bins support
        if hasattr(config, "feature_bins") and isinstance(config.feature_bins, dict):
            self.feature_bins_per_dim = config.feature_bins
        else:
            # Backward compatibility - use same bins for all dimensions
            self.feature_bins_per_dim = {
                dim: self.feature_bins for dim in config.feature_dimensions
            }

        logger.info(f"Initialized program database with {len(self.programs)} programs")

        # Runtime bookkeeping for scaling freeze and rebin
        self._total_adds: int = 0
        self._feature_stats_updates_frozen: bool = False
        self._feature_stats_cache_prev: Dict[str, Dict[str, float]] = {}
        self._rebin_due_to_drift: bool = False
        self._last_rebin_at_adds: int = 0
        self._last_rebin_at_time: float = time.time()

    def add(
        self, program: Program, iteration: Optional[int] = None, target_island: Optional[int] = None
    ) -> str:
        """
        Add a program to the database

        Args:
            program: Program to add
            iteration: Current iteration (defaults to last_iteration)
            target_island: Specific island to add to (uses current_island if None)

        Returns:
            Program ID
        """
        # Store the program
        # If iteration is provided, update the program's iteration_found
        if iteration is not None:
            program.iteration_found = iteration
            # Update last_iteration if needed
            self.last_iteration = max(self.last_iteration, iteration)

        self.programs[program.id] = program

        # Ensure MinHash signature exists when hash_diff is available
        if program.hash_diff and not program.minhash_signature:
            program.minhash_signature = minhash_signature(
                program.hash_diff,
                num_perm=getattr(self.config, "minhash_num_perm", 64),
                shingle_len=getattr(self.config, "minhash_shingle_len", 5),
            )

        # If no hash_diff yet but we do have a commit_hash, automatically compute diff
        if not program.hash_diff and program.commit_hash:
            try:
                prompt_diff, hash_diff = self._get_diff_from_root(program.commit_hash)
                # Only overwrite if still empty to respect external overrides
                if not program.prompt_diff:
                    program.prompt_diff = prompt_diff
                program.hash_diff = hash_diff
                program.minhash_signature = minhash_signature(
                    hash_diff,
                    num_perm=getattr(self.config, "minhash_num_perm", 64),
                    shingle_len=getattr(self.config, "minhash_shingle_len", 5),
                )
            except Exception as e:
                logger.warning(
                    f"Failed to generate diff for commit {program.commit_hash}: {e}"
                )

        # Fallback exact deduplication (should rarely trigger because pre-eval dedup runs earlier)
        try:
            is_empty_island_clone = bool(program.metadata.get("cloned_for_empty_island"))
        except Exception:
            is_empty_island_clone = False

        if not is_empty_island_clone:
            key = self._equivalence_key(program)
            if key and key in self.seen_equiv_keys:
                # Remove temporary insertion and skip heavy bookkeeping
                if program.id in self.programs:
                    try:
                        del self.programs[program.id]
                    except Exception:
                        self.programs.pop(program.id, None)
                existing_id = self.key_to_program_id.get(key)
                logger.warning(
                    f"Exact duplicate detected in add(): {program.id} duplicates existing {existing_id or 'program with same key'}; skipping insertion"
                )
                # Return canonical existing program id if known
                if existing_id:
                    return existing_id
                # Fallback: scan to find canonical id and update mapping
                try:
                    canonical_id = None
                    for _prog in self.programs.values():
                        try:
                            _k = self._equivalence_key(_prog)
                        except Exception:
                            _k = None
                        if _k == key:
                            canonical_id = _prog.id
                            break
                    if canonical_id:
                        self.key_to_program_id[key] = canonical_id
                        return canonical_id
                except Exception:
                    pass
                # As a last resort, return the original id
                return program.id

        # Calculate feature coordinates for MAP-Elites (write path should update stats)
        feature_coords = self._calculate_feature_coords(program, update_stats=True)

        # Add to feature map (replacing existing if better)
        feature_key = self._feature_coords_to_key(feature_coords)
        should_replace = feature_key not in self.feature_map

        if not should_replace:
            # Check if the existing program still exists before comparing
            existing_program_id = self.feature_map[feature_key]
            if existing_program_id not in self.programs:
                # Stale reference, replace it
                should_replace = True
                logger.debug(
                    f"Replacing stale program reference {existing_program_id} in feature map"
                )
            else:
                # Program exists, compare fitness
                should_replace = self._is_better(program, self.programs[existing_program_id])

        if should_replace:
            # Log significant MAP-Elites events
            coords_dict = {
                self.config.feature_dimensions[i]: feature_coords[i]
                for i in range(len(feature_coords))
            }

            if feature_key not in self.feature_map:
                # New cell occupation
                logger.info("New MAP-Elites cell occupied: %s", coords_dict)
                # Check coverage milestone using per-dimension bins and integer thresholds
                total_possible_cells = 1
                for dim in self.config.feature_dimensions:
                    total_possible_cells *= int(self.feature_bins_per_dim.get(dim, self.feature_bins))
                prev_occupied = len(self.feature_map)
                new_occupied = prev_occupied + 1
                milestones = [0.1, 0.25, 0.5, 0.75, 0.9]
                for m in milestones:
                    target_cells = max(1, int(math.ceil(total_possible_cells * m)))
                    if prev_occupied < target_cells <= new_occupied:
                        coverage = new_occupied / total_possible_cells
                        logger.info(
                            "MAP-Elites coverage reached %.1f%% (%d/%d cells)",
                            coverage * 100,
                            new_occupied,
                            total_possible_cells,
                        )
            else:
                # Cell replacement - existing program being replaced
                existing_program_id = self.feature_map[feature_key]
                if existing_program_id in self.programs:
                    existing_program = self.programs[existing_program_id]
                    new_fitness = safe_numeric_average(program.metrics)
                    existing_fitness = safe_numeric_average(existing_program.metrics)
                    logger.info(
                        "MAP-Elites cell improved: %s (fitness: %.3f -> %.3f)",
                        coords_dict,
                        existing_fitness,
                        new_fitness,
                    )

                    # use MAP-Elites to manage archive
                    if existing_program_id in self.archive:
                        self.archive.discard(existing_program_id)
                        self.archive.add(program.id)

            self.feature_map[feature_key] = program.id

        # Add to specific island (not random!)
        island_idx = target_island if target_island is not None else self.current_island
        island_idx = island_idx % len(self.islands)  # Ensure valid island
        self.islands[island_idx].add(program.id)

        # Track which island this program belongs to
        program.metadata["island"] = island_idx

        # Write derived fields for observability
        if program.hash_diff:
            program.complexity = float(len(program.hash_diff))
        elif program.prompt_diff:
            program.complexity = float(len(program.prompt_diff))
        else:
            program.complexity = 0.0
        try:
            program.diversity = float(self._get_cached_diversity(program))
        except Exception:
            program.diversity = 0.0

        # Update archive
        self._update_archive(program)

        # Enforce population size limit BEFORE updating best program tracking
        # This ensures newly added programs aren't immediately removed
        self._enforce_population_limit(exclude_program_id=program.id)

        # Update the absolute best program tracking (after population enforcement)
        self._update_best_program(program)

        # Update island-specific best program tracking
        self._update_island_best_program(program, island_idx)

        # Register deduplication key after successful registration
        self._register_equivalence_key(program)

        # Save to disk if persistence is enabled
        if self.persistence_enabled and self.config.db_path:
            self._save_program(program)

        logger.debug(f"Added program {program.id} to island {island_idx}")

        # Update counters for freeze/rebin triggers
        self._total_adds += 1
        # Freeze feature stats after warmup if enabled
        if bool(getattr(self.config, "feature_stats_freeze_enabled", False)) and not self._feature_stats_updates_frozen:
            try:
                warmup = int(getattr(self.config, "feature_stats_freeze_after_adds", 2000))
            except Exception:
                warmup = 2000
            if self._total_adds >= max(1, warmup):
                self._feature_stats_updates_frozen = True
                logger.info("Feature statistics updates frozen after warmup")

        # Periodic rebin triggers based on adds/time
        if bool(getattr(self.config, "feature_map_rebin_enabled", False)):
            now = time.time()
            adds_interval = int(getattr(self.config, "feature_map_rebin_interval_adds", 0) or 0)
            secs_interval = float(getattr(self.config, "feature_map_rebin_interval_seconds", 0.0) or 0.0)
            should_by_adds = adds_interval > 0 and (self._total_adds - self._last_rebin_at_adds) >= adds_interval
            should_by_time = secs_interval > 0.0 and (now - self._last_rebin_at_time) >= secs_interval
            if should_by_adds or should_by_time or self._rebin_due_to_drift:
                try:
                    self._rebin_feature_map(quiet=bool(getattr(self.config, "feature_map_rebin_quiet", True)))
                except Exception as e:
                    logger.debug(f"Feature map rebin failed: {e}")
                self._last_rebin_at_adds = self._total_adds
                self._last_rebin_at_time = now
                self._rebin_due_to_drift = False

        # Online/periodic maintenance of diversity reference set (post-write)
        try:
            self._consider_candidate_for_reference_set(program)
            self._divref_adds_since_build += 1
            self._maybe_refresh_diversity_reference_set()
        except Exception as e:
            logger.debug(f"Diversity reference maintenance error: {e}")

        return program.id

    def get(self, program_id: str) -> Optional[Program]:
        """
        Get a program by ID

        Args:
            program_id: Program ID

        Returns:
            Program or None if not found
        """
        return self.programs.get(program_id)

    def sample(self) -> Tuple[Program, List[Program]]:
        """
        Sample a program and inspirations for the next evolution step

        Returns:
            Tuple of (parent_program, inspiration_programs)
        """
        # Select parent program
        parent = self._sample_parent()

        # Select inspirations
        inspirations = self._sample_inspirations(
            parent, n=getattr(self.config, "num_inspirations", 5)
        )

        logger.debug(f"Sampled parent {parent.id} and {len(inspirations)} inspirations")
        return parent, inspirations

    def get_best_program(self, metric: Optional[str] = None) -> Optional[Program]:
        """
        Get the best program based on a metric

        Args:
            metric: Metric to use for ranking (uses combined_score or average if None)

        Returns:
            Best program or None if database is empty
        """
        if not self.programs:
            return None

        # If no specific metric and we have a tracked best program, return it
        if metric is None and self.best_program_id:
            if self.best_program_id in self.programs:
                logger.debug(f"Using tracked best program: {self.best_program_id}")
                return self.programs[self.best_program_id]
            else:
                logger.warning(
                    f"Tracked best program {self.best_program_id} no longer exists, will recalculate"
                )
                self.best_program_id = None

        if metric:
            # Sort by specific metric
            sorted_programs = sorted(
                [p for p in self.programs.values() if metric in p.metrics],
                key=lambda p: p.metrics[metric],
                reverse=True,
            )
            if sorted_programs:
                logger.debug(f"Found best program by metric '{metric}': {sorted_programs[0].id}")
        else:
            # Unified fitness ranking
            sorted_programs = sorted(self.programs.values(), key=self._fitness_value, reverse=True)
            if sorted_programs:
                logger.debug(f"Found best program by unified fitness: {sorted_programs[0].id}")

        return sorted_programs[0] if sorted_programs else None

    def get_top_programs(
        self, n: int = 10, metric: Optional[str] = None, island_idx: Optional[int] = None
    ) -> List[Program]:
        """
        Get the top N programs based on a metric

        Args:
            n: Number of programs to return
            metric: Metric to use for ranking. If None, uses combined_score when
                all candidates have it; otherwise falls back to average of numeric metrics.
            island_idx: If specified, only return programs from this island

        Returns:
            List of top programs
        """
        # Validate island_idx parameter
        if island_idx is not None and (island_idx < 0 or island_idx >= len(self.islands)):
            raise IndexError(f"Island index {island_idx} is out of range (0-{len(self.islands)-1})")

        if not self.programs:
            return []

        # Get candidate programs
        if island_idx is not None:
            # Island-specific query
            island_programs = [
                self.programs[pid] for pid in self.islands[island_idx] if pid in self.programs
            ]
            candidates = island_programs
        else:
            # Global query
            candidates = list(self.programs.values())

        if not candidates:
            return []

        if metric:
            # Sort by specific metric
            sorted_programs = sorted(
                [p for p in candidates if metric in p.metrics],
                key=lambda p: p.metrics[metric],
                reverse=True,
            )
        else:
            # Unified fitness sorting
            sorted_programs = sorted(candidates, key=self._fitness_value, reverse=True)

        return sorted_programs[:n]

    def save(self, path: Optional[str] = None, iteration: int = 0) -> None:
        """
        Save the database to disk

        Args:
            path: Path to save to (uses config.db_path if None)
            iteration: Current iteration number
        """
        # In-memory mode: only save when an explicit path is provided (e.g., checkpoint/snapshot)
        if not self.persistence_enabled and path is None:
            logger.info("In-memory mode: skipping save (no target path provided)")
            return

        save_path = path or self.config.db_path
        if not save_path:
            logger.warning("No database path specified, skipping save")
            return

        # create directory if it doesn't exist
        os.makedirs(save_path, exist_ok=True)

        # Save each program
        for program in self.programs.values():
            self._save_program(program, save_path)

        # Save metadata
        # Always persist seen_equiv_keys when an explicit save path is provided (e.g., snapshot)
        persist_seen_keys = True if path is not None else self.persistence_enabled
        metadata = {
            "feature_map": self.feature_map,
            "islands": [list(island) for island in self.islands],
            "archive": list(self.archive),
            "best_program_id": self.best_program_id,
            "island_best_programs": self.island_best_programs,
            "last_iteration": iteration or self.last_iteration,
            "current_island": self.current_island,
            "island_generations": self.island_generations,
            "last_migration_generation": self.last_migration_generation,
            # Persist dedup keys
            "seen_equiv_keys": list(self.seen_equiv_keys) if persist_seen_keys else [],
        }

        with open(os.path.join(save_path, "metadata.json"), "w") as f:
            json.dump(metadata, f)

        logger.info(f"Saved database with {len(self.programs)} programs to {save_path}")

    def load(self, path: str) -> None:
        """
        Load the database from disk

        Args:
            path: Path to load from
        """
        if not os.path.exists(path):
            logger.warning(f"Database path {path} does not exist, skipping load")
            return

        # Load metadata first
        metadata_path = os.path.join(path, "metadata.json")
        saved_islands = []
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)

            self.feature_map = metadata.get("feature_map", {})
            saved_islands = metadata.get("islands", [])
            self.archive = set(metadata.get("archive", []))
            self.best_program_id = metadata.get("best_program_id")
            self.island_best_programs = metadata.get(
                "island_best_programs", [None] * len(saved_islands)
            )
            self.last_iteration = metadata.get("last_iteration", 0)
            self.current_island = metadata.get("current_island", 0)
            self.island_generations = metadata.get("island_generations", [0] * len(saved_islands))
            self.last_migration_generation = metadata.get("last_migration_generation", 0)
            # Restore dedup keys (guarded by persistence flag)
            try:
                keys = metadata.get("seen_equiv_keys", [])
                if isinstance(keys, list):
                    self.seen_equiv_keys = set(str(k) for k in keys)
            except Exception as e:
                logger.warning(f"Failed to restore seen_equiv_keys: {e}")

            logger.info(f"Loaded database metadata with last_iteration={self.last_iteration}")

        # Load programs
        programs_dir = os.path.join(path, "programs")
        if os.path.exists(programs_dir):
            for program_file in os.listdir(programs_dir):
                if program_file.endswith(".json"):
                    program_path = os.path.join(programs_dir, program_file)
                    try:
                        with open(program_path, "r") as f:
                            program_data = json.load(f)

                        program = Program.from_dict(program_data)
                        self.programs[program.id] = program
                    except Exception as e:
                        logger.warning(f"Error loading program {program_file}: {str(e)}")

        # Reconstruct island assignments from metadata
        self._reconstruct_islands(saved_islands)

        # Ensure island_generations list has correct length
        if len(self.island_generations) != len(self.islands):
            self.island_generations = [0] * len(self.islands)

        # Ensure island_best_programs list has correct length
        if len(self.island_best_programs) != len(self.islands):
            self.island_best_programs = [None] * len(self.islands)

        logger.info(f"Loaded database with {len(self.programs)} programs from {path}")

        # Log the reconstructed island status
        self.log_island_status()

        # If dedup keys are empty but we have programs, rebuild from programs on load
        if not self.seen_equiv_keys and self.programs:
            self._rebuild_seen_keys_from_programs()

        # Repair and rebuild runtime indexes derived from programs to avoid orphan/index loss
        try:
            # 1) Ensure all programs have an island: if any island is empty or programs unassigned, distribute
            total_assigned = sum(len(island) for island in self.islands)
            if total_assigned < len(self.programs):
                # Some programs are not assigned; assign them round-robin and set metadata
                missing_ids = [
                    pid for pid in self.programs.keys()
                    if all(pid not in island for island in self.islands)
                ]
                if missing_ids:
                    for idx, pid in enumerate(missing_ids):
                        island_idx = idx % len(self.islands)
                        self.islands[island_idx].add(pid)
                        try:
                            self.programs[pid].metadata["island"] = island_idx
                        except Exception:
                            pass

            # 2) Rebuild feature_map from scratch using current programs and _is_better()
            rebuilt_feature_map: Dict[str, str] = {}
            for prog in self.programs.values():
                try:
                    coords = self._calculate_feature_coords(prog, update_stats=False)
                except Exception:
                    continue
                key = self._feature_coords_to_key(coords)
                if key not in rebuilt_feature_map:
                    rebuilt_feature_map[key] = prog.id
                else:
                    existing_id = rebuilt_feature_map[key]
                    if existing_id in self.programs:
                        if self._is_better(prog, self.programs[existing_id]):
                            rebuilt_feature_map[key] = prog.id
                    else:
                        rebuilt_feature_map[key] = prog.id
            self.feature_map = rebuilt_feature_map

            # 3) Recalculate island best programs coherently using _is_better()
            new_island_bests: List[Optional[str]] = [None] * len(self.islands)
            for i, island in enumerate(self.islands):
                best_id: Optional[str] = None
                for pid in island:
                    if pid not in self.programs:
                        continue
                    if best_id is None:
                        best_id = pid
                    else:
                        if self._is_better(self.programs[pid], self.programs[best_id]):
                            best_id = pid
                new_island_bests[i] = best_id
            self.island_best_programs = new_island_bests
        except Exception as e:
            logger.warning(f"Post-load repair/rebuild encountered an error: {e}")

        # Always rebuild key->program id map after load to ensure dedup consistency
        try:
            self._rebuild_key_to_program_id_map()
        except Exception as e:
            logger.warning(f"Failed to rebuild key_to_program_id map: {e}")

        # Rebuild feature stats from current programs to avoid read-path pollution after load
        if self.programs:
            try:
                self._rebuild_feature_stats_from_programs()
            except Exception as e:
                logger.warning(f"Failed to rebuild feature stats after load: {e}")

        # Normalize MinHash signatures to current configuration after load
        try:
            self._normalize_minhash_signatures_on_load()
        except Exception as e:
            logger.debug(f"Failed to normalize MinHash signatures after load: {e}")

        # Rebuild diversity reference set after load to ensure freshness
        try:
            self._refresh_diversity_reference_set_full()
        except Exception as e:
            logger.debug(f"Failed to rebuild diversity reference set after load: {e}")

    def _reconstruct_islands(self, saved_islands: List[List[str]]) -> None:
        """
        Reconstruct island assignments from saved metadata

        Args:
            saved_islands: List of island program ID lists from metadata
        """
        # Initialize empty islands
        num_islands = max(len(saved_islands), self.config.num_islands)
        self.islands = [set() for _ in range(num_islands)]

        missing_programs = []
        restored_programs = 0

        # Restore island assignments
        for island_idx, program_ids in enumerate(saved_islands):
            if island_idx >= len(self.islands):
                continue

            for program_id in program_ids:
                if program_id in self.programs:
                    # Program exists, add to island
                    self.islands[island_idx].add(program_id)
                    # Set island metadata on the program
                    self.programs[program_id].metadata["island"] = island_idx
                    restored_programs += 1
                else:
                    # Program missing, track it
                    missing_programs.append((island_idx, program_id))

        # Clean up archive - remove missing programs
        original_archive_size = len(self.archive)
        self.archive = {pid for pid in self.archive if pid in self.programs}

        # Clean up feature_map - remove missing programs
        feature_keys_to_remove = []
        for key, program_id in self.feature_map.items():
            if program_id not in self.programs:
                feature_keys_to_remove.append(key)
        for key in feature_keys_to_remove:
            del self.feature_map[key]

        # Clean up island best programs - remove stale references
        self._cleanup_stale_island_bests()

        # Check best program
        if self.best_program_id and self.best_program_id not in self.programs:
            logger.warning(f"Best program {self.best_program_id} not found, will recalculate")
            self.best_program_id = None

        # Log reconstruction results
        if missing_programs:
            logger.warning(
                f"Found {len(missing_programs)} missing programs during island reconstruction:"
            )
            for island_idx, program_id in missing_programs[:5]:  # Show first 5
                logger.warning(f"  Island {island_idx}: {program_id}")
            if len(missing_programs) > 5:
                logger.warning(f"  ... and {len(missing_programs) - 5} more")

        if original_archive_size > len(self.archive):
            logger.info(
                f"Removed {original_archive_size - len(self.archive)} missing programs from archive"
            )

        if feature_keys_to_remove:
            logger.info(f"Removed {len(feature_keys_to_remove)} missing programs from feature map")

        logger.info(f"Reconstructed islands: restored {restored_programs} programs to islands")

        # If we have programs but no island assignments, distribute them
        if self.programs and sum(len(island) for island in self.islands) == 0:
            logger.info("No island assignments found, distributing programs across islands")
            self._distribute_programs_to_islands()

        # After reconstruction, it's safe to rebin feature map once when enabled
        if bool(getattr(self.config, "feature_map_rebin_enabled", False)):
            try:
                self._rebin_feature_map(quiet=True)
            except Exception:
                pass

    def _distribute_programs_to_islands(self) -> None:
        """
        Distribute loaded programs across islands when no island metadata exists
        """
        program_ids = list(self.programs.keys())

        # Distribute programs round-robin across islands
        for i, program_id in enumerate(program_ids):
            island_idx = i % len(self.islands)
            self.islands[island_idx].add(program_id)
            self.programs[program_id].metadata["island"] = island_idx

        logger.info(f"Distributed {len(program_ids)} programs across {len(self.islands)} islands")

    def _save_program(
        self,
        program: Program,
        base_path: Optional[str] = None,
    ) -> None:
        """
        Save a program to disk

        Args:
            program: Program to save
            base_path: Base path to save to (uses config.db_path if None)
        """
        # Allow explicit snapshot path even in in-memory mode
        save_path = base_path or (self.config.db_path if self.persistence_enabled else None)
        if not save_path:
            return

        # Create programs directory if it doesn't exist
        programs_dir = os.path.join(save_path, "programs")
        os.makedirs(programs_dir, exist_ok=True)

        # Save program
        program_dict = program.to_dict()
        program_path = os.path.join(programs_dir, f"{program.id}.json")

        with open(program_path, "w") as f:
            json.dump(program_dict, f)

    def _calculate_feature_coords(self, program: Program, update_stats: bool = False) -> List[int]:
        """
        Calculate feature coordinates for the MAP-Elites grid

        Args:
            program: Program to calculate features for
            update_stats: When True, update sliding-window feature statistics while scaling.
                Use True for write-path operations (e.g., add). Use False for read-path
                operations (e.g., sampling, logging) to avoid read-path pollution.

        Returns:
            List of feature coordinates
        """
        coords = []

        for dim in self.config.feature_dimensions:
            if dim == "complexity":
                # Use diff length as complexity measure when available
                if program.hash_diff:
                    complexity = len(program.hash_diff)
                elif program.prompt_diff:
                    complexity = len(program.prompt_diff)
                else:
                    complexity = 0
                bin_idx = self._calculate_complexity_bin(complexity, update_stats=update_stats)
                coords.append(bin_idx)
            elif dim == "diversity":
                # Use cached diversity calculation with reference set
                if len(self.programs) < 2:
                    bin_idx = 0
                else:
                    diversity = self._get_cached_diversity(program)
                    # If program has no diff text, treat diversity as neutral and do NOT update stats
                    has_text = bool(program.hash_diff or program.prompt_diff)
                    bin_idx = self._calculate_diversity_bin(
                        diversity, update_stats=(update_stats and has_text)
                    )
                coords.append(bin_idx)
            elif dim == "score":
                # Use average of numeric metrics
                if not program.metrics:
                    bin_idx = 0
                else:
                    avg_score = safe_numeric_average(program.metrics)
                    # Update stats and scale
                    if update_stats:
                        self._update_feature_stats("score", avg_score)
                    scaled_value = self._scale_feature_value("score", avg_score)
                    num_bins = self.feature_bins_per_dim.get("score", self.feature_bins)
                    bin_idx = int(scaled_value * num_bins)
                    bin_idx = max(0, min(num_bins - 1, bin_idx))
                coords.append(bin_idx)
            elif dim in program.metrics:
                # Use specific metric
                score = program.metrics[dim]
                # Update stats and scale
                if update_stats:
                    self._update_feature_stats(dim, score)
                scaled_value = self._scale_feature_value(dim, score)
                num_bins = self.feature_bins_per_dim.get(dim, self.feature_bins)
                bin_idx = int(scaled_value * num_bins)
                bin_idx = max(0, min(num_bins - 1, bin_idx))
                coords.append(bin_idx)
            else:
                # Feature not found - this is an error
                raise ValueError(
                    f"Feature dimension '{dim}' specified in config but not found in program metrics. "
                    f"Available metrics: {list(program.metrics.keys())}. "
                    f"Either remove '{dim}' from feature_dimensions or ensure your evaluator returns it."
                )
        # Only log coordinates at debug level for troubleshooting
        logger.debug(
            "MAP-Elites coords: %s",
            str({self.config.feature_dimensions[i]: coords[i] for i in range(len(coords))}),
        )
        return coords

    def _calculate_complexity_bin(self, complexity: int, update_stats: bool = False) -> int:
        """
        Calculate the bin index for a given complexity value using feature scaling.

        Args:
            complexity: The complexity value (change lines)
            update_stats: When True, push value into sliding-window stats prior to scaling.

        Returns:
            Bin index in range [0, self.feature_bins - 1]
        """
        # Update feature statistics (only when requested by write path)
        if update_stats:
            self._update_feature_stats("complexity", float(complexity))

        # Scale the value using configured method
        scaled_value = self._scale_feature_value("complexity", float(complexity))

        # Get number of bins for this dimension
        num_bins = self.feature_bins_per_dim.get("complexity", self.feature_bins)

        # Convert to bin index
        bin_idx = int(scaled_value * num_bins)

        # Ensure bin index is within valid range
        bin_idx = max(0, min(num_bins - 1, bin_idx))

        return bin_idx

    def _calculate_diversity_bin(self, diversity: float, update_stats: bool = False) -> int:
        """
        Calculate the bin index for a given diversity value using feature scaling.

        Args:
            diversity: The average fast code diversity to other programs
            update_stats: When True, push value into sliding-window stats prior to scaling.

        Returns:
            Bin index in range [0, self.feature_bins - 1]
        """
        # Update feature statistics (only when requested by write path)
        if update_stats:
            self._update_feature_stats("diversity", diversity)

        # Scale the value using configured method
        scaled_value = self._scale_feature_value("diversity", diversity)

        # Get number of bins for this dimension
        num_bins = self.feature_bins_per_dim.get("diversity", self.feature_bins)

        # Convert to bin index
        bin_idx = int(scaled_value * num_bins)

        # Ensure bin index is within valid range
        bin_idx = max(0, min(num_bins - 1, bin_idx))

        return bin_idx

    def _feature_coords_to_key(self, coords: List[int]) -> str:
        """
        Convert feature coordinates to a string key

        Args:
            coords: Feature coordinates

        Returns:
            String key
        """
        return "-".join(str(c) for c in coords)

    def _is_better(self, program1: Program, program2: Program) -> bool:
        """
        Determine if program1 is better than program2

        Args:
            program1: First program
            program2: Second program

        Returns:
            True if program1 is better than program2
        """
        # If no metrics, use newest
        if not program1.metrics and not program2.metrics:
            return program1.timestamp > program2.timestamp

        # If only one has metrics, it's better
        if program1.metrics and not program2.metrics:
            return True
        if not program1.metrics and program2.metrics:
            return False

        # Check for combined_score first (this is the preferred metric)
        if "combined_score" in program1.metrics and "combined_score" in program2.metrics:
            return program1.metrics["combined_score"] > program2.metrics["combined_score"]

        # Fallback to average of all numeric metrics
        avg1 = safe_numeric_average(program1.metrics)
        avg2 = safe_numeric_average(program2.metrics)

        return avg1 > avg2

    def _fitness_value(self, program: Program) -> float:
        """Return a numeric fitness value for unified sorting/ranking.

        Prefer combined_score when present, otherwise average of numeric metrics.
        Programs with no metrics get -inf to rank them as worst when sorting by fitness.
        """
        try:
            if program.metrics:
                if "combined_score" in program.metrics:
                    return float(program.metrics.get("combined_score", float("-inf")))
                return float(safe_numeric_average(program.metrics))
        except Exception:
            pass
        return float("-inf")

    def _update_archive(self, program: Program) -> None:
        """
        Update the archive of elite programs

        Args:
            program: Program to consider for archive
        """
        # If archive not full, add program
        if len(self.archive) < self.config.archive_size:
            self.archive.add(program.id)
            return

        # Clean up stale references and get valid archive programs
        valid_archive_programs = []
        stale_ids = []

        for pid in self.archive:
            if pid in self.programs:
                valid_archive_programs.append(self.programs[pid])
            else:
                stale_ids.append(pid)

        # Remove stale references from archive
        for stale_id in stale_ids:
            self.archive.discard(stale_id)
            logger.debug(f"Removing stale program {stale_id} from archive")

        # If archive is now not full after cleanup, just add the new program
        if len(self.archive) < self.config.archive_size:
            self.archive.add(program.id)
            return

        # Find worst program among valid programs using unified fitness comparator
        if valid_archive_programs:
            worst_program = min(valid_archive_programs, key=self._fitness_value)

            # Replace if new program is better than the worst in archive
            if self._is_better(program, worst_program):
                self.archive.remove(worst_program.id)
                self.archive.add(program.id)
        else:
            # No valid programs in archive, just add the new one
            self.archive.add(program.id)

    def _update_best_program(self, program: Program) -> None:
        """
        Update the absolute best program tracking

        Args:
            program: Program to consider as the new best
        """
        # If we don't have a best program yet, this becomes the best
        if self.best_program_id is None:
            self.best_program_id = program.id
            logger.debug(f"Set initial best program to {program.id}")
            return

        # Compare with current best program (if it still exists)
        if self.best_program_id not in self.programs:
            logger.warning(
                f"Best program {self.best_program_id} no longer exists, clearing reference"
            )
            self.best_program_id = program.id
            logger.info(f"Set new best program to {program.id}")
            return

        current_best = self.programs[self.best_program_id]

        # Update if the new program is better
        if self._is_better(program, current_best):
            old_id = self.best_program_id
            self.best_program_id = program.id

            # Log the change
            if "combined_score" in program.metrics and "combined_score" in current_best.metrics:
                old_score = current_best.metrics["combined_score"]
                new_score = program.metrics["combined_score"]
                score_diff = new_score - old_score
                logger.info(
                    f"New best program {program.id} replaces {old_id} (combined_score: {old_score:.4f} → {new_score:.4f}, +{score_diff:.4f})"
                )
            else:
                logger.info(f"New best program {program.id} replaces {old_id}")

    def _update_island_best_program(self, program: Program, island_idx: int) -> None:
        """
        Update the best program tracking for a specific island

        Args:
            program: Program to consider as the new best for the island
            island_idx: Island index
        """
        # Ensure island_idx is valid
        if island_idx >= len(self.island_best_programs):
            logger.warning(f"Invalid island index {island_idx}, skipping island best update")
            return

        # If island doesn't have a best program yet, this becomes the best
        current_island_best_id = self.island_best_programs[island_idx]
        if current_island_best_id is None:
            self.island_best_programs[island_idx] = program.id
            logger.debug(f"Set initial best program for island {island_idx} to {program.id}")
            return

        # Check if current best still exists
        if current_island_best_id not in self.programs:
            logger.warning(
                f"Island {island_idx} best program {current_island_best_id} no longer exists, updating to {program.id}"
            )
            self.island_best_programs[island_idx] = program.id
            return

        current_island_best = self.programs[current_island_best_id]

        # Update if the new program is better
        if self._is_better(program, current_island_best):
            old_id = current_island_best_id
            self.island_best_programs[island_idx] = program.id

            # Log the change
            if (
                "combined_score" in program.metrics
                and "combined_score" in current_island_best.metrics
            ):
                old_score = current_island_best.metrics["combined_score"]
                new_score = program.metrics["combined_score"]
                score_diff = new_score - old_score
                logger.debug(
                    f"Island {island_idx}: New best program {program.id} replaces {old_id} "
                    f"(combined_score: {old_score:.4f} → {new_score:.4f}, +{score_diff:.4f})"
                )
            else:
                logger.debug(
                    f"Island {island_idx}: New best program {program.id} replaces {old_id}"
                )

    def _sample_parent(self) -> Program:
        """
        Sample a parent program from the current island for the next evolution step

        Returns:
            Parent program from current island
        """
        # Use exploration_ratio and exploitation_ratio to decide sampling strategy
        rand_val = self.rng.random()

        if rand_val < self.config.exploration_ratio:
            # EXPLORATION: Sample from current island (diverse sampling)
            return self._sample_exploration_parent()
        elif rand_val < self.config.exploration_ratio + self.config.exploitation_ratio:
            # EXPLOITATION: Sample from archive (elite programs)
            return self._sample_exploitation_parent()
        else:
            # RANDOM: Sample from any program (remaining probability)
            return self._sample_random_parent()

    def _sample_exploration_parent(self) -> Program:
        """
        Sample a parent for exploration (from current island)
        """
        current_island_programs = self.islands[self.current_island]

        if not current_island_programs:
            # If current island is empty, initialize with a cloned program (avoid multi-island assignment)
            source_prog: Optional[Program] = None
            if self.best_program_id and self.best_program_id in self.programs:
                source_prog = self.programs[self.best_program_id]
            else:
                try:
                    source_prog = next(iter(self.programs.values()))
                except StopIteration:
                    raise ValueError("No programs available to initialize empty island")

            migrant_copy = Program(
                id=self._generate_migrant_id(source_prog.id, self.current_island),
                commit_hash=source_prog.commit_hash,
                prompt_diff=source_prog.prompt_diff,
                hash_diff=source_prog.hash_diff,
                minhash_signature=source_prog.minhash_signature.copy(),
                language=source_prog.language,
                parent_id=source_prog.id,
                generation=source_prog.generation,
                metrics=source_prog.metrics.copy(),
                metadata={**source_prog.metadata, "migrant": True, "cloned_for_empty_island": True, "source_island": cast(Optional[int], source_prog.metadata.get("island")), "source_program_id": source_prog.id},
            )
            new_id = self.add(migrant_copy, target_island=self.current_island)
            logger.debug(f"Initialized empty island {self.current_island} with cloned program {new_id}")
            return self.programs[new_id]

        # Clean up stale references and sample from current island
        valid_programs = [pid for pid in current_island_programs if pid in self.programs]

        # Remove stale program IDs from island
        if len(valid_programs) < len(current_island_programs):
            stale_ids = current_island_programs - set(valid_programs)
            logger.debug(
                f"Removing {len(stale_ids)} stale program IDs from island {self.current_island}"
            )
            for stale_id in stale_ids:
                self.islands[self.current_island].discard(stale_id)

        # If no valid programs after cleanup, reinitialize island
        if not valid_programs:
            logger.warning(
                f"Island {self.current_island} has no valid programs after cleanup, reinitializing"
            )
            reinit_source_prog: Optional[Program] = None
            if self.best_program_id and self.best_program_id in self.programs:
                reinit_source_prog = self.programs[self.best_program_id]
            else:
                try:
                    reinit_source_prog = next(iter(self.programs.values()))
                except StopIteration:
                    raise ValueError("No programs available to reinitialize island")

            migrant_copy = Program(
                id=self._generate_migrant_id(reinit_source_prog.id, self.current_island),
                commit_hash=reinit_source_prog.commit_hash,
                prompt_diff=reinit_source_prog.prompt_diff,
                hash_diff=reinit_source_prog.hash_diff,
                minhash_signature=reinit_source_prog.minhash_signature.copy(),
                language=reinit_source_prog.language,
                parent_id=reinit_source_prog.id,
                generation=reinit_source_prog.generation,
                metrics=reinit_source_prog.metrics.copy(),
                metadata={**reinit_source_prog.metadata, "migrant": True, "cloned_for_empty_island": True, "source_island": cast(Optional[int], reinit_source_prog.metadata.get("island")), "source_program_id": reinit_source_prog.id},
            )
            new_id = self.add(migrant_copy, target_island=self.current_island)
            logger.debug(f"Reinitialized island {self.current_island} with cloned program {new_id}")
            return self.programs[new_id]

        # Sample from valid programs
        parent_id = self.rng.choice(valid_programs)
        return self.programs[parent_id]

    def _sample_exploitation_parent(self) -> Program:
        """
        Sample a parent for exploitation (from archive/elite programs)
        """
        if not self.archive:
            # Fallback to exploration if no archive
            return self._sample_exploration_parent()

        # Clean up stale references in archive
        valid_archive = [pid for pid in self.archive if pid in self.programs]

        # Remove stale program IDs from archive
        if len(valid_archive) < len(self.archive):
            stale_ids = self.archive - set(valid_archive)
            logger.debug(f"Removing {len(stale_ids)} stale program IDs from archive")
            for stale_id in stale_ids:
                self.archive.discard(stale_id)

        # If no valid archive programs, fallback to exploration
        if not valid_archive:
            logger.warning(
                "Archive has no valid programs after cleanup, falling back to exploration"
            )
            return self._sample_exploration_parent()

        # Prefer programs from current island in archive
        archive_programs_in_island = [
            pid
            for pid in valid_archive
            if cast(Optional[int], self.programs[pid].metadata.get("island")) == self.current_island
        ]

        # Prefer island-local archive programs; else use any
        candidate_ids = archive_programs_in_island or valid_archive

        # Optional near-dup filtering using MinHash similarity before picking
        if getattr(self.config, "dedup_near_enabled", False):
            try:
                threshold = float(getattr(self.config, "dedup_near_similarity_threshold", 0.98))
            except Exception:
                threshold = 0.98
            # Build a small target set: top programs in current island
            target_island = self.current_island
            topK = max(1, int(getattr(self.config, "migration_diversity_topk", 20)))
            target_programs = self.get_top_programs(n=topK, island_idx=target_island)
            def _is_too_similar(pid: str) -> bool:
                prog = self.programs[pid]
                for tp in target_programs:
                    if not prog.minhash_signature or not tp.minhash_signature:
                        continue
                    sim = minhash_similarity(prog.minhash_signature, tp.minhash_signature)
                    if sim >= threshold:
                        return True
                return False
            filtered = [pid for pid in candidate_ids if not _is_too_similar(pid)]
            if filtered:
                candidate_ids = filtered

        parent_id = self.rng.choice(candidate_ids)
        return self.programs[parent_id]

    def _sample_random_parent(self) -> Program:
        """
        Sample a completely random parent from all programs
        """
        if not self.programs:
            raise ValueError("No programs available for sampling")

        # Sample randomly from all programs
        program_id = self.rng.choice(list(self.programs.keys()))
        return self.programs[program_id]

    def _sample_inspirations(self, parent: Program, n: int = 5) -> List[Program]:
        """
        Sample inspiration programs for the next evolution step.

        For proper island-based evolution, inspirations are sampled ONLY from the
        current island, maintaining genetic isolation between islands.

        Args:
            parent: Parent program
            n: Number of inspirations to sample

        Returns:
            List of inspiration programs from the current island
        """
        inspirations = []

        # Get the parent's island (should be current_island)
        parent_island = cast(int, parent.metadata.get("island", self.current_island))

        # Get all programs from the current island
        island_program_ids = list(self.islands[parent_island])
        island_programs = [self.programs[pid] for pid in island_program_ids if pid in self.programs]

        if not island_programs:
            logger.warning(f"Island {parent_island} has no programs for inspiration sampling")
            return []

        # Include the island's best program if available and different from parent
        island_best_id = self.island_best_programs[parent_island]
        if (
            island_best_id is not None
            and island_best_id != parent.id
            and island_best_id in self.programs
        ):
            island_best = self.programs[island_best_id]
            inspirations.append(island_best)
            logger.debug(
                f"Including island {parent_island} best program {island_best_id} in inspirations"
            )
        elif island_best_id is not None and island_best_id not in self.programs:
            # Clean up stale island best reference
            logger.warning(
                f"Island {parent_island} best program {island_best_id} no longer exists, clearing reference"
            )
            self.island_best_programs[parent_island] = None

        # Add top programs from the island as inspirations
        top_n = max(1, int(n * self.config.elite_selection_ratio))
        top_island_programs = self.get_top_programs(n=top_n, island_idx=parent_island)
        for program in top_island_programs:
            if program.id not in [p.id for p in inspirations] and program.id != parent.id:
                inspirations.append(program)

        # Add diverse programs from within the island
        if len(island_programs) > n and len(inspirations) < n:
            remaining_slots = n - len(inspirations)

            # Try to sample from different feature cells within the island (read-only coords)
            feature_coords = self._calculate_feature_coords(parent, update_stats=False)
            nearby_programs = []

            # Create a mapping of feature cells to island programs for efficient lookup (cell -> list of program ids)
            island_feature_map = {}
            for prog_id in island_program_ids:
                if prog_id in self.programs:
                    prog = self.programs[prog_id]
                    prog_coords = self._calculate_feature_coords(prog, update_stats=False)
                    cell_key = self._feature_coords_to_key(prog_coords)
                    bucket = island_feature_map.get(cell_key)
                    if bucket is None:
                        island_feature_map[cell_key] = [prog_id]
                    else:
                        bucket.append(prog_id)

            # Try to find programs from nearby feature cells within the island
            for _ in range(remaining_slots * 3):  # Try more times to find nearby programs
                # Perturb coordinates (respect per-dimension bins)
                perturbed_coords = []
                for idx, c in enumerate(feature_coords):
                    dim = self.config.feature_dimensions[idx] if idx < len(self.config.feature_dimensions) else None
                    num_bins = int(self.feature_bins_per_dim.get(dim, self.feature_bins)) if dim is not None else int(self.feature_bins)
                    perturbed_val = c + self.rng.randint(-2, 2)
                    perturbed_coords.append(max(0, min(max(0, num_bins - 1), perturbed_val)))

                cell_key = self._feature_coords_to_key(perturbed_coords)
                if cell_key in island_feature_map:
                    candidate_ids = [
                        pid for pid in island_feature_map[cell_key]
                        if (
                            pid != parent.id
                            and pid not in [p.id for p in inspirations]
                            and pid not in [p.id for p in nearby_programs]
                            and pid in self.programs
                        )
                    ]
                    if candidate_ids:
                        chosen_id = self.rng.choice(candidate_ids)
                        nearby_programs.append(self.programs[chosen_id])
                        if len(nearby_programs) >= remaining_slots:
                            break

            # If we still need more, add random programs from the island
            if len(inspirations) + len(nearby_programs) < n:
                remaining = n - len(inspirations) - len(nearby_programs)

                # Get available programs from the island
                excluded_ids = (
                    {parent.id}
                    .union(p.id for p in inspirations)
                    .union(p.id for p in nearby_programs)
                )
                available_island_ids = [
                    pid
                    for pid in island_program_ids
                    if pid not in excluded_ids and pid in self.programs
                ]

                if available_island_ids:
                    random_ids = self.rng.sample(
                        available_island_ids, min(remaining, len(available_island_ids))
                    )
                    random_programs = [self.programs[pid] for pid in random_ids]
                    nearby_programs.extend(random_programs)

            inspirations.extend(nearby_programs)

        # Log island isolation info
        logger.debug(
            f"Sampled {len(inspirations)} inspirations from island {parent_island} "
            f"(island has {len(island_programs)} programs total)"
        )

        return inspirations[:n]

    def _enforce_population_limit(self, exclude_program_id: Optional[str] = None) -> None:
        """
        Enforce the population size limit by removing worst programs if needed

        Args:
            exclude_program_id: Program ID to never remove (e.g., newly added program)
        """
        if len(self.programs) <= self.config.population_size:
            return

        # Calculate how many programs to remove
        num_to_remove = len(self.programs) - self.config.population_size

        logger.info(
            f"Population size ({len(self.programs)}) exceeds limit ({self.config.population_size}), removing {num_to_remove} programs"
        )

        # Get programs sorted by unified fitness (worst first)
        all_programs = list(self.programs.values())
        sorted_programs = sorted(all_programs, key=self._fitness_value)

        # Remove worst programs, but never remove the best program or excluded program
        programs_to_remove = []
        protected_ids = {self.best_program_id, exclude_program_id} - {None}

        for program in sorted_programs:
            if len(programs_to_remove) >= num_to_remove:
                break
            # Don't remove the best program or excluded program
            if program.id not in protected_ids:
                programs_to_remove.append(program)

        # If we still need to remove more and only have protected programs,
        # remove from the remaining programs anyway (but keep the protected ones)
        if len(programs_to_remove) < num_to_remove:
            remaining_programs = [
                p
                for p in sorted_programs
                if p not in programs_to_remove and p.id not in protected_ids
            ]
            additional_removals = remaining_programs[: num_to_remove - len(programs_to_remove)]
            programs_to_remove.extend(additional_removals)

        # Remove the selected programs
        removed_count = 0
        for program in programs_to_remove:
            program_id = program.id

            # Remove from main programs dict
            if program_id in self.programs:
                del self.programs[program_id]

            # Remove from feature map
            keys_to_remove = []
            for key, pid in self.feature_map.items():
                if pid == program_id:
                    keys_to_remove.append(key)
            for key in keys_to_remove:
                del self.feature_map[key]

            # Remove from islands
            for island in self.islands:
                island.discard(program_id)

            # Remove from archive
            self.archive.discard(program_id)

            removed_count += 1
            logger.debug(f"Removed program {program_id} due to population limit")

        logger.info(f"Population size after cleanup: {len(self.programs)}")

        # Clean up any stale island best program references after removal
        self._cleanup_stale_island_bests()

        # If a large fraction of population was removed in one operation, rebuild stats
        try:
            original_size = len(all_programs)
            if original_size > 0:
                removed_ratio = removed_count / float(original_size)
                if removed_ratio >= self.feature_stats_rebuild_remove_ratio:
                    self._rebuild_feature_stats_from_programs()
                    # Optionally trigger diversity reference set rebuild on large removals
                    if bool(getattr(self.config, "diversity_reference_refresh_by_population_enabled", False)):
                        pop_ratio = float(getattr(self.config, "diversity_reference_rebuild_remove_ratio", 0.1))
                        if removed_ratio >= pop_ratio:
                            self._refresh_diversity_reference_set_full()
        except Exception as e:
            logger.debug(f"Feature stats rebuild check failed: {e}")

    # Island management methods
    def set_current_island(self, island_idx: int) -> None:
        """Set which island is currently being evolved"""
        self.current_island = island_idx % len(self.islands)
        logger.debug(f"Switched to evolving island {self.current_island}")

    def next_island(self) -> int:
        """Move to the next island in round-robin fashion"""
        self.current_island = (self.current_island + 1) % len(self.islands)
        logger.debug(f"Advanced to island {self.current_island}")
        return self.current_island

    def increment_island_generation(self, island_idx: Optional[int] = None) -> None:
        """Increment generation counter for an island"""
        idx = island_idx if island_idx is not None else self.current_island
        self.island_generations[idx] += 1
        logger.debug(f"Island {idx} generation incremented to {self.island_generations[idx]}")

    def should_migrate(self) -> bool:
        """Check if migration should occur based on generation counters"""
        max_generation = max(self.island_generations)
        return (max_generation - self.last_migration_generation) >= self.migration_interval

    def _generate_migrant_id(self, base_id: str, target_island: int) -> str:
        """Generate a UUID for a migrated program. The migrant nature is recorded in metadata."""
        return str(uuid.uuid4())

    def migrate_programs(self) -> None:
        """
        Perform migration between islands

        This should be called periodically to share good solutions between islands
        """
        if len(self.islands) < 2:
            return

        logger.info("Performing migration between islands")

        for i, island in enumerate(self.islands):
            if len(island) == 0:
                continue

            # Select candidate programs from this island for migration (sorted by fitness)
            island_programs = [self.programs[pid] for pid in island if pid in self.programs]
            if not island_programs:
                continue

            island_programs.sort(
                key=lambda p: p.metrics.get("combined_score", safe_numeric_average(p.metrics)),
                reverse=True,
            )

            # Determine how many we want to migrate per target
            desired = max(1, int(len(island_programs) * self.migration_rate))

            # Adjacent islands (ring topology)
            target_islands = [(i + 1) % len(self.islands), (i - 1) % len(self.islands)]

            # Optional near-dup configuration
            near_enabled = bool(getattr(self.config, "dedup_near_enabled", False))
            try:
                near_threshold = float(getattr(self.config, "dedup_near_similarity_threshold", 0.98))
            except Exception:
                near_threshold = 0.98
            topK = max(1, int(getattr(self.config, "migration_diversity_topk", 20)))

            for target_island in target_islands:
                selected = 0
                # Build a diversity reference set within target island
                target_top = self.get_top_programs(n=topK, island_idx=target_island)

                for candidate in island_programs:
                    if selected >= desired:
                        break

                    # Create a copy for migration (avoid removing from source)
                    migrant_copy = Program(
                        id=self._generate_migrant_id(candidate.id, target_island),
                        commit_hash=candidate.commit_hash,
                        prompt_diff=candidate.prompt_diff,
                        hash_diff=candidate.hash_diff,
                        minhash_signature=candidate.minhash_signature.copy(),
                        language=candidate.language,
                        parent_id=candidate.id,
                        generation=candidate.generation,
                        metrics=candidate.metrics.copy(),
                        metadata={**candidate.metadata, "migrant": True, "source_island": i, "source_program_id": candidate.id},
                    )

                    # Exact dedup check (global)
                    if self.is_duplicate(migrant_copy):
                        continue

                    # Near-dup check against target island top-K
                    if near_enabled and migrant_copy.minhash_signature:
                        too_similar = False
                        for tp in target_top:
                            if not tp.minhash_signature:
                                continue
                            sim = minhash_similarity(migrant_copy.minhash_signature, tp.minhash_signature)
                            if sim >= near_threshold:
                                too_similar = True
                                break
                        if too_similar:
                            continue

                    # Register via unified add() to ensure indexes and constraints are applied
                    self.add(migrant_copy, target_island=target_island)
                    selected += 1

                    # Optional: log MAP-Elites coordinates post-registration
                    feature_coords = self._calculate_feature_coords(migrant_copy, update_stats=False)
                    coords_dict = {
                        self.config.feature_dimensions[j]: feature_coords[j]
                        for j in range(len(feature_coords))
                    }
                    logger.info(
                        "Program migrated to island %d at MAP-Elites coords: %s",
                        target_island,
                        coords_dict,
                    )

        # Update last migration generation
        self.last_migration_generation = max(self.island_generations)
        logger.info(f"Migration completed at generation {self.last_migration_generation}")

        # Validate migration results
        self._validate_migration_results()

    def _validate_migration_results(self) -> None:
        """
        Validate migration didn't create inconsistencies

        Checks that:
        1. Program island metadata matches actual island assignment
        2. No programs are assigned to multiple islands
        3. All island best programs exist and are in correct islands
        """
        seen_program_ids = set()

        for i, island in enumerate(self.islands):
            for program_id in island:
                # Check for duplicate assignments
                if program_id in seen_program_ids:
                    logger.error(f"Program {program_id} assigned to multiple islands")
                    continue
                seen_program_ids.add(program_id)

                # Check program exists
                if program_id not in self.programs:
                    logger.warning(f"Island {i} contains nonexistent program {program_id}")
                    continue

                # Check metadata consistency
                program = self.programs[program_id]
                stored_island = cast(Optional[int], program.metadata.get("island"))
                if stored_island != i:
                    logger.warning(
                        f"Island mismatch for program {program_id}: "
                        f"in island {i} but metadata says {stored_island}"
                    )

        # Validate island best programs
        for i, best_id in enumerate(self.island_best_programs):
            if best_id is not None:
                if best_id not in self.programs:
                    logger.warning(f"Island {i} best program {best_id} does not exist")
                elif best_id not in self.islands[i]:
                    logger.warning(f"Island {i} best program {best_id} not in island")

    # ---------------- Deduplication helpers ----------------
    def _equivalence_key(self, program: Program) -> Optional[str]:
        """Return a stable equivalence key for exact deduplication.

        Uses SHA1 of normalized diff text (hash_diff or prompt_diff).
        Returns None if no diff information is available.
        """
        text = program.hash_diff or program.prompt_diff
        if not text:
            return None
        # Keep consistent normalization with clean_diff output
        try:
            return hashlib.sha1(text.encode("utf-8")).hexdigest()
        except Exception:
            return None

    def _find_program_id_by_equivalence_key(self, key: str) -> Optional[str]:
        """Find the first program id with the given equivalence key via linear scan."""
        for prog in self.programs.values():
            try:
                k = self._equivalence_key(prog)
            except Exception:
                k = None
            if k == key:
                return prog.id
        return None

    def is_duplicate(self, program: Program) -> bool:
        """Check if a program is an exact duplicate based on equivalence key."""
        key = self._equivalence_key(program)
        return bool(key and key in self.seen_equiv_keys)

    def _register_equivalence_key(self, program: Program) -> None:
        """Register a program's equivalence key after successful insertion."""
        key = self._equivalence_key(program)
        if key:
            self.seen_equiv_keys.add(key)
            # Remember the first program id for this key
            if key not in self.key_to_program_id:
                self.key_to_program_id[key] = program.id

    def _rebuild_seen_keys_from_programs(self) -> None:
        """Rebuild dedup keys from current programs (used on load when keys missing)."""
        rebuilt = 0
        for prog in self.programs.values():
            key = self._equivalence_key(prog)
            if key and key not in self.seen_equiv_keys:
                self.seen_equiv_keys.add(key)
                if key not in self.key_to_program_id:
                    self.key_to_program_id[key] = prog.id
                rebuilt += 1
        if rebuilt:
            logger.info(f"Rebuilt {rebuilt} dedup keys from programs on load")

    def _rebuild_key_to_program_id_map(self) -> None:
        """Rebuild mapping from equivalence key to the first program id (post-load)."""
        self.key_to_program_id.clear()
        for prog in self.programs.values():
            key = self._equivalence_key(prog)
            if key and key not in self.key_to_program_id:
                self.key_to_program_id[key] = prog.id

    def _cleanup_stale_island_bests(self) -> None:
        """
        Remove stale island best program references

        Cleans up references to programs that no longer exist in the database
        or are not actually in their assigned islands.
        """
        cleaned_count = 0

        for i, best_id in enumerate(self.island_best_programs):
            if best_id is not None:
                should_clear = False

                # Check if program still exists
                if best_id not in self.programs:
                    logger.debug(
                        f"Clearing stale island {i} best program {best_id} (program deleted)"
                    )
                    should_clear = True
                # Check if program is still in the island
                elif best_id not in self.islands[i]:
                    logger.debug(
                        f"Clearing stale island {i} best program {best_id} (not in island)"
                    )
                    should_clear = True

                if should_clear:
                    self.island_best_programs[i] = None
                    cleaned_count += 1

        if cleaned_count > 0:
            logger.info(f"Cleaned up {cleaned_count} stale island best program references")

            # Recalculate best programs for islands that were cleared
            for i, best_id in enumerate(self.island_best_programs):
                if best_id is None and len(self.islands[i]) > 0:
                    # Find new best program for this island
                    island_programs = [
                        self.programs[pid] for pid in self.islands[i] if pid in self.programs
                    ]
                    if island_programs:
                        # Sort by fitness and update
                        best_program = max(
                            island_programs,
                            key=lambda p: p.metrics.get(
                                "combined_score", safe_numeric_average(p.metrics)
                            ),
                        )
                        self.island_best_programs[i] = best_program.id
                        logger.debug(f"Recalculated island {i} best program: {best_program.id}")

    def get_island_stats(self) -> List[dict]:
        """Get statistics for each island"""
        stats = []

        for i, island in enumerate(self.islands):
            island_programs = [self.programs[pid] for pid in island if pid in self.programs]

            if island_programs:
                scores = [
                    p.metrics.get("combined_score", safe_numeric_average(p.metrics))
                    for p in island_programs
                ]

                best_score = max(scores) if scores else 0.0
                avg_score = sum(scores) / len(scores) if scores else 0.0
                diversity = self._calculate_island_diversity(island_programs)
            else:
                best_score = avg_score = diversity = 0.0

            stats.append(
                {
                    "island": i,
                    "population_size": len(island_programs),
                    "best_score": best_score,
                    "average_score": avg_score,
                    "diversity": diversity,
                    "generation": self.island_generations[i],
                    "is_current": i == self.current_island,
                }
            )

        return stats

    def _calculate_island_diversity(self, programs: List[Program]) -> float:
        """Calculate diversity within an island (deterministic version)"""
        if len(programs) < 2:
            return 0.0

        total_diversity = 0
        comparisons = 0

        # Use deterministic sampling instead of random.sample() to ensure consistent results
        sample_size = min(getattr(self.config, "island_diversity_sample_size", 5), len(programs))

        # Sort programs by ID for deterministic ordering
        sorted_programs = sorted(programs, key=lambda p: p.id)

        # Take first N programs instead of random sampling
        sample_programs = sorted_programs[:sample_size]

        # Limit total comparisons for performance
        max_comparisons = getattr(self.config, "island_diversity_max_comparisons", 6)

        for i, prog1 in enumerate(sample_programs):
            for prog2 in sample_programs[i + 1 :]:
                if comparisons >= max_comparisons:
                    break

                # Use fast approximation instead of expensive edit distance
                diversity = self._fast_code_diversity(
                    prog1.minhash_signature,
                    prog2.minhash_signature,
                )
                total_diversity += diversity
                comparisons += 1

            if comparisons >= max_comparisons:
                break

        return total_diversity / max(1, comparisons)

    def _fast_code_diversity(self, sig1: List[int], sig2: List[int]) -> float:
        """Return diversity (1 - Jaccard similarity) between two MinHash signatures."""
        if not sig1 or not sig2 or len(sig1) != len(sig2):
            return 0.0
        return 1.0 - minhash_similarity(sig1, sig2)

    def _get_cached_diversity(self, program: Program) -> float:
        """
        Get diversity score for a program using cache and reference set

        Args:
            program: The program to calculate diversity for

        Returns:
            Diversity score (cached or newly computed)
        """
        text_for_key = program.hash_diff or program.prompt_diff or ""
        # Neutral handling for empty text: return 0.5, no cache updates, no stats updates by caller
        if not text_for_key:
            return 0.5
        try:
            code_key = hashlib.sha1(text_for_key.encode("utf-8")).hexdigest()
        except Exception:
            code_key = ""

        # Check cache first
        if code_key in self.diversity_cache:
            return self.diversity_cache[code_key]["value"]

        # Ensure program has MinHash signature
        if not program.minhash_signature and (program.hash_diff or program.prompt_diff):
            text_for_sig = program.hash_diff or program.prompt_diff
            program.minhash_signature = minhash_signature(
                text_for_sig or "",
                num_perm=getattr(self.config, "minhash_num_perm", 64),
                shingle_len=getattr(self.config, "minhash_shingle_len", 5),
            )

        # Update reference set if needed (prefer full refresh for consistency)
        need_build = (
            not self.diversity_reference_set
            or len(self.diversity_reference_set) < self.diversity_reference_size
        )
        try:
            sig_len = len(program.minhash_signature) if program.minhash_signature else int(getattr(self.config, "minhash_num_perm", 64))
        except Exception:
            sig_len = int(getattr(self.config, "minhash_num_perm", 64))
        if need_build or sig_len != int(getattr(self, "_divref_sig_len", sig_len)):
            try:
                self._refresh_diversity_reference_set_full()
            except Exception:
                self._update_diversity_reference_set()

        # Compute diversity against reference set
        diversity_scores = []
        for ref_sig in self.diversity_reference_set:
            diversity_scores.append(1.0 - minhash_similarity(program.minhash_signature, ref_sig))

        diversity = (
            sum(diversity_scores) / max(1, len(diversity_scores)) if diversity_scores else 0.0
        )

        # Cache the result with LRU eviction (skip caching for empty text which is handled above)
        if code_key:
            self._cache_diversity_value(code_key, diversity)

        return diversity

    def _normalize_minhash_signatures_on_load(self) -> None:
        """Normalize all programs' MinHash signatures to current configuration after load.

        Recompute signatures that do not match the configured num_perm; clear if no diff text.
        """
        try:
            target_num_perm = int(getattr(self.config, "minhash_num_perm", 64))
            shingle_len = int(getattr(self.config, "minhash_shingle_len", 5))
        except Exception:
            target_num_perm = 64
            shingle_len = 5
        updated = 0
        for prog in self.programs.values():
            sig = prog.minhash_signature
            need_rebuild = False
            try:
                if not isinstance(sig, list) or len(sig) != target_num_perm:
                    need_rebuild = True
            except Exception:
                need_rebuild = True
            if need_rebuild:
                text = prog.hash_diff or prog.prompt_diff or ""
                if text:
                    try:
                        prog.minhash_signature = minhash_signature(
                            text,
                            num_perm=target_num_perm,
                            shingle_len=shingle_len,
                        )
                        updated += 1
                    except Exception:
                        # Leave as-is if recompute fails
                        pass
                else:
                    prog.minhash_signature = []
        if updated:
            logger.info(
                f"Normalized MinHash signatures for {updated} programs to num_perm={target_num_perm}"
            )

    def _update_diversity_reference_set(self) -> None:
        """Update the reference set for diversity calculation"""
        if len(self.programs) == 0:
            return

        # Select diverse programs for reference set
        all_programs = list(self.programs.values())

        if len(all_programs) <= self.diversity_reference_size:
            sigs = []
            ids = []
            for p in all_programs:
                if not p.minhash_signature:
                    p.minhash_signature = minhash_signature(
                        p.hash_diff or p.prompt_diff or "",
                        num_perm=getattr(self.config, "minhash_num_perm", 64),
                        shingle_len=getattr(self.config, "minhash_shingle_len", 5),
                    )
                sigs.append(p.minhash_signature)
                ids.append(p.id)
            self.diversity_reference_set = sigs
            self.diversity_reference_program_ids = ids
        else:
            # Select programs with maximum diversity based on MinHash
            selected: List[Program] = []
            remaining = all_programs.copy()

            # Start with a random program
            first_idx = self.rng.randint(0, len(remaining) - 1)
            selected.append(remaining.pop(first_idx))

            # Greedily add programs that maximize diversity to selected set
            while len(selected) < self.diversity_reference_size and remaining:
                max_diversity = -1.0
                best_idx = -1

                for i, candidate in enumerate(remaining):
                    if not candidate.minhash_signature:
                        candidate.minhash_signature = minhash_signature(
                            candidate.hash_diff or candidate.prompt_diff or "",
                            num_perm=getattr(self.config, "minhash_num_perm", 64),
                            shingle_len=getattr(self.config, "minhash_shingle_len", 5),
                        )
                    min_div = float("inf")
                    for selected_prog in selected:
                        if not selected_prog.minhash_signature:
                            selected_prog.minhash_signature = minhash_signature(
                                selected_prog.hash_diff or selected_prog.prompt_diff or "",
                                num_perm=getattr(self.config, "minhash_num_perm", 64),
                                shingle_len=getattr(self.config, "minhash_shingle_len", 5),
                            )
                        div = 1.0 - minhash_similarity(candidate.minhash_signature, selected_prog.minhash_signature)
                        min_div = min(min_div, div)

                    if min_div > max_diversity:
                        max_diversity = min_div
                        best_idx = i

                if best_idx >= 0:
                    selected.append(remaining.pop(best_idx))

            self.diversity_reference_set = [p.minhash_signature for p in selected]
            self.diversity_reference_program_ids = [p.id for p in selected]

        logger.debug(
            f"Updated diversity reference set with {len(self.diversity_reference_set)} programs"
        )

    # ---------------- Diversity reference set maintenance (online + periodic) ----------------
    def _consider_candidate_for_reference_set(self, program: Program) -> None:
        """Online maintenance: consider adding/replacing a candidate into the reference set.

        Uses a farthest-first style heuristic to keep the set diverse.
        """
        # Ensure candidate has signature
        if not program.minhash_signature:
            text_for_sig = program.hash_diff or program.prompt_diff or ""
            program.minhash_signature = minhash_signature(
                text_for_sig,
                num_perm=getattr(self.config, "minhash_num_perm", 64),
                shingle_len=getattr(self.config, "minhash_shingle_len", 5),
            )

        sig = program.minhash_signature
        if not sig:
            return

        # If reference set empty or not full, append directly
        if not self.diversity_reference_set or len(self.diversity_reference_set) < self.diversity_reference_size:
            self.diversity_reference_set.append(sig)
            self.diversity_reference_program_ids.append(program.id)
            # Update bookkeeping
            self._divref_sig_len = len(sig)
            if len(self.diversity_reference_set) == self.diversity_reference_size:
                self._divref_last_built_at_time = time.time()
            return

        # Signature length mismatch -> trigger a full refresh instead of online replace
        if len(sig) != self._divref_sig_len:
            self._refresh_diversity_reference_set_full()
            return

        # Avoid duplicate ids
        if program.id in self.diversity_reference_program_ids:
            return

        # Compute candidate's minimum similarity to current reference set
        cand_min_sim = 1.0
        for ref_sig in self.diversity_reference_set:
            try:
                s = minhash_similarity(sig, ref_sig)
            except Exception:
                s = 1.0
            if s < cand_min_sim:
                cand_min_sim = s

        # For each existing member, compute its nearest neighbor similarity within the set
        # Identify the most redundant member (highest nearest-neighbor similarity)
        most_redundant_idx = -1
        most_redundant_nn_sim = -1.0
        k = len(self.diversity_reference_set)
        for i in range(k):
            ref_i = self.diversity_reference_set[i]
            nn_sim = 1.0
            for j in range(k):
                if i == j:
                    continue
                try:
                    s = minhash_similarity(ref_i, self.diversity_reference_set[j])
                except Exception:
                    s = 1.0
                if s < nn_sim:
                    nn_sim = s
            # Choose the member with largest nearest-neighbor similarity (closest to others)
            if nn_sim > most_redundant_nn_sim:
                most_redundant_nn_sim = nn_sim
                most_redundant_idx = i

        margin = float(getattr(self.config, "diversity_reference_online_margin", 0.05))
        # Replace if candidate is notably less similar to the set than the most-redundant member's nearest neighbor
        if cand_min_sim + margin < most_redundant_nn_sim and 0 <= most_redundant_idx < len(self.diversity_reference_set):
            self.diversity_reference_set[most_redundant_idx] = sig
            # Maintain same ordering length in ids list
            if most_redundant_idx < len(self.diversity_reference_program_ids):
                self.diversity_reference_program_ids[most_redundant_idx] = program.id
            else:
                # Fallback safety
                self.diversity_reference_program_ids.append(program.id)

    def _prune_diversity_reference_set(self) -> None:
        """Remove entries whose program ids are stale or signatures invalid."""
        if not self.diversity_reference_set:
            return
        new_sigs: List[List[int]] = []
        new_ids: List[str] = []
        for idx, sig in enumerate(self.diversity_reference_set):
            pid = self.diversity_reference_program_ids[idx] if idx < len(self.diversity_reference_program_ids) else None
            if pid is None or pid not in self.programs:
                continue
            if not isinstance(sig, list) or (self._divref_sig_len and len(sig) != self._divref_sig_len):
                continue
            new_sigs.append(sig)
            new_ids.append(pid)
        self.diversity_reference_set = new_sigs
        self.diversity_reference_program_ids = new_ids

    def _should_refresh_diversity_reference_set(self) -> bool:
        """Check periodic refresh conditions based on config toggles."""
        now = time.time()
        # Inserts-based trigger
        if bool(getattr(self.config, "diversity_reference_refresh_by_inserts_enabled", True)):
            try:
                threshold = int(getattr(self.config, "diversity_reference_refresh_adds", 40))
            except Exception:
                threshold = 40
            if self._divref_adds_since_build >= max(1, threshold):
                return True
        # Time-based trigger
        if bool(getattr(self.config, "diversity_reference_refresh_by_time_enabled", False)):
            try:
                seconds = float(getattr(self.config, "diversity_reference_refresh_seconds", 300.0))
            except Exception:
                seconds = 300.0
            if now - float(getattr(self, "_divref_last_built_at_time", 0.0)) >= seconds:
                return True
        return False

    def _maybe_refresh_diversity_reference_set(self) -> None:
        if self._should_refresh_diversity_reference_set():
            self._refresh_diversity_reference_set_full()

    def _refresh_diversity_reference_set_full(self) -> None:
        """Rebuild the diversity reference set and parallel program id list using a greedy farthest-first strategy."""
        # If no programs, clear
        if not self.programs:
            self.diversity_reference_set = []
            self.diversity_reference_program_ids = []
            self._divref_adds_since_build = 0
            self._divref_last_built_at_time = time.time()
            return

        all_programs: List[Program] = list(self.programs.values())
        # Ensure signatures
        for p in all_programs:
            if not p.minhash_signature:
                p.minhash_signature = minhash_signature(
                    p.hash_diff or p.prompt_diff or "",
                    num_perm=getattr(self.config, "minhash_num_perm", 64),
                    shingle_len=getattr(self.config, "minhash_shingle_len", 5),
                )

        if len(all_programs) <= self.diversity_reference_size:
            self.diversity_reference_set = [p.minhash_signature for p in all_programs]
            self.diversity_reference_program_ids = [p.id for p in all_programs]
        else:
            remaining = all_programs.copy()
            # Start with a random program
            first_idx = self.rng.randint(0, len(remaining) - 1)
            selected: List[Program] = [remaining.pop(first_idx)]
            # Greedy farthest-first
            while len(selected) < self.diversity_reference_size and remaining:
                best_idx = -1
                best_min_div = -1.0
                for i, cand in enumerate(remaining):
                    # Compute minimum diversity (1 - similarity) vs current selected
                    min_div = float("inf")
                    for s in selected:
                        try:
                            sim = minhash_similarity(cand.minhash_signature, s.minhash_signature)
                        except Exception:
                            sim = 1.0
                        div = 1.0 - sim
                        if div < min_div:
                            min_div = div
                    if min_div > best_min_div:
                        best_min_div = min_div
                        best_idx = i
                if best_idx >= 0:
                    selected.append(remaining.pop(best_idx))
                else:
                    break
            self.diversity_reference_set = [p.minhash_signature for p in selected]
            self.diversity_reference_program_ids = [p.id for p in selected]

        # Update bookkeeping
        self._divref_sig_len = len(self.diversity_reference_set[0]) if self.diversity_reference_set else int(getattr(self.config, "minhash_num_perm", 64))
        self._divref_adds_since_build = 0
        self._divref_last_built_at_time = time.time()
        # Invalidate diversity cache because reference set changed
        self._invalidate_diversity_cache()

    def _rebuild_feature_stats_from_programs(self) -> None:
        """Rebuild feature stats buffers and cache from current programs (read-only scan).

        This avoids read-path pollution by recreating statistics using the current
        program set, without incrementally appending via read calls.
        """
        # Reset buffers and counters
        self.feature_stats.clear()
        self._feature_value_buffer.clear()
        self._feature_updates_since_recompute.clear()
        self._feature_stats_cache.clear()

        if not self.programs:
            return

        # Helper to push a value into buffers (without triggering recompute yet)
        def _append(feature_name: str, v: float) -> None:
            stats = self.feature_stats.get(feature_name)
            if stats is None:
                self.feature_stats[feature_name] = {"min": v, "max": v}
            else:
                stats["min"] = min(stats["min"], v)
                stats["max"] = max(stats["max"], v)
            buf = self._feature_value_buffer.get(feature_name)
            if buf is None:
                buf = deque(maxlen=self.feature_stats_window_size)
                self._feature_value_buffer[feature_name] = buf
            buf.append(float(v))

        # Populate buffers from existing programs
        for prog in self.programs.values():
            # complexity
            if prog.hash_diff:
                comp = float(len(prog.hash_diff))
            elif prog.prompt_diff:
                comp = float(len(prog.prompt_diff))
            else:
                comp = 0.0
            _append("complexity", comp)

            # diversity (use stored value if present, otherwise estimate lazily)
            try:
                div = float(prog.diversity) if isinstance(prog.diversity, (int, float)) else 0.0
            except Exception:
                div = 0.0
            _append("diversity", div)

            # aggregated score
            if prog.metrics:
                avg_score = safe_numeric_average(prog.metrics)
                _append("score", float(avg_score))

            # each specific metric used by feature_dimensions
            for dim in self.config.feature_dimensions:
                if dim not in ("complexity", "diversity", "score") and dim in prog.metrics:
                    try:
                        _append(dim, float(prog.metrics[dim]))
                    except Exception:
                        pass

        # Recompute cached statistics for all features we filled
        for feature_name in list(self._feature_value_buffer.keys()):
            try:
                self._recompute_feature_stats(feature_name)
                # mark zero updates since recompute
                self._feature_updates_since_recompute[feature_name] = 0
            except Exception:
                continue

    def _cache_diversity_value(self, code_key: str, diversity: float) -> None:
        """Cache a diversity value with LRU eviction (string key)"""
        # Check if cache is full
        if len(self.diversity_cache) >= self.diversity_cache_size:
            # Remove oldest entry
            oldest_key = min(self.diversity_cache.items(), key=lambda x: x[1]["timestamp"])[0]
            del self.diversity_cache[oldest_key]

        # Add new entry
        self.diversity_cache[code_key] = {"value": diversity, "timestamp": time.time()}

    def _invalidate_diversity_cache(self) -> None:
        """Invalidate the diversity cache when programs change significantly"""
        self.diversity_cache.clear()
        self.diversity_reference_set = []
        logger.debug("Diversity cache invalidated")

    def _update_feature_stats(self, feature_name: str, value: float) -> None:
        """
        Update statistics for a feature dimension

        Args:
            feature_name: Name of the feature dimension
            value: New value to incorporate into stats
        """
        # Skip updates if stats are frozen
        if bool(getattr(self.config, "feature_stats_freeze_enabled", False)) and self._feature_stats_updates_frozen:
            return

        if feature_name not in self.feature_stats:
            self.feature_stats[feature_name] = {
                "min": value,
                "max": value,
            }

        stats = self.feature_stats[feature_name]
        stats["min"] = min(stats["min"], value)
        stats["max"] = max(stats["max"], value)

        # Sliding window buffer for robust/modern scaling (not persisted)
        buf = self._feature_value_buffer.get(feature_name)
        if buf is None:
            buf = deque(maxlen=self.feature_stats_window_size)
            self._feature_value_buffer[feature_name] = buf
        buf.append(float(value))

        # Periodic recompute scheduling (purely periodic, no population-change triggers)
        self._feature_updates_since_recompute[feature_name] = (
            self._feature_updates_since_recompute.get(feature_name, 0) + 1
        )
        self._maybe_recompute_feature_stats(feature_name)

    def _scale_feature_value(self, feature_name: str, value: float) -> float:
        """
        Scale a feature value according to the configured scaling method

        Args:
            feature_name: Name of the feature dimension
            value: Raw feature value

        Returns:
            Scaled value in range [0, 1]
        """
        method = self._get_scaling_method_for_dim(feature_name)

        # Prefer cached window stats when available
        cache = self._feature_stats_cache.get(feature_name, {})

        # Helper: readonly fallback strategy when cache is missing
        def _readonly_fallback() -> float:
            mode = getattr(self.config, "feature_readonly_fallback_mode", "use_feature_stats")
            if mode == "neutral_0_5":
                return 0.5
            if mode == "clip_0_1":
                return min(1.0, max(0.0, float(value)))
            if mode == "static_ranges":
                try:
                    static_ranges = getattr(self.config, "feature_readonly_static_minmax", {})
                    if feature_name in static_ranges and isinstance(static_ranges[feature_name], (list, tuple)) and len(static_ranges[feature_name]) == 2:
                        min_v, max_v = float(static_ranges[feature_name][0]), float(static_ranges[feature_name][1])
                        if max_v <= min_v:
                            return 0.5
                        return min(1.0, max(0.0, (float(value) - min_v) / (max_v - min_v)))
                except Exception:
                    pass
                # fallback to neutral if static range invalid
                return 0.5
            # use_feature_stats (default): try legacy stats; if still missing, final fallback is clip
            stats = self.feature_stats.get(feature_name)
            if stats is not None:
                min_v = stats.get("min")
                max_v = stats.get("max")
                if min_v is not None and max_v is not None and max_v != min_v:
                    return min(1.0, max(0.0, (float(value) - float(min_v)) / (float(max_v) - float(min_v))))
            # last resort
            return min(1.0, max(0.0, float(value)))

        # Robust scaling (recommended default)
        if method == "robust":
            q_low = cache.get("q_low")
            q_high = cache.get("q_high")
            if q_low is None or q_high is None:
                # Not enough stats yet; readonly fallback
                return _readonly_fallback()
            if q_high <= q_low:
                return 0.5
            scaled = (float(value) - q_low) / (q_high - q_low)
            return min(1.0, max(0.0, scaled))

        # Min-max scaling
        if method == "minmax":
            # Use cached min/max if present, else legacy stats
            min_val = cache.get("min")
            max_val = cache.get("max")
            if min_val is None or max_val is None:
                # Readonly fallback
                return _readonly_fallback()
            if max_val == min_val:
                return 0.5
            scaled = (float(value) - float(min_val)) / (float(max_val) - float(min_val))
            return min(1.0, max(0.0, scaled))

        # Percentile rank in current window
        if method == "percentile":
            values = list(self._feature_value_buffer.get(feature_name, []))
            if not values:
                return _readonly_fallback()
            # Linear scan rank (W is small, default 5000)
            count = sum(1 for v in values if v <= float(value))
            return count / len(values)

        # Z-score scaling mapped to [0,1] using erf
        if method == "zscore":
            mean = cache.get("mean")
            std = cache.get("std")
            if mean is None or std is None or std == 0:
                return _readonly_fallback()
            z = (float(value) - mean) / std
            # Map via standard normal CDF approximation using erf
            return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

        # Unknown method -> readonly fallback
        return _readonly_fallback()

    def _scale_feature_value_minmax(self, feature_name: str, value: float) -> float:
        """Helper for min-max scaling"""
        cache = self._feature_stats_cache.get(feature_name)
        if cache is not None and "min" in cache and "max" in cache:
            min_val = cache["min"]
            max_val = cache["max"]
            if max_val == min_val:
                return 0.5
            scaled = (float(value) - min_val) / (max_val - min_val)
            return min(1.0, max(0.0, scaled))

        if feature_name not in self.feature_stats:
            return min(1.0, max(0.0, float(value)))

        stats = self.feature_stats[feature_name]
        min_val = stats["min"]
        max_val = stats["max"]

        if max_val == min_val:
            return 0.5

        scaled = (float(value) - min_val) / (max_val - min_val)
        return min(1.0, max(0.0, scaled))

    # ---------------- Feature scaling recompute helpers ----------------
    def _get_scaling_method_for_dim(self, feature_name: str) -> str:
        try:
            override = self.feature_scaling_method_per_dim.get(feature_name)
        except Exception:
            override = None
        return override or self.feature_scaling_method

    def _maybe_recompute_feature_stats(self, feature_name: str) -> None:
        updates = int(self._feature_updates_since_recompute.get(feature_name, 0))
        buf = self._feature_value_buffer.get(feature_name)
        if not buf:
            return
        if len(buf) < self.feature_stats_min_samples:
            return
        if updates >= self.feature_stats_recompute_interval:
            # Save previous cache snapshot for drift detection (selected keys)
            prev = self._feature_stats_cache.get(feature_name, {}).copy()
            self._recompute_feature_stats(feature_name)
            self._feature_updates_since_recompute[feature_name] = 0
            # If rebin is enabled, detect drift and set flag
            if bool(getattr(self.config, "feature_map_rebin_enabled", False)):
                try:
                    drift_threshold = float(getattr(self.config, "feature_map_rebin_drift_threshold", 0.1))
                except Exception:
                    drift_threshold = 0.1
                curr = self._feature_stats_cache.get(feature_name, {})
                def _rel_change(old: float, new: float) -> float:
                    try:
                        denom = max(1e-9, abs(old))
                        return abs(new - old) / denom
                    except Exception:
                        return 0.0
                # Use min/max and robust quantiles when available
                keys = ["min", "max", "q_low", "q_high"]
                for k in keys:
                    if k in prev and k in curr:
                        if _rel_change(float(prev[k]), float(curr[k])) >= drift_threshold:
                            self._rebin_due_to_drift = True
                            break

    def _recompute_feature_stats(self, feature_name: str) -> None:
        buf = self._feature_value_buffer.get(feature_name)
        if not buf:
            return
        values = list(buf)
        if not values:
            return
        sorted_vals = sorted(values)
        n = len(sorted_vals)
        def _quantile(q: float) -> float:
            if n == 1:
                return sorted_vals[0]
            q = min(1.0, max(0.0, q))
            pos = q * (n - 1)
            lo = int(math.floor(pos))
            hi = int(math.ceil(pos))
            if lo == hi:
                return sorted_vals[lo]
            frac = pos - lo
            return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac

        min_v = sorted_vals[0]
        max_v = sorted_vals[-1]
        low_q = _quantile(self.feature_stats_robust_low_q)
        high_q = _quantile(self.feature_stats_robust_high_q)
        mean_v = sum(values) / n
        # numerically-stable two-pass variance (n could be up to 5000)
        var = sum((x - mean_v) * (x - mean_v) for x in values) / n
        std_v = math.sqrt(var) if var > 0 else 0.0

        self._feature_stats_cache[feature_name] = {
            "min": float(min_v),
            "max": float(max_v),
            "q_low": float(low_q),
            "q_high": float(high_q),
            "mean": float(mean_v),
            "std": float(std_v),
        }

    def log_island_status(self) -> None:
        """Log current status of all islands"""
        stats = self.get_island_stats()
        logger.info("Island Status:")
        for stat in stats:
            current_marker = " *" if stat["is_current"] else "  "
            island_idx = stat["island"]
            island_best_id = (
                self.island_best_programs[island_idx]
                if island_idx < len(self.island_best_programs)
                else None
            )
            best_indicator = f" (best: {island_best_id})" if island_best_id else ""
            logger.info(
                f"{current_marker} Island {stat['island']}: {stat['population_size']} programs, "
                f"best={stat['best_score']:.4f}, avg={stat['average_score']:.4f}, "
                f"diversity={stat['diversity']:.2f}, gen={stat['generation']}{best_indicator}"
            )

    def _rebin_feature_map(self, quiet: bool = True) -> None:
        """Rebuild feature_map and island bests according to current scaling stats.

        This is equivalent to the load-time repair but available at runtime.
        Does not mutate islands membership, only remaps feature_map occupancy.
        """
        # Rebuild feature_map
        rebuilt_feature_map: Dict[str, str] = {}
        for prog in self.programs.values():
            try:
                coords = self._calculate_feature_coords(prog, update_stats=False)
            except Exception:
                continue
            key = self._feature_coords_to_key(coords)
            if key not in rebuilt_feature_map:
                rebuilt_feature_map[key] = prog.id
            else:
                existing_id = rebuilt_feature_map[key]
                if existing_id in self.programs:
                    if self._is_better(prog, self.programs[existing_id]):
                        rebuilt_feature_map[key] = prog.id
                else:
                    rebuilt_feature_map[key] = prog.id
        self.feature_map = rebuilt_feature_map

        # Recalculate island best programs coherently using _is_better()
        new_island_bests: List[Optional[str]] = [None] * len(self.islands)
        for i, island in enumerate(self.islands):
            best_id: Optional[str] = None
            for pid in island:
                if pid not in self.programs:
                    continue
                if best_id is None:
                    best_id = pid
                else:
                    if self._is_better(self.programs[pid], self.programs[best_id]):
                        best_id = pid
            new_island_bests[i] = best_id
        self.island_best_programs = new_island_bests

        if not quiet:
            logger.info("Feature map rebin completed. Occupied cells: %d", len(self.feature_map))

    # ---------------- Git helpers ----------------

    def _get_diff_from_root(self, commit_hash: str):
        """Return (prompt_diff, hash_diff) between root_commit and *commit_hash*.

        root_commit is user-defined starting point of evolution.

        Uses `git diff` under the hood and falls back to empty strings if diff fails.
        """
        root = getattr(self.config, "root_commit", "HEAD")
        repo_path = getattr(self.config, "git_repo_path", ".")

        try:
            raw_diff = diff_between(repo_path, root, commit_hash)
        except Exception as exc:
            logger.warning(f"git diff failed: {exc}")
            return "", ""

        return clean_diff(raw_diff)

    # ---------------- Convenience APIs ----------------

    def snapshot(self, path: str, iteration: int = 0) -> None:
        """Write a one-off snapshot to path regardless of in-memory mode.

        This is a thin wrapper over save(path=..., iteration=...).
        """
        try:
            self.save(path=path, iteration=iteration)
            logger.info(f"Snapshot saved to {path} (iteration={iteration})")
        except Exception as e:
            logger.warning(f"Snapshot failed for {path}: {e}")
