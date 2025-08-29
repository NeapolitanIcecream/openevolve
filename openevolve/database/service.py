from __future__ import annotations

import logging
import os
import math
import random
import time
import uuid
from typing import Any, Deque, Dict, List, Optional, Set, Tuple, cast

from openevolve.config import DatabaseConfig
from openevolve.database.models import Program
from openevolve.database.repository import ProgramRepository
from openevolve.database.scaler import FeatureStatsManager
from openevolve.database.minhash import DiversityService
from openevolve.database.grid import FeatureMap
from openevolve.database.islands import IslandManager
from openevolve.database.elite import EliteArchive
from openevolve.database.dedup import DedupService
from openevolve.database.diff_provider import GitDiffProvider
from openevolve.utils.metrics_utils import safe_numeric_average
from openevolve.database.sampler import Sampler

logger = logging.getLogger(__name__)


class ProgramDatabase:
    """Coordinator that composes sub-services to keep API compatibility."""

    def __init__(self, config: DatabaseConfig):
        self.config = config

        # Core services/state holders
        self.repo = ProgramRepository(config)
        self.scaler = FeatureStatsManager(config)
        # diversity depends on RNG; construct after RNG initialization
        self.feature_map = FeatureMap(config)
        self._islands = IslandManager(config)
        self.archive = EliteArchive(config)
        self.dedup = DedupService()
        self.diff = GitDiffProvider(config)

        # Derived configs
        if isinstance(config.feature_bins, int):
            self.feature_bins = max(
                config.feature_bins,
                int(pow(config.archive_size, 1 / len(config.feature_dimensions)) + 0.99),
            )
        else:
            self.feature_bins = 10
        if hasattr(config, "feature_bins") and isinstance(config.feature_bins, dict):
            self.feature_bins_per_dim = config.feature_bins
        else:
            self.feature_bins_per_dim = {dim: self.feature_bins for dim in config.feature_dimensions}

        # Runtime bookkeeping mirroring original behavior
        self.last_iteration: int = 0
        self._total_adds: int = 0
        self._rebin_due_to_drift: bool = False
        self._last_rebin_at_adds: int = 0
        self._last_rebin_at_time: float = time.time()

        # RNG
        if config.random_seed is not None:
            self.rng = random.Random(config.random_seed)
        else:
            self.rng = random.Random()

        # Diversity service depends on RNG (for deterministic reference set)
        self.diversity = DiversityService(config, rng=self.rng)

        # Sampler
        self._sampler = Sampler(
            config=self.config,
            rng=self.rng,
            repo=self.repo,
            islands=self._islands,
            archive=self.archive,
            feature_map=self.feature_map,
            feature_dimensions=self.config.feature_dimensions,
            feature_bins_per_dim=self.feature_bins_per_dim,
            feature_bins=self.feature_bins,
            coords_fn=lambda p: self._calculate_feature_coords(p, update_stats=False),
            add_fn=lambda prog, island_idx: self.add(prog, target_island=island_idx),
        )

        # Load from disk if persistence enabled
        if self.repo.persistence_enabled and getattr(self.config, "db_path", None) and self.config.db_path and os.path.exists(self.config.db_path):  # type: ignore[name-defined]
            self.load(self.config.db_path)

    # --- DB-like API ---
    # Backward-compat properties expected by callers
    @property
    def islands(self):
        return self._islands.islands

    @property
    def current_island(self) -> int:
        return self._islands.current_island

    @property
    def best_program_id(self) -> Optional[str]:
        return self.archive.best_program_id

    @best_program_id.setter
    def best_program_id(self, value: Optional[str]) -> None:
        self.archive.best_program_id = value

    @property
    def programs(self) -> Dict[str, Program]:
        return self.repo.programs

    def add(self, program: Program, iteration: Optional[int] = None, target_island: Optional[int] = None) -> str:
        if iteration is not None:
            program.iteration_found = iteration
            self.last_iteration = max(self.last_iteration, iteration)

        # Possibly compute diffs -> signatures
        if not program.hash_diff and program.commit_hash:
            try:
                prompt_diff, hash_diff = self.diff.get_diff_from_root(program.commit_hash)
                if not program.prompt_diff:
                    program.prompt_diff = prompt_diff
                program.hash_diff = hash_diff
            except Exception:
                pass

        if program.hash_diff and not program.minhash_signature:
            self.diversity.ensure_signature(program)

        # Exact dedup skip
        if not bool(program.metadata.get("cloned_for_empty_island", False)):
            if self.dedup.is_duplicate(program):
                existing_id = self.dedup.find_program_id(program)
                if existing_id:
                    return existing_id
                # Fallback: linear scan to find canonical program id and update mapping
                try:
                    key = self.dedup.equivalence_key(program)
                except Exception:
                    key = None
                if key:
                    try:
                        for _p in self.repo.programs.values():
                            try:
                                if self.dedup.equivalence_key(_p) == key:
                                    # update canonical mapping for future queries
                                    self.dedup.key_to_program_id[key] = _p.id
                                    return _p.id
                            except Exception:
                                continue
                    except Exception:
                        pass
                # As a last resort, block insertion and return the provided id (legacy parity)
                return program.id

        # Insert
        self.repo.put(program)

        # Feature coords (write-path updates stats)
        coords = self._calculate_feature_coords(program, update_stats=True)
        replaced_id = self.feature_map.place(coords, program, self.repo.programs)
        if replaced_id and replaced_id in self.archive.archive:
            self.archive.archive.discard(replaced_id)
            self.archive.archive.add(program.id)

        # Island membership
        island_idx = (target_island if target_island is not None else self._islands.current_island) % len(self._islands.islands)
        self._islands.add_to_island(program.id, island_idx)
        program.metadata["island"] = island_idx

        # Derived fields
        if program.hash_diff:
            program.complexity = float(len(program.hash_diff))
        elif program.prompt_diff:
            program.complexity = float(len(program.prompt_diff))
        else:
            program.complexity = 0.0
        try:
            program.diversity = float(self.diversity.get_diversity(program))
        except Exception:
            program.diversity = 0.0

        # Archive & bests
        self.archive.update_archive(program, self.repo.programs)
        self._enforce_population_limit(exclude_program_id=program.id)
        self.archive.update_best_program(program, self.repo.programs, self.feature_map)
        self.archive.update_island_best(program, island_idx, self.repo.programs, self.feature_map)

        # Register dedup key
        self.dedup.register(program)

        # Persistence
        if self.repo.persistence_enabled and self.config.db_path:
            self.repo._save_program(program, self.config.db_path)

        # Maintenance
        self._total_adds += 1
        if self.scaler.freeze_after_warmup_if_needed(self._total_adds):
            logger.info("Feature statistics updates frozen after warmup")

        if bool(getattr(self.config, "feature_map_rebin_enabled", False)):
            now = time.time()
            adds_interval = int(getattr(self.config, "feature_map_rebin_interval_adds", 0) or 0)
            secs_interval = float(getattr(self.config, "feature_map_rebin_interval_seconds", 0.0) or 0.0)
            should_by_adds = adds_interval > 0 and (self._total_adds - self._last_rebin_at_adds) >= adds_interval
            should_by_time = secs_interval > 0.0 and (now - self._last_rebin_at_time) >= secs_interval
            # Added: rebin trigger when scaler detected drift
            scaler_drift = bool(getattr(self.scaler, "_rebin_due_to_drift", False))
            if should_by_adds or should_by_time or self._rebin_due_to_drift or scaler_drift:
                try:
                    self._rebin_feature_map(quiet=bool(getattr(self.config, "feature_map_rebin_quiet", True)))
                except Exception as e:
                    logger.debug(f"Feature map rebin failed: {e}")
                self._last_rebin_at_adds = self._total_adds
                self._last_rebin_at_time = now
                self._rebin_due_to_drift = False
                if scaler_drift:
                    try:
                        self.scaler._rebin_due_to_drift = False
                    except Exception:
                        pass

        try:
            self.diversity.consider_candidate_for_reference_set(program)
            self.diversity._divref_adds_since_build += 1
            self.diversity.maybe_refresh_reference_set(list(self.repo.programs.values()))
        except Exception as e:
            logger.debug(f"Diversity reference maintenance error: {e}")

        return program.id

    def get(self, program_id: str) -> Optional[Program]:
        return self.repo.get(program_id)

    def sample(self) -> Tuple[Program, List[Program]]:
        return self._sampler.sample()

    def get_best_program(self, metric: Optional[str] = None) -> Optional[Program]:
        if not self.repo.programs:
            return None
        if metric is None and self.archive.best_program_id:
            if self.archive.best_program_id in self.repo.programs:
                return self.repo.programs[self.archive.best_program_id]
            else:
                self.archive.best_program_id = None
        candidates = list(self.repo.programs.values())
        if metric:
            sorted_programs = sorted(
                [p for p in candidates if metric in p.metrics], key=lambda p: p.metrics[metric], reverse=True
            )
        else:
            sorted_programs = sorted(candidates, key=self._fitness_value, reverse=True)
        return sorted_programs[0] if sorted_programs else None

    def get_top_programs(self, n: int = 10, metric: Optional[str] = None, island_idx: Optional[int] = None) -> List[Program]:
        if island_idx is not None and (island_idx < 0 or island_idx >= len(self._islands.islands)):
            raise IndexError(f"Island index {island_idx} is out of range (0-{len(self._islands.islands)-1})")
        if not self.repo.programs:
            return []
        if island_idx is not None:
            island_programs = [self.repo.programs[pid] for pid in self._islands.islands[island_idx] if pid in self.repo.programs]
            candidates = island_programs
        else:
            candidates = list(self.repo.programs.values())
        if not candidates:
            return []
        if metric:
            sorted_programs = sorted(
                [p for p in candidates if metric in p.metrics], key=lambda p: p.metrics[metric], reverse=True
            )
        else:
            sorted_programs = sorted(candidates, key=self._fitness_value, reverse=True)
        return sorted_programs[:n]

    def save(self, path: Optional[str] = None, iteration: int = 0) -> None:  # type: ignore[name-defined]
        # In-memory mode: only save when explicit path provided
        if not self.repo.persistence_enabled and path is None:
            logger.info("In-memory mode: skipping save (no target path provided)")
            return
        save_path = path or self.config.db_path
        if not save_path:
            logger.warning("No database path specified, skipping save")
            return
        import os
        os.makedirs(save_path, exist_ok=True)
        self.repo.save_all(save_path, iteration)
        metadata = {
            "feature_map": self.feature_map.feature_map,
            "islands": [list(island) for island in self._islands.islands],
            "archive": list(self.archive.archive),
            "best_program_id": self.archive.best_program_id,
            "island_best_programs": self.archive.island_best_programs,
            "last_iteration": iteration or self.last_iteration,
            "current_island": self._islands.current_island,
            "island_generations": self._islands.island_generations,
            "last_migration_generation": self._islands.last_migration_generation,
            # Persist all seen keys to maintain blocking semantics across restarts
            "seen_equiv_keys": self.dedup.export_seen_keys(),
        }
        import json
        with open(os.path.join(save_path, "metadata.json"), "w") as f:
            json.dump(metadata, f)
        logger.info(f"Saved database with {len(self.repo.programs)} programs to {save_path}")

    def load(self, path: str) -> None:
        import os
        if not os.path.exists(path):
            logger.warning(f"Database path {path} does not exist, skipping load")
            return
        metadata_path = os.path.join(path, "metadata.json")
        saved_islands = []
        if os.path.exists(metadata_path):
            import json
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            self.feature_map.feature_map = metadata.get("feature_map", {})
            saved_islands = metadata.get("islands", [])
            self.archive.archive = set(metadata.get("archive", []))
            self.archive.best_program_id = metadata.get("best_program_id")
            self.archive.island_best_programs = metadata.get("island_best_programs", [None] * len(saved_islands))
            self.last_iteration = metadata.get("last_iteration", 0)
            self._islands.current_island = metadata.get("current_island", 0)
            self._islands.island_generations = metadata.get("island_generations", [0] * len(saved_islands))
            self._islands.last_migration_generation = metadata.get("last_migration_generation", 0)
            keys = metadata.get("seen_equiv_keys", [])
            if isinstance(keys, list):
                # Preload seen keys to immediately block duplicates; canonical map rebuilt later
                self.dedup.preload_seen_keys(keys)
        self.repo.load_all(path)
        # reconstruct islands
        self._reconstruct_islands(saved_islands)
        # rebuild dedup mapping from programs
        self.dedup.rebuild_from_programs(self.repo.programs)
        # rebuild feature stats & minhash norms & diversity reference
        if self.repo.programs:
            self.scaler.rebuild_feature_stats_from_programs(list(self.repo.programs.values()), self.config.feature_dimensions)
            try:
                self.diversity.normalize_signatures_on_load(list(self.repo.programs.values()))
            except Exception:
                pass
            try:
                self.diversity.refresh_reference_set_full(list(self.repo.programs.values()))
            except Exception:
                pass
        # For consistency with legacy behavior: always rebuild feature_map and island bests
        # after loading, independent of config.toggle
        try:
            self._rebin_feature_map(quiet=True)
        except Exception:
            pass

    def _reconstruct_islands(self, saved_islands: List[List[str]]) -> None:
        # Initialize empty islands
        num_islands = max(len(saved_islands), self.config.num_islands)
        self._islands.islands = [set() for _ in range(num_islands)]
        missing_programs: List[Tuple[int, str]] = []
        restored_programs = 0
        for island_idx, program_ids in enumerate(saved_islands):
            if island_idx >= len(self._islands.islands):
                continue
            for program_id in program_ids:
                if program_id in self.repo.programs:
                    self._islands.islands[island_idx].add(program_id)
                    self.repo.programs[program_id].metadata["island"] = island_idx
                    restored_programs += 1
                else:
                    missing_programs.append((island_idx, program_id))
        # Clean up feature_map & archive referencing missing programs
        feature_keys_to_remove = [key for key, pid in self.feature_map.feature_map.items() if pid not in self.repo.programs]
        for key in feature_keys_to_remove:
            del self.feature_map.feature_map[key]
        self.archive.archive = {pid for pid in self.archive.archive if pid in self.repo.programs}
        # Clean up island bests
        self.archive.cleanup_stale_island_bests(self.repo.programs, self._islands.islands)
        # If we have programs but no island assignments, distribute them
        if self.repo.programs and sum(len(island) for island in self._islands.islands) == 0:
            program_ids = list(self.repo.programs.keys())
            for i, program_id in enumerate(program_ids):
                idx = i % len(self._islands.islands)
                self._islands.islands[idx].add(program_id)
                self.repo.programs[program_id].metadata["island"] = idx
        # Rebin feature map once if enabled
        if bool(getattr(self.config, "feature_map_rebin_enabled", False)):
            try:
                self._rebin_feature_map(quiet=True)
            except Exception:
                pass

    # ---- helpers copied/adapted ----
    def _fitness_value(self, program: Program) -> float:
        try:
            if program.metrics:
                if "combined_score" in program.metrics:
                    return float(program.metrics.get("combined_score", float("-inf")))
                return float(safe_numeric_average(program.metrics))
        except Exception:
            pass
        return float("-inf")

    def _calculate_feature_coords(self, program: Program, update_stats: bool = False) -> List[int]:
        coords: List[int] = []
        for dim in self.config.feature_dimensions:
            if dim == "complexity":
                if program.hash_diff:
                    complexity = len(program.hash_diff)
                elif program.prompt_diff:
                    complexity = len(program.prompt_diff)
                else:
                    complexity = 0
                bin_idx = self.scaler.calculate_complexity_bin(complexity, update_stats, int(self.feature_bins_per_dim.get("complexity", self.feature_bins)))
                coords.append(bin_idx)
            elif dim == "diversity":
                if len(self.repo.programs) < 2:
                    bin_idx = 0
                else:
                    # Ensure reference set is available and conforms to current configuration
                    need_build = (
                        not self.diversity.diversity_reference_set
                        or len(self.diversity.diversity_reference_set) < self.diversity.diversity_reference_size
                    )
                    try:
                        target_sig_len = int(getattr(self.config, "minhash_num_perm", 64))
                    except Exception:
                        target_sig_len = 64
                    # Use program signature length if available to compare with current ref-set signature length
                    curr_ref_sig_len = int(getattr(self.diversity, "_divref_sig_len", target_sig_len))
                    if need_build or curr_ref_sig_len != target_sig_len:
                        try:
                            self.diversity.refresh_reference_set_full(list(self.repo.programs.values()))
                        except Exception:
                            pass

                    diversity_val = self.diversity.get_diversity(program)
                    has_text = bool(program.hash_diff or program.prompt_diff)
                    bin_idx = self.scaler.calculate_diversity_bin(
                        diversity_val,
                        update_stats and has_text,
                        int(self.feature_bins_per_dim.get("diversity", self.feature_bins)),
                    )
                coords.append(bin_idx)
            elif dim == "score":
                if not program.metrics:
                    bin_idx = 0
                else:
                    avg_score = safe_numeric_average(program.metrics)
                    if update_stats:
                        self.scaler.update("score", avg_score)
                    scaled_value = self.scaler.scale("score", avg_score)
                    num_bins = int(self.feature_bins_per_dim.get("score", self.feature_bins))
                    bin_idx = int(scaled_value * num_bins)
                    bin_idx = max(0, min(num_bins - 1, bin_idx))
                coords.append(bin_idx)
            elif dim in program.metrics:
                score = program.metrics[dim]
                if update_stats:
                    self.scaler.update(dim, score)
                scaled_value = self.scaler.scale(dim, score)
                num_bins = int(self.feature_bins_per_dim.get(dim, self.feature_bins))
                bin_idx = int(scaled_value * num_bins)
                bin_idx = max(0, min(num_bins - 1, bin_idx))
                coords.append(bin_idx)
            else:
                raise ValueError(
                    f"Feature dimension '{dim}' specified in config but not found in program metrics. "
                    f"Available metrics: {list(program.metrics.keys())}. "
                    f"Either remove '{dim}' from feature_dimensions or ensure your evaluator returns it."
                )
        return coords

    def _enforce_population_limit(self, exclude_program_id: Optional[str] = None) -> None:
        if len(self.repo.programs) <= self.config.population_size:
            return
        num_to_remove = len(self.repo.programs) - self.config.population_size
        all_programs = list(self.repo.programs.values())
        sorted_programs = sorted(all_programs, key=self._fitness_value)
        programs_to_remove: List[Program] = []
        protected_ids = {self.archive.best_program_id, exclude_program_id} - {None}  # type: ignore[arg-type]
        for program in sorted_programs:
            if len(programs_to_remove) >= num_to_remove:
                break
            if program.id not in protected_ids:
                programs_to_remove.append(program)
        if len(programs_to_remove) < num_to_remove:
            remaining_programs = [p for p in sorted_programs if p not in programs_to_remove and p.id not in protected_ids]
            programs_to_remove.extend(remaining_programs[: num_to_remove - len(programs_to_remove)])
        for program in programs_to_remove:
            program_id = program.id
            if program_id in self.repo.programs:
                del self.repo.programs[program_id]
            keys_to_remove = [key for key, pid in self.feature_map.feature_map.items() if pid == program_id]
            for key in keys_to_remove:
                del self.feature_map.feature_map[key]
            self._islands.remove_from_islands(program_id)
            self.archive.archive.discard(program_id)
        self.archive.cleanup_stale_island_bests(self.repo.programs, self._islands.islands)

        # If a large fraction of population was removed in one operation, rebuild stats and optionally diversity refs
        try:
            original_size = len(all_programs)
            removed_count = len(programs_to_remove)
            if original_size > 0:
                removed_ratio = removed_count / float(original_size)
                threshold = float(getattr(self.scaler, "feature_stats_rebuild_remove_ratio", 0.1))
                if removed_ratio >= threshold:
                    # Rebuild feature stats from current programs
                    self.scaler.rebuild_feature_stats_from_programs(
                        list(self.repo.programs.values()), self.config.feature_dimensions
                    )
                    # Optionally trigger diversity reference set rebuild on large removals
                    if bool(getattr(self.config, "diversity_reference_refresh_by_population_enabled", False)):
                        pop_ratio = float(getattr(self.config, "diversity_reference_rebuild_remove_ratio", 0.1))
                        if removed_ratio >= pop_ratio:
                            try:
                                self.diversity.refresh_reference_set_full(list(self.repo.programs.values()))
                            except Exception:
                                pass
        except Exception:
            pass

    def _rebin_feature_map(self, quiet: bool = True) -> None:
        programs = list(self.repo.programs.values())
        self.feature_map.rebuild(programs, lambda p: self._calculate_feature_coords(p, update_stats=False))
        # recalc island bests
        new_island_bests: List[Optional[str]] = [None] * len(self._islands.islands)
        for i, island in enumerate(self._islands.islands):
            best_id: Optional[str] = None
            for pid in island:
                if pid not in self.repo.programs:
                    continue
                if best_id is None:
                    best_id = pid
                else:
                    if self.feature_map.is_better(self.repo.programs[pid], self.repo.programs[best_id]):
                        best_id = pid
            new_island_bests[i] = best_id
        self.archive.island_best_programs = new_island_bests
        if not quiet:
            logger.info("Feature map rebin completed. Occupied cells: %d", len(self.feature_map.feature_map))

    # ---- island-facing API compatibility ----
    def set_current_island(self, island_idx: int) -> None:
        self._islands.set_current_island(island_idx)

    def next_island(self) -> int:
        return self._islands.next_island()

    def increment_island_generation(self, island_idx: Optional[int] = None) -> None:
        self._islands.increment_generation(island_idx)

    def should_migrate(self) -> bool:
        return self._islands.should_migrate()

    def migrate_programs(self) -> None:
        # delegate selection logic; then insert migrants into target islands via add()
        migrants = self._islands.migrate_programs(self.repo.programs, self.feature_map, self.diversity)
        for prog, target_island in migrants:
            try:
                self.add(prog, target_island=target_island)
            except Exception as e:
                logger.debug(f"Failed to insert migrant into island {target_island}: {e}")
        logger.info("Migration completed at generation %s", max(self._islands.island_generations))
        self._validate_migration_results()

    def _validate_migration_results(self) -> None:
        seen_program_ids = set()
        for i, island in enumerate(self._islands.islands):
            for program_id in island:
                if program_id in seen_program_ids:
                    logger.error(f"Program {program_id} assigned to multiple islands")
                    continue
                seen_program_ids.add(program_id)
                if program_id not in self.repo.programs:
                    logger.warning(f"Island {i} contains nonexistent program {program_id}")
                    continue
                program = self.repo.programs[program_id]
                stored_island = cast(Optional[int], program.metadata.get("island"))
                if stored_island != i:
                    logger.warning(
                        f"Island mismatch for program {program_id}: in island {i} but metadata says {stored_island}"
                    )
        for i, best_id in enumerate(self.archive.island_best_programs):
            if best_id is not None:
                if best_id not in self.repo.programs:
                    logger.warning(f"Island {i} best program {best_id} does not exist")
                elif best_id not in self._islands.islands[i]:
                    logger.warning(f"Island {i} best program {best_id} not in island")

    # ---- exploration/exploitation sampling ----
    def _sample_parent(self) -> Program: ...  # delegated to Sampler
    def _sample_exploration_parent(self) -> Program: ...  # delegated to Sampler
    def _sample_exploitation_parent(self) -> Program: ...  # delegated to Sampler
    def _sample_random_parent(self) -> Program: ...  # delegated to Sampler
    def _sample_inspirations(self, parent: Program, n: int = 5) -> List[Program]: ...  # delegated to Sampler

    # ---- utilities exposed for logging ----
    def get_island_stats(self) -> List[dict]:
        stats: List[dict] = []
        for i, island in enumerate(self._islands.islands):
            island_programs = [self.repo.programs[pid] for pid in island if pid in self.repo.programs]
            if island_programs:
                scores = [p.metrics.get("combined_score", safe_numeric_average(p.metrics)) for p in island_programs]
                best_score = max(scores) if scores else 0.0
                avg_score = sum(scores) / len(scores) if scores else 0.0
                diversity = self._calculate_island_diversity(island_programs)
            else:
                best_score = avg_score = diversity = 0.0
            stats.append({
                "island": i,
                "population_size": len(island_programs),
                "best_score": best_score,
                "average_score": avg_score,
                "diversity": diversity,
                "generation": self._islands.island_generations[i],
                "is_current": i == self._islands.current_island,
            })
        return stats

    def log_island_status(self) -> None:
        stats = self.get_island_stats()
        logger.info("Island Status:")
        for stat in stats:
            current_marker = " *" if stat["is_current"] else "  "
            island_idx = stat["island"]
            island_best_id = self.archive.island_best_programs[island_idx] if island_idx < len(self.archive.island_best_programs) else None
            best_indicator = f" (best: {island_best_id})" if island_best_id else ""
            logger.info(
                f"{current_marker} Island {stat['island']}: {stat['population_size']} programs, "
                f"best={stat['best_score']:.4f}, avg={stat['average_score']:.4f}, "
                f"diversity={stat['diversity']:.2f}, gen={stat['generation']}{best_indicator}"
            )

    def _calculate_island_diversity(self, programs: List[Program]) -> float:
        if len(programs) < 2:
            return 0.0
        total_diversity = 0
        comparisons = 0
        sample_size = min(getattr(self.config, "island_diversity_sample_size", 5), len(programs))
        sorted_programs = sorted(programs, key=lambda p: p.id)
        sample_programs = sorted_programs[:sample_size]
        max_comparisons = getattr(self.config, "island_diversity_max_comparisons", 6)
        from openevolve.utils.diff_utils import minhash_similarity
        for i, prog1 in enumerate(sample_programs):
            for prog2 in sample_programs[i + 1 :]:
                if comparisons >= max_comparisons:
                    break
                diversity = 1.0 - minhash_similarity(prog1.minhash_signature, prog2.minhash_signature)
                total_diversity += diversity
                comparisons += 1
            if comparisons >= max_comparisons:
                break
        return total_diversity / max(1, comparisons)


