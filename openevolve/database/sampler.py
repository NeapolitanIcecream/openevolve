from __future__ import annotations

import random
import uuid
from typing import Callable, Dict, List, Optional, Tuple, cast

from openevolve.config import DatabaseConfig
from openevolve.database.models import Program
from openevolve.database.repository import ProgramRepository
from openevolve.database.islands import IslandManager
from openevolve.database.elite import EliteArchive
from openevolve.database.grid import FeatureMap
from openevolve.utils.metrics_utils import safe_numeric_average
from openevolve.utils.diff_utils import minhash_similarity


class Sampler:
    """Encapsulates exploration/exploitation sampling and inspiration selection.

    Delegates all state access to injected services and uses a provided
    coordinate function to avoid owning feature scaling logic.
    """

    def __init__(
        self,
        config: DatabaseConfig,
        rng: random.Random,
        repo: ProgramRepository,
        islands: IslandManager,
        archive: EliteArchive,
        feature_map: FeatureMap,
        feature_dimensions: List[str],
        feature_bins_per_dim: Dict[str, int],
        feature_bins: int,
        coords_fn: Callable[[Program], List[int]],
        add_fn: Callable[[Program, int], str],
    ) -> None:
        self.config = config
        self.rng = rng
        self.repo = repo
        self.islands = islands
        self.archive = archive
        self.feature_map = feature_map
        self.feature_dimensions = feature_dimensions
        self.feature_bins_per_dim = feature_bins_per_dim
        self.feature_bins = feature_bins
        self.coords_fn = coords_fn
        self.add_fn = add_fn

    # ---- Public API ----
    def sample(self) -> Tuple[Program, List[Program]]:
        parent = self._sample_parent()
        inspirations = self._sample_inspirations(parent, n=getattr(self.config, "num_inspirations", 5))
        return parent, inspirations

    # ---- Internal helpers ----
    def _sample_parent(self) -> Program:
        rand_val = self.rng.random()
        if rand_val < self.config.exploration_ratio:
            return self._sample_exploration_parent()
        elif rand_val < self.config.exploration_ratio + self.config.exploitation_ratio:
            return self._sample_exploitation_parent()
        else:
            return self._sample_random_parent()

    def _sample_exploration_parent(self) -> Program:
        current_island_programs = self.islands.islands[self.islands.current_island]
        if not current_island_programs:
            source_prog: Optional[Program] = None
            if self.archive.best_program_id and self.archive.best_program_id in self.repo.programs:
                source_prog = self.repo.programs[self.archive.best_program_id]
            else:
                try:
                    source_prog = next(iter(self.repo.programs.values()))
                except StopIteration:
                    raise ValueError("No programs available to initialize empty island")
            migrant_copy = Program(
                id=str(uuid.uuid4()),
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
            new_id = self.add_fn(migrant_copy, self.islands.current_island)
            return self.repo.programs[new_id]

        # Clean stale and sample within island
        valid_programs = [pid for pid in current_island_programs if pid in self.repo.programs]
        if len(valid_programs) < len(current_island_programs):
            stale_ids = current_island_programs - set(valid_programs)
            for stale_id in stale_ids:
                self.islands.islands[self.islands.current_island].discard(stale_id)
        if not valid_programs:
            # Reinitialize island
            reinit_source_prog: Optional[Program] = None
            if self.archive.best_program_id and self.archive.best_program_id in self.repo.programs:
                reinit_source_prog = self.repo.programs[self.archive.best_program_id]
            else:
                try:
                    reinit_source_prog = next(iter(self.repo.programs.values()))
                except StopIteration:
                    raise ValueError("No programs available to reinitialize island")
            migrant_copy = Program(
                id=str(uuid.uuid4()),
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
            new_id = self.add_fn(migrant_copy, self.islands.current_island)
            return self.repo.programs[new_id]
        parent_id = self.rng.choice(valid_programs)
        return self.repo.programs[parent_id]

    def _sample_exploitation_parent(self) -> Program:
        if not self.archive.archive:
            return self._sample_exploration_parent()
        valid_archive = [pid for pid in self.archive.archive if pid in self.repo.programs]
        if len(valid_archive) < len(self.archive.archive):
            stale_ids = self.archive.archive - set(valid_archive)
            for stale_id in stale_ids:
                self.archive.archive.discard(stale_id)
        if not valid_archive:
            return self._sample_exploration_parent()
        archive_programs_in_island = [pid for pid in valid_archive if cast(Optional[int], self.repo.programs[pid].metadata.get("island")) == self.islands.current_island]
        candidate_ids = archive_programs_in_island or valid_archive
        if getattr(self.config, "dedup_near_enabled", False):
            try:
                threshold = float(getattr(self.config, "dedup_near_similarity_threshold", 0.98))
            except Exception:
                threshold = 0.98
            target_island = self.islands.current_island
            topK = max(1, int(getattr(self.config, "migration_diversity_topk", 20)))
            target_programs = self._get_top_programs(n=topK, island_idx=target_island)
            def _is_too_similar(pid: str) -> bool:
                prog = self.repo.programs[pid]
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
        return self.repo.programs[parent_id]

    def _sample_random_parent(self) -> Program:
        if not self.repo.programs:
            raise ValueError("No programs available for sampling")
        program_id = self.rng.choice(list(self.repo.programs.keys()))
        return self.repo.programs[program_id]

    def _sample_inspirations(self, parent: Program, n: int = 5) -> List[Program]:
        inspirations: List[Program] = []
        parent_island = cast(int, parent.metadata.get("island", self.islands.current_island))
        island_program_ids = list(self.islands.islands[parent_island])
        island_programs = [self.repo.programs[pid] for pid in island_program_ids if pid in self.repo.programs]
        if not island_programs:
            return []
        island_best_id = self.archive.island_best_programs[parent_island]
        if island_best_id is not None and island_best_id != parent.id and island_best_id in self.repo.programs:
            island_best = self.repo.programs[island_best_id]
            inspirations.append(island_best)
        elif island_best_id is not None and island_best_id not in self.repo.programs:
            self.archive.island_best_programs[parent_island] = None
        top_n = max(1, int(n * self.config.elite_selection_ratio))
        top_island_programs = self._get_top_programs(n=top_n, island_idx=parent_island)
        for program in top_island_programs:
            if program.id not in [p.id for p in inspirations] and program.id != parent.id:
                inspirations.append(program)
        if len(island_programs) > n and len(inspirations) < n:
            remaining_slots = n - len(inspirations)
            feature_coords = self.coords_fn(parent)
            nearby_programs: List[Program] = []
            island_feature_map: Dict[str, List[str]] = {}
            for prog_id in island_program_ids:
                if prog_id in self.repo.programs:
                    prog = self.repo.programs[prog_id]
                    prog_coords = self.coords_fn(prog)
                    cell_key = self.feature_map.coords_to_key(prog_coords)
                    bucket = island_feature_map.get(cell_key)
                    if bucket is None:
                        island_feature_map[cell_key] = [prog_id]
                    else:
                        bucket.append(prog_id)
            for _ in range(remaining_slots * 3):
                perturbed_coords: List[int] = []
                for idx, c in enumerate(feature_coords):
                    dim = self.feature_dimensions[idx] if idx < len(self.feature_dimensions) else None
                    num_bins = int(self.feature_bins_per_dim.get(dim, self.feature_bins)) if dim is not None else int(self.feature_bins)
                    perturbed_val = c + self.rng.randint(-2, 2)
                    perturbed_coords.append(max(0, min(max(0, num_bins - 1), perturbed_val)))
                cell_key = self.feature_map.coords_to_key(perturbed_coords)
                if cell_key in island_feature_map:
                    candidate_ids = [pid for pid in island_feature_map[cell_key] if (pid != parent.id and pid not in [p.id for p in inspirations] and pid not in [p.id for p in nearby_programs] and pid in self.repo.programs)]
                    if candidate_ids:
                        chosen_id = self.rng.choice(candidate_ids)
                        nearby_programs.append(self.repo.programs[chosen_id])
                        if len(nearby_programs) >= remaining_slots:
                            break
            if len(inspirations) + len(nearby_programs) < n:
                remaining = n - len(inspirations) - len(nearby_programs)
                excluded_ids = {parent.id}.union(p.id for p in inspirations).union(p.id for p in nearby_programs)
                available_island_ids = [pid for pid in island_program_ids if pid not in excluded_ids and pid in self.repo.programs]
                if available_island_ids:
                    random_ids = self.rng.sample(available_island_ids, min(remaining, len(available_island_ids)))
                    random_programs = [self.repo.programs[pid] for pid in random_ids]
                    nearby_programs.extend(random_programs)
            inspirations.extend(nearby_programs)
        return inspirations[:n]

    # Local top N implementation (unified fitness)
    def _get_top_programs(self, n: int = 10, island_idx: Optional[int] = None) -> List[Program]:
        if island_idx is not None and (island_idx < 0 or island_idx >= len(self.islands.islands)):
            return []
        if not self.repo.programs:
            return []
        if island_idx is not None:
            candidates = [self.repo.programs[pid] for pid in self.islands.islands[island_idx] if pid in self.repo.programs]
        else:
            candidates = list(self.repo.programs.values())
        candidates.sort(key=self._fitness_value, reverse=True)
        return candidates[:n]

    def _fitness_value(self, program: Program) -> float:
        try:
            if program.metrics:
                if "combined_score" in program.metrics:
                    return float(program.metrics.get("combined_score", float("-inf")))
                return float(safe_numeric_average(program.metrics))
        except Exception:
            pass
        return float("-inf")


