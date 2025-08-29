from __future__ import annotations

import uuid
from typing import Dict, List, Optional, Set, Tuple

from openevolve.config import DatabaseConfig
from openevolve.database.models import Program
from openevolve.database.minhash import DiversityService
from openevolve.database.grid import FeatureMap
from openevolve.utils.diff_utils import minhash_similarity
from openevolve.utils.metrics_utils import safe_numeric_average


class IslandManager:
    """Manage island memberships, generations, and migrations."""

    def __init__(self, config: DatabaseConfig):
        self.config = config
        self.islands: List[Set[str]] = [set() for _ in range(config.num_islands)]
        self.current_island: int = 0
        self.island_generations: List[int] = [0] * config.num_islands
        self.last_migration_generation: int = 0
        self.migration_interval: int = getattr(config, "migration_interval", 10)
        self.migration_rate: float = getattr(config, "migration_rate", 0.1)

    # ---- basics ----
    def set_current_island(self, island_idx: int) -> None:
        self.current_island = island_idx % len(self.islands)

    def next_island(self) -> int:
        self.current_island = (self.current_island + 1) % len(self.islands)
        return self.current_island

    def increment_generation(self, island_idx: Optional[int] = None) -> None:
        idx = island_idx if island_idx is not None else self.current_island
        self.island_generations[idx] += 1

    def should_migrate(self) -> bool:
        max_generation = max(self.island_generations)
        return (max_generation - self.last_migration_generation) >= self.migration_interval

    def add_to_island(self, program_id: str, island_idx: int) -> None:
        idx = island_idx % len(self.islands)
        self.islands[idx].add(program_id)

    def remove_from_islands(self, program_id: str) -> None:
        for island in self.islands:
            island.discard(program_id)

    # ---- migration ----
    def migrate_programs(self, programs_lookup: Dict[str, Program], fmap: FeatureMap, diversity: DiversityService) -> List[Tuple[Program, int]]:
        if len(self.islands) < 2:
            return []
        migrants: List[Tuple[Program, int]] = []
        for i, island in enumerate(self.islands):
            if len(island) == 0:
                continue
            island_programs = [programs_lookup[pid] for pid in island if pid in programs_lookup]
            if not island_programs:
                continue
            island_programs.sort(
                key=lambda p: p.metrics.get("combined_score", safe_numeric_average(p.metrics)),
                reverse=True,
            )
            desired = max(1, int(len(island_programs) * self.migration_rate))
            target_islands = [(i + 1) % len(self.islands), (i - 1) % len(self.islands)]
            near_enabled = bool(getattr(self.config, "dedup_near_enabled", False))
            try:
                near_threshold = float(getattr(self.config, "dedup_near_similarity_threshold", 0.98))
            except Exception:
                near_threshold = 0.98
            topK = max(1, int(getattr(self.config, "migration_diversity_topk", 20)))
            for target_island in target_islands:
                selected = 0
                target_top = self._get_top_programs_in_island(programs_lookup, target_island, topK)
                for candidate in island_programs:
                    if selected >= desired:
                        break
                    migrant_copy = Program(
                        id=str(uuid.uuid4()),
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
                    # exact dedup left to coordinator; here only near-dup optional
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
                    # Collect migrant copy for coordinator to insert
                    migrants.append((migrant_copy, target_island))
                    selected += 1
        self.last_migration_generation = max(self.island_generations)
        return migrants

    def _get_top_programs_in_island(self, programs_lookup: Dict[str, Program], island_idx: int, n: int) -> List[Program]:
        island = self.islands[island_idx]
        candidates = [programs_lookup[pid] for pid in island if pid in programs_lookup]
        from openevolve.utils.metrics_utils import safe_numeric_average

        candidates.sort(key=lambda p: p.metrics.get("combined_score", safe_numeric_average(p.metrics)), reverse=True)
        return candidates[:n]


