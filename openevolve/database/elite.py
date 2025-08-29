from __future__ import annotations

from typing import Dict, List, Optional, Set

from openevolve.config import DatabaseConfig
from openevolve.database.models import Program
from openevolve.database.grid import FeatureMap
from openevolve.utils.metrics_utils import safe_numeric_average


class EliteArchive:
    """Capacity-limited archive tracking elite programs and best pointers."""

    def __init__(self, config: DatabaseConfig):
        self.config = config
        self.archive: Set[str] = set()
        self.best_program_id: Optional[str] = None
        self.island_best_programs: List[Optional[str]] = [None] * config.num_islands

    def _fitness_value(self, program: Program) -> float:
        try:
            if program.metrics:
                if "combined_score" in program.metrics:
                    return float(program.metrics.get("combined_score", float("-inf")))
                return float(safe_numeric_average(program.metrics))
        except Exception:
            pass
        return float("-inf")

    def update_archive(self, program: Program, programs_lookup: Dict[str, Program]) -> None:
        if len(self.archive) < self.config.archive_size:
            self.archive.add(program.id)
            return
        valid_archive_programs = []
        stale_ids = []
        for pid in self.archive:
            if pid in programs_lookup:
                valid_archive_programs.append(programs_lookup[pid])
            else:
                stale_ids.append(pid)
        for sid in stale_ids:
            self.archive.discard(sid)
        if len(self.archive) < self.config.archive_size:
            self.archive.add(program.id)
            return
        if valid_archive_programs:
            worst_program = min(valid_archive_programs, key=self._fitness_value)
            if self._fitness_value(program) > self._fitness_value(worst_program):
                self.archive.remove(worst_program.id)
                self.archive.add(program.id)
        else:
            self.archive.add(program.id)

    def update_best_program(self, program: Program, programs_lookup: Dict[str, Program], feature_map: FeatureMap) -> None:
        if self.best_program_id is None:
            self.best_program_id = program.id
            return
        if self.best_program_id not in programs_lookup:
            self.best_program_id = program.id
            return
        if self._fitness_value(program) > self._fitness_value(programs_lookup[self.best_program_id]):
            self.best_program_id = program.id

    def update_island_best(self, program: Program, island_idx: int, programs_lookup: Dict[str, Program], fmap: FeatureMap) -> None:
        if island_idx >= len(self.island_best_programs):
            return
        current = self.island_best_programs[island_idx]
        if current is None:
            self.island_best_programs[island_idx] = program.id
            return
        if current not in programs_lookup:
            self.island_best_programs[island_idx] = program.id
            return
        if fmap.is_better(program, programs_lookup[current]):
            self.island_best_programs[island_idx] = program.id

    def cleanup_stale_island_bests(self, programs_lookup: Dict[str, Program], islands: List[set]) -> None:
        for i, best_id in enumerate(self.island_best_programs):
            if best_id is None:
                continue
            if best_id not in programs_lookup or best_id not in islands[i]:
                self.island_best_programs[i] = None


