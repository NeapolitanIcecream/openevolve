from __future__ import annotations

from typing import Dict, List, Optional

from openevolve.config import DatabaseConfig
from openevolve.database.models import Program
from openevolve.utils.metrics_utils import safe_numeric_average


class FeatureMap:
    """MAP-Elites grid that decides occupancy and replacement by fitness.

    This component knows how to place a program into a cell defined by
    feature coordinates (which are calculated by the coordinator using
    FeatureStatsManager). It does not update stats by itself.
    """

    def __init__(self, config: DatabaseConfig):
        self.config = config
        self.feature_map: Dict[str, str] = {}

    def coords_to_key(self, coords: List[int]) -> str:
        return "-".join(str(c) for c in coords)

    def is_better(self, program1: Program, program2: Program) -> bool:
        # If no metrics, use newest
        if not program1.metrics and not program2.metrics:
            return program1.timestamp > program2.timestamp

        if program1.metrics and not program2.metrics:
            return True
        if not program1.metrics and program2.metrics:
            return False

        if "combined_score" in program1.metrics and "combined_score" in program2.metrics:
            return program1.metrics["combined_score"] > program2.metrics["combined_score"]

        avg1 = safe_numeric_average(program1.metrics)
        avg2 = safe_numeric_average(program2.metrics)
        return avg1 > avg2

    def place(self, coords: List[int], program: Program, programs_lookup: Dict[str, Program]) -> Optional[str]:
        """Place program to cell; return replaced program id if any."""
        key = self.coords_to_key(coords)
        if key not in self.feature_map:
            self.feature_map[key] = program.id
            return None
        existing_id = self.feature_map[key]
        existing = programs_lookup.get(existing_id)
        if existing is None or self.is_better(program, existing):
            self.feature_map[key] = program.id
            return existing_id
        return None

    def rebuild(self, programs: List[Program], coords_fn) -> None:
        rebuilt: Dict[str, str] = {}
        # Build a lookup to avoid O(n^2) scans when comparing replacements
        by_id: Dict[str, Program] = {p.id: p for p in programs}
        for prog in programs:
            try:
                coords = coords_fn(prog)
            except Exception:
                continue
            key = self.coords_to_key(coords)
            if key not in rebuilt:
                rebuilt[key] = prog.id
            else:
                existing_id = rebuilt[key]
                existing_prog = by_id.get(existing_id)
                if existing_prog is None or self.is_better(prog, existing_prog):
                    rebuilt[key] = prog.id
        self.feature_map = rebuilt


