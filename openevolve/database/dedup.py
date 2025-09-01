from __future__ import annotations

import hashlib
from typing import Dict, Optional, List, Set

from openevolve.database.models import Program


class DedupService:
    """Exact deduplication via normalized diff hash -> canonical program id, scoped per island.

    Each island maintains its own exact-dedup state so that identical individuals
    can exist across different islands while remaining unique within an island.
    """

    def __init__(self) -> None:
        # Per-island: key -> canonical program id
        self.key_to_program_id_by_island: Dict[int, Dict[str, str]] = {}
        # Per-island: all observed equivalence keys (persisted)
        self.seen_keys_by_island: Dict[int, Set[str]] = {}

    @staticmethod
    def equivalence_key(program: Program) -> Optional[str]:
        text = program.hash_diff or program.prompt_diff
        if not text:
            return None
        try:
            return hashlib.sha1(text.encode("utf-8")).hexdigest()
        except Exception:
            return None

    def _ensure_island_containers(self, island_idx: int) -> None:
        if island_idx not in self.key_to_program_id_by_island:
            self.key_to_program_id_by_island[island_idx] = {}
        if island_idx not in self.seen_keys_by_island:
            self.seen_keys_by_island[island_idx] = set()

    def is_duplicate(self, program: Program, island_idx: int) -> bool:
        self._ensure_island_containers(island_idx)
        key = self.equivalence_key(program)
        if not key:
            return False
        return bool(
            key in self.key_to_program_id_by_island[island_idx]
            or key in self.seen_keys_by_island[island_idx]
        )

    def register(self, program: Program, island_idx: int) -> None:
        self._ensure_island_containers(island_idx)
        key = self.equivalence_key(program)
        if not key:
            return
        self.seen_keys_by_island[island_idx].add(key)
        if key not in self.key_to_program_id_by_island[island_idx]:
            self.key_to_program_id_by_island[island_idx][key] = program.id

    def find_program_id(self, program: Program, island_idx: int) -> Optional[str]:
        self._ensure_island_containers(island_idx)
        key = self.equivalence_key(program)
        if not key:
            return None
        return self.key_to_program_id_by_island[island_idx].get(key)

    def rebuild_from_programs(self, programs: Dict[str, Program], islands: List[Set[str]]) -> None:
        self.key_to_program_id_by_island.clear()
        self.seen_keys_by_island.clear()
        # Build per-island maps strictly from memberships
        for idx, island in enumerate(islands):
            self._ensure_island_containers(idx)
            for pid in island:
                prog = programs.get(pid)
                if not prog:
                    continue
                key = self.equivalence_key(prog)
                if not key:
                    continue
                self.seen_keys_by_island[idx].add(key)
                if key not in self.key_to_program_id_by_island[idx]:
                    self.key_to_program_id_by_island[idx][key] = prog.id

    # --- Persistence helpers for seen keys (per island) ---
    def preload_seen_keys_by_island(self, keys_by_island: Dict[int, List[str]]) -> None:
        try:
            for k_island, keys in (keys_by_island or {}).items():
                # JSON may coerce dict keys to strings; accept both
                try:
                    island_idx = int(k_island)  # type: ignore[arg-type]
                except Exception:
                    continue
                self._ensure_island_containers(island_idx)
                for k in keys or []:
                    if isinstance(k, str) and k:
                        self.seen_keys_by_island[island_idx].add(k)
        except Exception:
            pass

    def export_seen_keys_by_island(self) -> Dict[int, List[str]]:
        out: Dict[int, List[str]] = {}
        for idx, keys in self.seen_keys_by_island.items():
            out[idx] = list(keys or set(self.key_to_program_id_by_island.get(idx, {}).keys()))
        return out


