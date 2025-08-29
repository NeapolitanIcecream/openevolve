from __future__ import annotations

import hashlib
from typing import Dict, Optional, List, Set

from openevolve.database.models import Program


class DedupService:
    """Exact deduplication via normalized diff hash -> canonical program id."""

    def __init__(self) -> None:
        # key -> canonical program id
        self.key_to_program_id: Dict[str, str] = {}
        # All observed equivalence keys (persisted). Used to block duplicates even
        # when canonical mapping is unavailable (e.g., program file missing).
        self.seen_keys: Set[str] = set()

    @staticmethod
    def equivalence_key(program: Program) -> Optional[str]:
        text = program.hash_diff or program.prompt_diff
        if not text:
            return None
        try:
            return hashlib.sha1(text.encode("utf-8")).hexdigest()
        except Exception:
            return None

    def is_duplicate(self, program: Program) -> bool:
        key = self.equivalence_key(program)
        return bool(key and (key in self.key_to_program_id or key in self.seen_keys))

    def register(self, program: Program) -> None:
        key = self.equivalence_key(program)
        if not key:
            return
        # Record in seen set to provide immediate blocking across restarts
        self.seen_keys.add(key)
        # Remember canonical mapping if not yet set
        if key not in self.key_to_program_id:
            self.key_to_program_id[key] = program.id

    def find_program_id(self, program: Program) -> Optional[str]:
        key = self.equivalence_key(program)
        if not key:
            return None
        return self.key_to_program_id.get(key)

    def rebuild_from_programs(self, programs: Dict[str, Program]) -> None:
        self.key_to_program_id.clear()
        # Rebuild both canonical map and seen set from existing programs
        self.seen_keys = set()
        for prog in programs.values():
            key = self.equivalence_key(prog)
            if key:
                self.seen_keys.add(key)
                if key not in self.key_to_program_id:
                    self.key_to_program_id[key] = prog.id

    # --- Persistence helpers for seen keys ---
    def preload_seen_keys(self, keys: List[str]) -> None:
        """Preload previously seen equivalence keys from metadata.

        These keys are used to block exact duplicates even if the canonical
        program JSON is not present in the repository at load time.
        """
        try:
            for k in keys:
                if isinstance(k, str) and k:
                    self.seen_keys.add(k)
        except Exception:
            pass

    def export_seen_keys(self) -> List[str]:
        return list(self.seen_keys or set(self.key_to_program_id.keys()))


