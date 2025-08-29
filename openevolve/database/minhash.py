from __future__ import annotations

import hashlib
import time
from typing import Dict, List, Optional, Tuple, Callable
import random

from openevolve.config import DatabaseConfig
from openevolve.database.models import Program
from openevolve.utils.diff_utils import minhash_signature, minhash_similarity


class DiversityService:
    """MinHash-based diversity and reference set management.

    Encapsulates signature generation, diversity cache, and reference set
    maintenance. Stateless with respect to repositories; only keeps in-memory
    caches derived from programs.
    """

    def __init__(self, config: DatabaseConfig, rng: Optional[random.Random] = None):
        self.config = config
        # Use injected RNG (seeded by ProgramDatabase) for determinism; fallback to local RNG
        self._rng: random.Random = rng if rng is not None else random.Random(getattr(config, "random_seed", None))

        # Diversity caching infrastructure (use stable string key)
        self.diversity_cache: Dict[str, Dict[str, float]] = {}
        self.diversity_cache_size: int = getattr(config, "diversity_cache_size", 1000)
        self.diversity_reference_set: List[List[int]] = []  # Reference signatures
        self.diversity_reference_size: int = getattr(config, "diversity_reference_size", 20)
        # Reference set maintenance state
        self.diversity_reference_program_ids: List[str] = []
        self._divref_adds_since_build: int = 0
        self._divref_last_built_at_time: float = time.time()
        self._divref_sig_len: int = int(getattr(config, "minhash_num_perm", 64))

    # ---------- Signature helpers ----------
    def ensure_signature(self, program: Program) -> None:
        if not program.minhash_signature and (program.hash_diff or program.prompt_diff):
            text_for_sig = program.hash_diff or program.prompt_diff or ""
            program.minhash_signature = minhash_signature(
                text_for_sig,
                num_perm=getattr(self.config, "minhash_num_perm", 64),
                shingle_len=getattr(self.config, "minhash_shingle_len", 5),
            )

    def normalize_signatures_on_load(self, programs: List[Program]) -> int:
        updated = 0
        try:
            target_num_perm = int(getattr(self.config, "minhash_num_perm", 64))
            shingle_len = int(getattr(self.config, "minhash_shingle_len", 5))
        except Exception:
            target_num_perm = 64
            shingle_len = 5
        for prog in programs:
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
                        pass
                else:
                    prog.minhash_signature = []
        return updated

    # ---------- Diversity / Reference set ----------
    def get_diversity(self, program: Program) -> float:
        text_for_key = program.hash_diff or program.prompt_diff or ""
        if not text_for_key:
            return 0.5
        try:
            code_key = hashlib.sha1(text_for_key.encode("utf-8")).hexdigest()
        except Exception:
            code_key = ""

        if code_key in self.diversity_cache:
            return self.diversity_cache[code_key]["value"]  # type: ignore[index]

        self.ensure_signature(program)
        need_build = (
            not self.diversity_reference_set
            or len(self.diversity_reference_set) < self.diversity_reference_size
        )
        try:
            sig_len = len(program.minhash_signature) if program.minhash_signature else int(getattr(self.config, "minhash_num_perm", 64))
        except Exception:
            sig_len = int(getattr(self.config, "minhash_num_perm", 64))
        # Do not clear the reference set here. Building requires full program list
        # which is coordinated by the caller (ProgramDatabase) before diversity is used.
        # Keeping the current set preserves previously accumulated references.
        # If a rebuild is truly required, callers should invoke refresh_reference_set_full(programs).
        if need_build or sig_len != int(getattr(self, "_divref_sig_len", sig_len)):
            pass

        diversity_scores = []
        for ref_sig in self.diversity_reference_set:
            diversity_scores.append(1.0 - minhash_similarity(program.minhash_signature, ref_sig))
        diversity = sum(diversity_scores) / max(1, len(diversity_scores)) if diversity_scores else 0.0

        if code_key:
            self._cache_diversity_value(code_key, diversity)
        return diversity

    def consider_candidate_for_reference_set(self, program: Program) -> None:
        self.ensure_signature(program)
        sig = program.minhash_signature
        if not sig:
            return
        if not self.diversity_reference_set or len(self.diversity_reference_set) < self.diversity_reference_size:
            self.diversity_reference_set.append(sig)
            self.diversity_reference_program_ids.append(program.id)
            self._divref_sig_len = len(sig)
            if len(self.diversity_reference_set) == self.diversity_reference_size:
                self._divref_last_built_at_time = time.time()
            return
        if len(sig) != self._divref_sig_len:
            self.refresh_reference_set_full([])
            return
        if program.id in self.diversity_reference_program_ids:
            return
        cand_min_sim = 1.0
        for ref_sig in self.diversity_reference_set:
            try:
                s = minhash_similarity(sig, ref_sig)
            except Exception:
                s = 1.0
            if s < cand_min_sim:
                cand_min_sim = s
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
            if nn_sim > most_redundant_nn_sim:
                most_redundant_nn_sim = nn_sim
                most_redundant_idx = i
        margin = float(getattr(self.config, "diversity_reference_online_margin", 0.05))
        if cand_min_sim + margin < most_redundant_nn_sim and 0 <= most_redundant_idx < len(self.diversity_reference_set):
            self.diversity_reference_set[most_redundant_idx] = sig
            if most_redundant_idx < len(self.diversity_reference_program_ids):
                self.diversity_reference_program_ids[most_redundant_idx] = program.id
            else:
                self.diversity_reference_program_ids.append(program.id)

    def prune_reference_set(self, program_exists: Callable[[str], bool]) -> None:
        if not self.diversity_reference_set:
            return
        new_sigs: List[List[int]] = []
        new_ids: List[str] = []
        for idx, sig in enumerate(self.diversity_reference_set):
            pid = self.diversity_reference_program_ids[idx] if idx < len(self.diversity_reference_program_ids) else None
            if pid is None or not program_exists(pid):
                continue
            if not isinstance(sig, list) or (self._divref_sig_len and len(sig) != self._divref_sig_len):
                continue
            new_sigs.append(sig)
            new_ids.append(pid)
        self.diversity_reference_set = new_sigs
        self.diversity_reference_program_ids = new_ids

    def should_refresh_reference_set(self) -> bool:
        now = time.time()
        if bool(getattr(self.config, "diversity_reference_refresh_by_inserts_enabled", True)):
            try:
                threshold = int(getattr(self.config, "diversity_reference_refresh_adds", 40))
            except Exception:
                threshold = 40
            if self._divref_adds_since_build >= max(1, threshold):
                return True
        if bool(getattr(self.config, "diversity_reference_refresh_by_time_enabled", False)):
            try:
                seconds = float(getattr(self.config, "diversity_reference_refresh_seconds", 300.0))
            except Exception:
                seconds = 300.0
            if now - float(getattr(self, "_divref_last_built_at_time", 0.0)) >= seconds:
                return True
        return False

    def maybe_refresh_reference_set(self, programs: List[Program]) -> None:
        if self.should_refresh_reference_set():
            self.refresh_reference_set_full(programs)

    def refresh_reference_set_full(self, programs: List[Program]) -> None:
        if not programs:
            self.diversity_reference_set = []
            self.diversity_reference_program_ids = []
            self._divref_adds_since_build = 0
            self._divref_last_built_at_time = time.time()
            return

        for p in programs:
            if not p.minhash_signature:
                p.minhash_signature = minhash_signature(
                    p.hash_diff or p.prompt_diff or "",
                    num_perm=getattr(self.config, "minhash_num_perm", 64),
                    shingle_len=getattr(self.config, "minhash_shingle_len", 5),
                )

        if len(programs) <= self.diversity_reference_size:
            self.diversity_reference_set = [p.minhash_signature for p in programs]
            self.diversity_reference_program_ids = [p.id for p in programs]
        else:
            remaining = programs.copy()
            # Choose first index deterministically using injected RNG
            first_idx = self._rng.randint(0, len(remaining) - 1)
            selected: List[Program] = [remaining.pop(first_idx)]
            while len(selected) < self.diversity_reference_size and remaining:
                best_idx = -1
                best_min_div = -1.0
                for i, cand in enumerate(remaining):
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

        self._divref_sig_len = len(self.diversity_reference_set[0]) if self.diversity_reference_set else int(getattr(self.config, "minhash_num_perm", 64))
        self._divref_adds_since_build = 0
        self._divref_last_built_at_time = time.time()
        self.invalidate_diversity_cache()

    def update_reference_set(self, programs: List[Program]) -> None:
        # kept for parity with original; here we just recompute if small set
        self.refresh_reference_set_full(programs)

    def invalidate_diversity_cache(self) -> None:
        self.diversity_cache.clear()

    def _cache_diversity_value(self, code_key: str, diversity: float) -> None:
        if len(self.diversity_cache) >= self.diversity_cache_size:
            oldest_key = sorted(self.diversity_cache.items(), key=lambda x: x[1]["timestamp"])[0][0]
            del self.diversity_cache[oldest_key]
        self.diversity_cache[code_key] = {"value": diversity, "timestamp": time.time()}

    # ---------- Near-duplicate utilities ----------
    def is_near_duplicate(self, sig: List[int], against: List[List[int]], threshold: Optional[float] = None) -> bool:
        if not sig:
            return False
        thr = float(getattr(self.config, "dedup_near_similarity_threshold", 0.98)) if threshold is None else float(threshold)
        for other in against:
            if not other:
                continue
            if minhash_similarity(sig, other) >= thr:
                return True
        return False


