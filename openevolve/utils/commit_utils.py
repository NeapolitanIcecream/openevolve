"""
Commit-level utilities built on top of git_utils and diff cleaning.

Provides one-stop APIs to compute prompt/hash diffs, MinHash signatures,
changed files, and basic complexity signals between a root ref and target ref.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from .git_utils import diff_between, list_changed_files
from .diff_utils import clean_diff, minhash_signature


def compute_commit_features(
    repo_path: str, root_ref: str, target_ref: str
) -> Dict[str, object]:
    """Compute commit features between root_ref and target_ref.

    Returns a dictionary containing:
      - prompt_diff: str
      - hash_diff: str
      - minhash_signature: List[int]
      - changed_files: List[str]
      - complexity: int (length of hash_diff)
    """
    raw = diff_between(repo_path, root_ref, target_ref)
    prompt_diff, hash_diff = clean_diff(raw)
    signature = minhash_signature(hash_diff or prompt_diff or "")
    changed = list_changed_files(repo_path, root_ref, target_ref)

    return {
        "prompt_diff": prompt_diff,
        "hash_diff": hash_diff,
        "minhash_signature": signature,
        "changed_files": changed,
        "complexity": len(hash_diff),
    }


def compute_patch_from_worktree(repo_path: str, root_ref: str) -> Tuple[str, str, List[int]]:
    """Read working tree diff vs root_ref and return (prompt_diff, hash_diff, minhash_signature)."""
    from .git_utils import diff_worktree

    raw = diff_worktree(repo_path, root_ref)
    prompt_diff, hash_diff = clean_diff(raw)
    signature = minhash_signature(hash_diff or prompt_diff or "")
    return prompt_diff, hash_diff, signature


