"""
Lightweight Git interaction helpers for commit-based evolution.

All functions are pure helpers without hidden cwd changes. They always take
an explicit repository path and use "git -C <repo_path> ..." under the hood.
"""

from __future__ import annotations

import subprocess
from typing import List, Optional


def _run_git(repo_path: str, args: List[str]) -> subprocess.CompletedProcess:
    """Run a git command inside repo_path and return the completed process.

    Does not raise on non-zero return code; callers decide how to handle.
    """
    return subprocess.run(
        ["git", "-C", repo_path, *args], capture_output=True, text=True, check=False
    )


def ensure_repo(repo_path: str) -> None:
    """Raise if repo_path is not a valid git repository."""
    proc = _run_git(repo_path, ["rev-parse", "--is-inside-work-tree"])
    if proc.returncode != 0 or proc.stdout.strip() != "true":
        raise RuntimeError(f"Path is not a git repository: {repo_path}")


def ref_exists(repo_path: str, ref: str) -> bool:
    """Return True if ref resolves to a commit in repo."""
    proc = _run_git(repo_path, ["cat-file", "-e", f"{ref}^{{commit}}"])
    return proc.returncode == 0


def get_head(repo_path: str) -> str:
    """Return current HEAD commit hash."""
    proc = _run_git(repo_path, ["rev-parse", "HEAD"])
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr or proc.stdout)
    return proc.stdout.strip()


def diff_between(repo_path: str, base_ref: str, target_ref: str) -> str:
    """Return raw git diff between base_ref and target_ref.

    Includes binary changes and uses no color for stable downstream parsing.
    """
    proc = _run_git(
        repo_path,
        [
            "diff",
            "--binary",
            "--no-color",
            base_ref,
            target_ref,
            "--",
        ],
    )
    if proc.returncode != 0 and not proc.stdout:
        # Return empty string on failure to keep callers resilient
        return ""
    return proc.stdout or ""


def diff_worktree(repo_path: str, base_ref: str) -> str:
    """Return raw git diff between base_ref and the working tree (uncommitted changes)."""
    proc = _run_git(
        repo_path,
        [
            "diff",
            "--binary",
            "--no-color",
            base_ref,
            "--",
        ],
    )
    if proc.returncode != 0 and not proc.stdout:
        return ""
    return proc.stdout or ""


def list_changed_files(repo_path: str, base_ref: str, target_ref: str) -> List[str]:
    """Return list of files changed between base_ref and target_ref."""
    proc = _run_git(repo_path, ["diff", "--name-only", base_ref, target_ref, "--"])
    if proc.returncode != 0:
        return []
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


def show_file_at(repo_path: str, ref: str, path: str) -> str:
    """Return file content at ref for given path. Empty string on error."""
    proc = _run_git(repo_path, ["show", f"{ref}:{path}"])
    if proc.returncode != 0:
        return ""
    return proc.stdout


def create_commit_from_worktree(
    repo_path: str, message: str, add_patterns: Optional[List[str]] = None
) -> str:
    """Create a commit from the current working tree.

    - Runs `git add` on provided patterns or `-A` if none are provided
    - Creates a commit and returns the new commit hash (empty commit allowed)
    """
    # Stage changes
    if add_patterns:
        for pattern in add_patterns:
            _run_git(repo_path, ["add", pattern])
    else:
        _run_git(repo_path, ["add", "-A"])  # stage all

    # Commit (allow empty to ensure a consistent return value)
    proc_commit = _run_git(repo_path, ["commit", "--allow-empty", "-m", message])
    if proc_commit.returncode != 0:
        # Even if commit fails, try to return HEAD to keep callers moving
        return get_head(repo_path)

    # Return new HEAD
    return get_head(repo_path)


