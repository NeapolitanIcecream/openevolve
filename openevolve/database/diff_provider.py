from __future__ import annotations

from typing import Tuple

from openevolve.config import DatabaseConfig
from openevolve.utils.diff_utils import clean_diff
from openevolve.utils.git_utils import diff_between


class GitDiffProvider:
    """Provide cleaned diffs between root and a commit hash."""

    def __init__(self, config: DatabaseConfig):
        self.config = config

    def get_diff_from_root(self, commit_hash: str) -> Tuple[str, str]:
        root = getattr(self.config, "root_commit", "HEAD")
        repo_path = getattr(self.config, "git_repo_path", ".")
        try:
            raw_diff = diff_between(repo_path, root, commit_hash)
        except Exception:
            return "", ""
        return clean_diff(raw_diff)


