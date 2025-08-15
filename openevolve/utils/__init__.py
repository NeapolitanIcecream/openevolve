"""
Utilities module (commit-based)
"""

from openevolve.utils.async_utils import (
    TaskPool,
    gather_with_concurrency,
    retry_async,
    run_in_executor,
)
from openevolve.utils.format_utils import (
    format_metrics_safe,
    format_improvement_safe,
)
from openevolve.utils.metrics_utils import (
    safe_numeric_average,
    safe_numeric_sum,
)
from .diff_utils import clean_diff, minhash_signature, minhash_similarity
from .git_utils import (
    ensure_repo,
    ref_exists,
    get_head,
    diff_between,
    diff_worktree,
    list_changed_files,
    show_file_at,
    create_commit_from_worktree,
)
from .commit_utils import compute_commit_features, compute_patch_from_worktree

__all__ = [
    # async
    "TaskPool",
    "gather_with_concurrency",
    "retry_async",
    "run_in_executor",
    # formatting/metrics
    "format_metrics_safe",
    "format_improvement_safe",
    "safe_numeric_average",
    "safe_numeric_sum",
    # commit-level diffs/signatures
    "clean_diff",
    "minhash_signature",
    "minhash_similarity",
    # git helpers
    "ensure_repo",
    "ref_exists",
    "get_head",
    "diff_between",
    "diff_worktree",
    "list_changed_files",
    "show_file_at",
    "create_commit_from_worktree",
    # commit features
    "compute_commit_features",
    "compute_patch_from_worktree",
]
