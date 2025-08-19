"""
Cleanup utility for OpenEvolve commit-based runs.

Features:
- Remove all evolution branches (by prefix, default: "oe/it_") and worktrees
- Or keep a specific commit by preserving a branch/tag reference to it
- Optional pruning of git objects and removal of output folders

Usage examples:
    openevolve-clean --repo /abs/path/to/repo --all --yes
    openevolve-clean --repo /abs/path/to/repo --keep <commit> --yes
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from typing import List, Optional, Tuple


# We use internal git helpers where available, but avoid hard dependency here
def _run_git(repo_path: str, args: List[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", repo_path, *args], capture_output=True, text=True, check=False
    )


def _ensure_repo(repo_path: str) -> None:
    proc = _run_git(repo_path, ["rev-parse", "--git-dir"])
    if proc.returncode != 0:
        raise RuntimeError(f"Not a git repository: {repo_path}")


def _current_branch(repo_path: str) -> Optional[str]:
    proc = _run_git(repo_path, ["symbolic-ref", "--quiet", "--short", "HEAD"])
    if proc.returncode != 0:
        return None
    return proc.stdout.strip()


def _short_sha(sha: str) -> str:
    return sha[:8]


@dataclass
class BranchInfo:
    name: str
    sha: str


def list_local_branches_with_prefix(repo_path: str, prefix: str) -> List[BranchInfo]:
    proc = _run_git(
        repo_path,
        [
            "for-each-ref",
            "--format=%(refname:short) %(objectname)",
            "refs/heads/",
        ],
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr or proc.stdout)
    branches: List[BranchInfo] = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            name, sha = line.split(" ", 1)
        except ValueError:
            continue
        if name.startswith(prefix):
            branches.append(BranchInfo(name=name, sha=sha.strip()))
    return branches


def commit_exists(repo_path: str, ref: str) -> bool:
    proc = _run_git(repo_path, ["cat-file", "-e", f"{ref}^{{commit}}"])
    return proc.returncode == 0


def create_keep_branch(repo_path: str, commit: str, name: Optional[str] = None) -> str:
    branch_name = name or f"oe/keep/{_short_sha(commit)}"
    proc = _run_git(repo_path, ["branch", "-f", branch_name, commit])
    if proc.returncode != 0:
        raise RuntimeError(f"Failed to create keep branch {branch_name}: {proc.stderr}")
    return branch_name


def create_tag(repo_path: str, commit: str, tag_name: str) -> None:
    proc = _run_git(repo_path, ["tag", "-f", tag_name, commit])
    if proc.returncode != 0:
        raise RuntimeError(f"Failed to create tag {tag_name}: {proc.stderr}")


def list_worktrees(repo_path: str) -> List[str]:
    proc = _run_git(repo_path, ["worktree", "list", "--porcelain"])
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr or proc.stdout)
    paths: List[str] = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if line.startswith("worktree "):
            wt_path = line.split(" ", 1)[1].strip()
            paths.append(wt_path)
    return paths


def remove_worktree(repo_path: str, worktree_path: str, *, dry_run: bool) -> Tuple[bool, str]:
    if dry_run:
        return True, f"Would remove worktree: {worktree_path}"
    proc = _run_git(repo_path, ["worktree", "remove", "-f", worktree_path])
    ok = proc.returncode == 0
    msg = proc.stdout.strip() or proc.stderr.strip()
    return ok, msg


def delete_local_branch(repo_path: str, branch: str, *, dry_run: bool) -> Tuple[bool, str]:
    if dry_run:
        return True, f"Would delete branch: {branch}"
    proc = _run_git(repo_path, ["branch", "-D", branch])
    ok = proc.returncode == 0
    msg = proc.stdout.strip() or proc.stderr.strip()
    return ok, msg


def delete_remote_branches(repo_path: str, remote: str, branches: List[str], *, dry_run: bool) -> List[Tuple[str, bool, str]]:
    results: List[Tuple[str, bool, str]] = []
    if not branches:
        return results
    # Limit to existing heads on remote
    proc = _run_git(repo_path, ["ls-remote", "--heads", remote])
    if proc.returncode != 0:
        return [("<remote-list>", False, proc.stderr or proc.stdout)]
    remote_heads = set()
    for line in proc.stdout.splitlines():
        parts = line.strip().split()
        if len(parts) == 2 and parts[1].startswith("refs/heads/"):
            remote_heads.add(parts[1].replace("refs/heads/", "", 1))
    for br in branches:
        if br not in remote_heads:
            results.append((br, True, "Skipped (not on remote)"))
            continue
        if dry_run:
            results.append((br, True, f"Would delete remote branch {remote}/{br}"))
            continue
        proc_del = _run_git(repo_path, ["push", remote, "--delete", br])
        ok = proc_del.returncode == 0
        msg = proc_del.stdout.strip() or proc_del.stderr.strip()
        results.append((br, ok, msg))
    return results


def prune_repository(repo_path: str, *, dry_run: bool) -> None:
    if dry_run:
        print("[DRY-RUN] Would run: git reflog expire --expire=now --all")
        print("[DRY-RUN] Would run: git gc --prune=now --aggressive")
        return
    _run_git(repo_path, ["reflog", "expire", "--expire=now", "--all"])  # ignore rc
    _run_git(repo_path, ["gc", "--prune=now", "--aggressive"])  # ignore rc


def remove_output_dir(repo_path: str, output_dir: Optional[str], *, dry_run: bool) -> None:
    if not output_dir:
        output_dir = os.path.join(repo_path, "openevolve_output")
    if os.path.abspath(output_dir).startswith(os.path.abspath(repo_path)) and os.path.exists(output_dir):
        if dry_run:
            print(f"[DRY-RUN] Would remove output dir: {output_dir}")
        else:
            shutil.rmtree(output_dir, ignore_errors=True)
            print(f"Removed output dir: {output_dir}")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cleanup evolution branches and commits for OpenEvolve")
    parser.add_argument("--repo", required=True, help="Path to the git repository")

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--all", action="store_true", help="Delete all evolution branches by prefix")
    mode.add_argument("--keep", metavar="COMMIT", help="Keep the specified commit; delete other evolution branches")

    parser.add_argument("--prefix", default="oe/it_", help="Evolution branch prefix to match (default: oe/it_)")

    parser.add_argument("--keep-branch", default=None, help="Branch name to create for --keep commit if none of the evolution branches point to it")
    parser.add_argument("--tag", nargs="?", const="openevolve/selected", default=None, help="Create/update a tag pointing to the kept commit (default name if no value provided)")

    # Worktrees and output
    parser.add_argument("--remove-worktrees", dest="remove_worktrees", action="store_true", help="Remove .openevolve/worktrees entries (default: true)")
    parser.add_argument("--no-remove-worktrees", dest="remove_worktrees", action="store_false", help="Do not remove worktrees")
    parser.set_defaults(remove_worktrees=True)

    parser.add_argument("--remove-output", action="store_true", help="Remove openevolve_output directory under repo")
    parser.add_argument("--output-dir", default=None, help="Custom output directory to remove (overrides default)")

    # Remote
    parser.add_argument("--remote", default=None, help="Also delete matching branches on the given remote (e.g., origin)")

    # Execution mode
    parser.add_argument("--yes", action="store_true", help="Execute (otherwise dry-run)")
    parser.add_argument("--dry-run", action="store_true", help="Force dry-run even with --yes")

    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    repo = os.path.abspath(args.repo)
    try:
        _ensure_repo(repo)
    except Exception as exc:
        print(f"Error: {exc}")
        return 1

    effective_dry_run = not args.yes or args.dry_run
    if effective_dry_run:
        print("[DRY-RUN] No changes will be made. Use --yes to apply.")

    # Gather local branches to delete
    try:
        evo_branches = list_local_branches_with_prefix(repo, args.prefix)
    except Exception as exc:
        print(f"Error listing branches: {exc}")
        return 1

    # Filter to delete set based on mode
    to_delete: List[BranchInfo] = evo_branches.copy()
    keep_commit = None
    keep_branches: List[str] = []

    if args.keep:
        keep_commit = args.keep
        if not commit_exists(repo, keep_commit):
            print(f"Error: Commit not found: {keep_commit}")
            return 1
        # Exclude branches whose HEAD exactly equals the kept commit
        keep_branches = [b.name for b in evo_branches if b.sha.startswith(keep_commit)]
        to_delete = [b for b in evo_branches if b.name not in keep_branches]

    # Worktrees under .openevolve/worktrees
    worktrees_to_remove: List[str] = []
    if args.remove_worktrees:
        try:
            wts = list_worktrees(repo)
        except Exception as exc:
            print(f"Error listing worktrees: {exc}")
            return 1
        base_wt_dir = os.path.join(repo, ".openevolve", "worktrees")
        for wt in wts:
            # Only remove worktrees managed by OpenEvolve
            if os.path.abspath(wt).startswith(os.path.abspath(base_wt_dir)):
                worktrees_to_remove.append(wt)

    # Plan summary
    print(f"Repository: {repo}")
    print(f"Mode: {'ALL' if args.all else 'KEEP'}")
    print(f"Matched evolution branches with prefix '{args.prefix}': {len(evo_branches)}")
    if args.keep:
        print(f"Kept commit: {keep_commit}")
        if keep_branches:
            print(f"Preserving branches with HEAD at kept commit: {', '.join(keep_branches)}")
        else:
            print("No evolution branches have the kept commit as HEAD; will create a keep branch if requested.")
    if worktrees_to_remove:
        print(f"Worktrees to remove: {len(worktrees_to_remove)} under .openevolve/worktrees")
    if args.remote:
        print(f"Remote cleanup enabled for: {args.remote}")
    if args.remove_output:
        print("Will remove output directory after branch cleanup")
    if not to_delete and not args.all and not args.keep:
        print("Nothing to do")

    # Prevent deleting current branch if it matches deletion set
    current_br = _current_branch(repo)
    protected: Optional[str] = None
    if current_br and any(b.name == current_br for b in to_delete):
        protected = current_br
        to_delete = [b for b in to_delete if b.name != current_br]
        print(f"Warning: Current branch '{current_br}' matches deletion prefix; it will be skipped.")

    # KEEP mode: ensure a ref points to kept commit
    created_keep_branch: Optional[str] = None
    if args.keep:
        if not keep_branches:
            # Create a dedicated keep branch if requested (or default name)
            keep_branch_name = args.keep_branch or f"oe/keep/{_short_sha(keep_commit)}"
            if effective_dry_run:
                print(f"[DRY-RUN] Would create keep branch: {keep_branch_name} -> {keep_commit}")
            else:
                try:
                    created_keep_branch = create_keep_branch(repo, keep_commit, keep_branch_name)
                    print(f"Created keep branch: {created_keep_branch} -> {keep_commit}")
                except Exception as exc:
                    print(f"Error creating keep branch: {exc}")
                    return 1
        # Optional tag
        if args.tag:
            if effective_dry_run:
                print(f"[DRY-RUN] Would create/update tag: {args.tag} -> {keep_commit}")
            else:
                try:
                    create_tag(repo, keep_commit, args.tag)
                    print(f"Created/updated tag: {args.tag} -> {keep_commit}")
                except Exception as exc:
                    print(f"Error creating tag: {exc}")
                    return 1

    # Remove worktrees first to unlock branches
    if worktrees_to_remove:
        for wt in worktrees_to_remove:
            ok, msg = remove_worktree(repo, wt, dry_run=effective_dry_run)
            prefix = "[OK]" if ok else "[ERR]"
            print(prefix, msg)

    # Delete remote branches (optional)
    if args.remote:
        remote_results = delete_remote_branches(repo, args.remote, [b.name for b in to_delete], dry_run=effective_dry_run)
        for br, ok, msg in remote_results:
            prefix = "[OK]" if ok else "[ERR]"
            print(prefix, f"remote {args.remote} {br}:", msg)

    # Delete local branches
    for b in to_delete:
        ok, msg = delete_local_branch(repo, b.name, dry_run=effective_dry_run)
        prefix = "[OK]" if ok else "[ERR]"
        print(prefix, msg or f"Deleted {b.name}")

    # Output directory cleanup
    if args.remove_output:
        remove_output_dir(repo, args.output_dir, dry_run=effective_dry_run)

    # Prune repository
    prune_repository(repo, dry_run=effective_dry_run)

    # Final summary
    if effective_dry_run:
        print("[DRY-RUN] Completed. Re-run with --yes to apply.")
    else:
        print("Cleanup completed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())


