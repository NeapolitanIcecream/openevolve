import os
import sys
from pathlib import Path

from openevolve.cleanup import main as cleanup_main


def _make_evo_branches(repo, count: int = 3, prefix: str = "oe/it_") -> list[str]:
    heads = []
    for i in range(count):
        repo.write_file(f"src/file_{i}.txt", f"content {i}\n")
        repo.add_all()
        head = repo.commit(f"feat: add file {i}")
        br = f"{prefix}{i}"
        repo.create_branch(br, head)
        heads.append(br)
    return heads


def test_dry_run_all_keeps_everything_and_outputs_preview(git_repo):
    repo = git_repo
    _make_evo_branches(repo, count=2)

    # run in dry-run (default) with --all
    code = cleanup_main([
        "--repo",
        str(repo.path),
        "--all",
    ])
    assert code == 0

    # branches should still exist (dry run)
    branches = repo.list_branches()
    assert any(b.startswith("oe/it_") for b in branches)


def test_keep_with_tag_dry_run_shows_actions_without_changes(git_repo, capsys):
    repo = git_repo
    _make_evo_branches(repo, count=2)
    kept_commit = repo.rev_parse("HEAD")[:8]  # short acceptable for prefix match in code

    code = cleanup_main([
        "--repo",
        str(repo.path),
        "--keep",
        kept_commit,
        "--tag",
        # no value -> default tag name
    ])
    assert code == 0

    out = capsys.readouterr().out
    assert "[DRY-RUN]" in out
    assert "Would create/update tag" in out

    # nothing should be deleted in dry run
    branches = repo.list_branches()
    assert any(b.startswith("oe/it_") for b in branches)


def test_yes_deletes_branches_and_removes_output_dir(git_repo, tmp_path):
    repo = git_repo
    _make_evo_branches(repo, count=2)

    # create output dir
    out_dir = repo.path / "openevolve_output"
    (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    assert out_dir.exists()

    # run with --yes and remove-output
    code = cleanup_main([
        "--repo",
        str(repo.path),
        "--all",
        "--remove-output",
        "--yes",
    ])
    assert code == 0

    # branches with prefix should be gone; current branch remains
    branches = repo.list_branches()
    assert all(not b.startswith("oe/it_") for b in branches)

    # output directory removed
    assert not out_dir.exists()


def test_keep_creates_keep_branch_and_tag_when_no_matching_branch(git_repo):
    repo = git_repo
    _make_evo_branches(repo, count=2)

    # create an extra commit that no oe/it_ branch points to
    repo.write_file("extra.txt", "extra\n")
    repo.add_all()
    kept_full_sha = repo.commit("chore: extra commit")
    short_sha = kept_full_sha[:8]

    tag_name = "openevolve/test-selected"

    code = cleanup_main([
        "--repo",
        str(repo.path),
        "--keep",
        kept_full_sha,
        "--tag",
        tag_name,
        "--yes",
    ])
    assert code == 0

    branches = repo.list_branches()
    # keep branch should be created
    assert f"oe/keep/{short_sha}" in branches
    # all oe/it_ branches removed
    assert all(not b.startswith("oe/it_") for b in branches)

    # tag exists and points to commit (best-effort check tag exists)
    proc = repo.git("tag", "--list", tag_name)
    assert proc.returncode == 0
    assert tag_name in proc.stdout.split()


def test_keep_preserves_existing_branch_and_creates_tag_without_keep_branch(git_repo):
    repo = git_repo
    _make_evo_branches(repo, count=3)

    # choose the head of the last evolution branch
    kept_full_sha = repo.rev_parse("oe/it_2")
    tag_name = "openevolve/test-selected-2"

    code = cleanup_main([
        "--repo",
        str(repo.path),
        "--keep",
        kept_full_sha,
        "--tag",
        tag_name,
        "--yes",
    ])
    assert code == 0

    branches = repo.list_branches()
    # branch with head at kept commit should remain
    assert "oe/it_2" in branches
    # other evolution branches should be removed
    assert "oe/it_0" not in branches and "oe/it_1" not in branches
    # no keep branch created since an evo branch already points to kept commit
    assert not any(b.startswith("oe/keep/") for b in branches)

    # tag exists
    proc = repo.git("tag", "--list", tag_name)
    assert proc.returncode == 0
    assert tag_name in proc.stdout.split()


