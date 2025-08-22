import os
import subprocess
from pathlib import Path

import pytest


class GitRepo:
    def __init__(self, path: Path):
        self.path = Path(path)

    def git(self, *args: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["git", "-C", str(self.path), *args], capture_output=True, text=True, check=False
        )

    def write_file(self, rel_path: str, content: str) -> None:
        abs_path = self.path / rel_path
        abs_path.parent.mkdir(parents=True, exist_ok=True)
        abs_path.write_text(content)

    def add_all(self) -> None:
        self.git("add", "-A")

    def commit(self, message: str) -> str:
        proc = self.git("commit", "-m", message)
        assert proc.returncode == 0, proc.stderr or proc.stdout
        return self.rev_parse("HEAD")

    def create_branch(self, name: str, ref: str = "HEAD") -> None:
        proc = self.git("branch", name, ref)
        assert proc.returncode == 0, proc.stderr or proc.stdout

    def checkout(self, ref: str) -> None:
        proc = self.git("checkout", ref)
        assert proc.returncode == 0, proc.stderr or proc.stdout

    def rev_parse(self, ref: str) -> str:
        proc = self.git("rev-parse", ref)
        assert proc.returncode == 0, proc.stderr or proc.stdout
        return proc.stdout.strip()

    def list_branches(self) -> list[str]:
        proc = self.git("branch", "--list")
        assert proc.returncode == 0, proc.stderr or proc.stdout
        # lines like '* main' or '  feature'
        lines = [ln.strip().lstrip("* ") for ln in proc.stdout.splitlines() if ln.strip()]
        return lines


@pytest.fixture()
def git_repo(tmp_path: Path) -> GitRepo:
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir(parents=True, exist_ok=True)

    # init repository
    subprocess.run(["git", "init", str(repo_dir)], check=True)

    # basic identity for commits
    subprocess.run(["git", "-C", str(repo_dir), "config", "user.email", "test@example.com"], check=True)
    subprocess.run(["git", "-C", str(repo_dir), "config", "user.name", "Test User"], check=True)

    repo = GitRepo(repo_dir)

    # initial commit
    repo.write_file("README.md", "test repo\n")
    repo.add_all()
    repo.commit("chore: initial commit")

    return repo


