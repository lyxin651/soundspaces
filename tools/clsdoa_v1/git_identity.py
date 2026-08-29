"""Canonical generation identity; production plans may not accept a fake SHA."""

import subprocess
from pathlib import Path


class GitIdentityError(RuntimeError):
    pass


def current_clean_head(repo_root):
    repo_root = str(Path(repo_root))
    sha = subprocess.check_output(["git", "-C", repo_root, "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", repo_root, "status", "--porcelain"], text=True)
    if dirty.strip():
        raise GitIdentityError("canonical generation requires a clean git worktree")
    return sha
