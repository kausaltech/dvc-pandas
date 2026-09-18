from __future__ import annotations

import subprocess
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


def git(path: Path, *args: str) -> str:
    result = subprocess.run(  # noqa: S603 - local test repositories only
        ['git', '-C', str(path), *args], check=True, capture_output=True, text=True,  # noqa: S607
    )
    return result.stdout.strip()


def configure_git(path: Path) -> None:
    git(path, 'config', 'user.email', 'test@example.invalid')
    git(path, 'config', 'user.name', 'Test')
    git(path, 'config', 'commit.gpgsign', 'false')
