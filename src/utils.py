import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional


def open_directory(path: str) -> None:
    if sys.platform == "win32":
        os.startfile(path)
    elif sys.platform == "darwin":
        subprocess.run(["open", path])
    elif sys.platform == "linux":
        subprocess.run(["xdg-open", path])
    else:
        raise Exception(f"Unsupported platform: {sys.platform}.")


def first_subdirectory(path: Path) -> Optional[str]:
    for item in path.iterdir():
        if item.is_dir():
            return item.name
    return None


def recreate_directory(path: Path) -> Path:
    shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True)
    return path
