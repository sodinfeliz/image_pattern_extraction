import subprocess
import os
import shutil
import sys
from pathlib import Path


def open_directory(path: str) -> None:
    match sys.platform:
        case 'win32':
            os.startfile(path)
        case 'darwin':
            subprocess.run(['open', path])
        case 'linux':
            subprocess.run(['xdg-open', path])
        case platform_name:
            raise Exception(f'Unsupported platform: {platform_name}.')


def first_subdirectory(path: Path) -> str | None:
    for item in path.iterdir():
        if item.is_dir():
            return item.name
    return None


def recreate_directory(path: Path) -> Path:
    shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True)
    return path

