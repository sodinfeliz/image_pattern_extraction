import subprocess
import os
import sys


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
