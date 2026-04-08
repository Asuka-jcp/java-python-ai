from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path

REQUIRED_MODULES = ("fastapi", "uvicorn", "httpx", "pydantic")


def _missing_modules() -> list[str]:
    missing: list[str] = []
    for module in REQUIRED_MODULES:
        try:
            importlib.import_module(module)
        except ModuleNotFoundError:
            missing.append(module)
    return missing


def _install_requirements() -> None:
    requirements = Path(__file__).with_name("requirements.txt")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", str(requirements)])


def main() -> None:
    missing = _missing_modules()
    if missing:
        print(f"[bootstrap] missing modules: {', '.join(missing)}")
        _install_requirements()
    subprocess.check_call([sys.executable, "-m", "uvicorn", "main:app", "--reload", "--port", "8000"])


if __name__ == "__main__":
    main()
