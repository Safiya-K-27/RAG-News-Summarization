"""Colab bootstrap utility: installs dependencies and runs the pipeline."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str]) -> None:
    print("$", " ".join(cmd))
    subprocess.check_call(cmd)


def install_dependencies(project_root: Path) -> None:
    req_file = project_root / "requirements-colab.txt"
    if not req_file.exists():
        raise FileNotFoundError(f"Missing {req_file}")

    _run([sys.executable, "-m", "pip", "install", "-q", "-r", str(req_file)])
    _run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])


def run_pipeline(project_root: Path) -> None:
    _run([sys.executable, str(project_root / "main.py")])


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    install_dependencies(root)
    run_pipeline(root)
