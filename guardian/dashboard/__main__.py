"""Launch the optional dashboard demo: ``python -m guardian.dashboard``."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    app_path = Path(__file__).resolve().parent / "app.py"
    cmd = [sys.executable, "-m", "streamlit", "run", str(app_path)]
    if argv:
        cmd.extend(argv)
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
