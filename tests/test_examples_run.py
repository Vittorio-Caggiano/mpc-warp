import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_run_all_envs():
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    p = subprocess.run(
        [sys.executable, "examples/run_all_envs.py"],
        cwd=ROOT,
        env=env,
        check=False,
    )
    assert p.returncode == 0
