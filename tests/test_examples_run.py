import subprocess
import sys


def test_run_all_envs():
    p = subprocess.run([sys.executable, "examples/run_all_envs.py"], check=False)
    assert p.returncode == 0
