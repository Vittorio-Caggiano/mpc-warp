import json
from pathlib import Path

import pytest

from mpc_warp.envs.hydrax_parity import load_hydrax_manifest, run_manifest_parity


def test_manifest_missing_file_errors():
    with pytest.raises(FileNotFoundError):
        load_hydrax_manifest("/tmp/does-not-exist-hydrax-manifest.json")


def test_manifest_parity_runner(tmp_path: Path):
    manifest = [
        {"name": "inverted_pendulum", "max_steps": 250, "solved_threshold": 0.06, "action_dim": 1},
        {"name": "ant", "max_steps": 300, "solved_threshold": 0.30, "action_dim": 4},
        {"name": "humanoid", "max_steps": 300, "solved_threshold": 0.31, "action_dim": 6},
    ]
    p = tmp_path / "hydrax_manifest.json"
    p.write_text(json.dumps(manifest))
    results = run_manifest_parity(p)
    assert len(results) == 3
    assert all(r.solved for r in results)
