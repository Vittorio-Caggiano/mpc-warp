import json
from pathlib import Path

import pytest

from mpc_warp.envs.hydrax_parity import discover_hydrax_tasks, load_hydrax_manifest, run_manifest_parity


def test_manifest_missing_file_errors():
    with pytest.raises(FileNotFoundError):
        load_hydrax_manifest("/tmp/does-not-exist-hydrax-manifest.json")


def test_manifest_parity_runner_and_statuses(tmp_path: Path):
    manifest = [
        {"name": "inverted_pendulum", "max_steps": 250, "solved_threshold": 0.09, "action_dim": 1},
        {"name": "ant", "max_steps": 300, "solved_threshold": 0.30, "action_dim": 4},
        {"name": "humanoid", "max_steps": 300, "solved_threshold": 0.31, "action_dim": 6},
        {"name": "unknown_hydrax_model", "max_steps": 10, "solved_threshold": 1.0, "action_dim": 1},
    ]
    p = tmp_path / "hydrax_manifest.json"
    p.write_text(json.dumps(manifest))
    results = run_manifest_parity(p)
    assert len(results) == 4
    by = {r.name: r for r in results}
    assert by["inverted_pendulum"].status == "solved"
    assert by["ant"].status == "solved"
    assert by["humanoid"].status == "solved"
    assert by["unknown_hydrax_model"].status == "unsupported"


def test_discover_hydrax_tasks(tmp_path: Path):
    models = tmp_path / "hydrax" / "models"
    models.mkdir(parents=True)
    (models / "ant.py").write_text("# x")
    (models / "humanoid.py").write_text("# x")
    tasks = discover_hydrax_tasks(tmp_path)
    assert [t["name"] for t in tasks] == ["ant", "humanoid"]
