from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from mpc_warp.backends.mujoco_warp_backend import MujocoWarpBackend
from mpc_warp.core.costs import quadratic_cost
from mpc_warp.envs.task_envs import TASK_FACTORIES, make_task_env, state_norm
from mpc_warp.solvers.mppi import MPPIConfig, MPPISolver


@dataclass(frozen=True)
class ParityResult:
    name: str
    status: str
    initial_norm: float | None
    final_norm: float | None
    target_norm: float | None
    solved: bool
    expected_final_norm: float | None = None
    matches_expected_output: bool | None = None
    message: str | None = None


def load_hydrax_manifest(path: str | Path) -> list[dict]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"HydraX manifest not found at {p}. Provide a local exported manifest from hydrax models/tests."
        )
    data = json.loads(p.read_text())
    if not isinstance(data, list):
        raise ValueError("HydraX manifest must be a list of task definitions")
    required = {"name", "max_steps", "solved_threshold", "action_dim"}
    optional = {"expected_final_norm", "output_tolerance"}
    for i, item in enumerate(data):
        missing = required - set(item.keys())
        unknown = set(item.keys()) - (required | optional)
        if missing:
            raise ValueError(f"Task index {i} missing keys: {sorted(missing)}")
        if unknown:
            raise ValueError(f"Task index {i} has unknown keys: {sorted(unknown)}")
    return data


def discover_hydrax_tasks(hydrax_root: str | Path) -> list[dict]:
    models_dir = Path(hydrax_root) / "hydrax" / "models"
    if not models_dir.exists():
        raise FileNotFoundError(f"HydraX models dir not found at {models_dir}")

    discovered = []
    for py in sorted(models_dir.glob("*.py")):
        if py.name == "__init__.py":
            continue
        name = py.stem
        cfg = {
            "name": name,
            "max_steps": 300,
            "solved_threshold": 0.31,
            "action_dim": 1,
        }
        if name in TASK_FACTORIES:
            cfg["action_dim"] = make_task_env(name).act_dim
        discovered.append(cfg)
    return discovered


def run_task_with_mpc(name: str, max_steps: int, solved_threshold: float, action_dim: int) -> ParityResult:
    if name not in TASK_FACTORIES:
        return ParityResult(
            name=name,
            status="unsupported",
            initial_norm=None,
            final_norm=None,
            target_norm=solved_threshold,
            solved=False,
            message="No local task adapter",
        )

    try:
        backend = MujocoWarpBackend(lambda: make_task_env(name))
        x = backend.reset(seed=0)
        init = state_norm(x)
        solver = MPPISolver(
            backend,
            MPPIConfig(horizon=12, num_samples=64, noise_sigma=0.3, temperature=0.8, action_dim=action_dim),
            goal=[0.0] * len(x),
            cost_fn=quadratic_cost,
        )
        solver.reset_seed(0)
        for _ in range(max_steps):
            x = backend.step(solver.command(x)).state
        final = state_norm(x)
        solved = final <= solved_threshold
        return ParityResult(
            name=name,
            status="solved" if solved else "mismatch",
            initial_norm=init,
            final_norm=final,
            target_norm=solved_threshold,
            solved=solved,
        )
    except Exception as exc:
        return ParityResult(
            name=name,
            status="error",
            initial_norm=None,
            final_norm=None,
            target_norm=solved_threshold,
            solved=False,
            message=str(exc),
        )


def run_manifest_parity(manifest_path: str | Path) -> list[ParityResult]:
    specs = load_hydrax_manifest(manifest_path)
    results: list[ParityResult] = []
    for spec in specs:
        base = run_task_with_mpc(
            str(spec["name"]),
            int(spec["max_steps"]),
            float(spec["solved_threshold"]),
            int(spec["action_dim"]),
        )
        expected = spec.get("expected_final_norm")
        if expected is None or base.final_norm is None:
            results.append(base)
            continue
        tol = float(spec.get("output_tolerance", 1e-6))
        match = abs(base.final_norm - float(expected)) <= tol
        status = base.status
        if base.status == "solved" and not match:
            status = "mismatch"
        results.append(
            ParityResult(
                name=base.name,
                status=status,
                initial_norm=base.initial_norm,
                final_norm=base.final_norm,
                target_norm=base.target_norm,
                solved=base.solved and match,
                expected_final_norm=float(expected),
                matches_expected_output=match,
                message=base.message,
            )
        )
    return results
