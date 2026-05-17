# mpc-warp

Minimal MPC repository with a `mujoco_warp`-style backend interface and a pure-Python MPPI controller.

## Verified MPC task solves

The repository includes MuJoCo-style control tasks and verifies MPC solves for:

- `inverted_pendulum`
- `ant`
- `humanoid`

A task is considered solved when the terminal normalized state norm is below a task-specific threshold in the registry.

## Run

```bash
PYTHONPATH=src pytest -q
PYTHONPATH=src python examples/run_all_envs.py
```


## HydraX parity workflow

Because network access to GitHub may be restricted in some environments, this repo supports parity checks from a **local HydraX-exported manifest JSON** instead of hard-coding external data fetches.

Run:

```bash
PYTHONPATH=src python examples/run_hydrax_parity.py path/to/hydrax_manifest.json
```

Manifest schema (list of objects): `name`, `max_steps`, `solved_threshold`, `action_dim`.
When convergence fails, the script exits non-zero and prints per-task convergence error.
