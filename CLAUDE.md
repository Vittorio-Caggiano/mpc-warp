# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
uv sync                                      # install dependencies
uv run pytest -q                             # run all tests
uv run pytest tests/test_solver_smoke.py -q  # single test file

# Headless (no viewer)
uv run python examples/run_mujoco_task.py pendulum

# Native 3-D desktop viewer (requires mjpython on macOS for Cocoa event loop)
uv run mjpython examples/run_mujoco_task.py cart_pole --viewer mujoco --render

# Browser-based mjviser viewer + MPPI panels (cost, ESS, actions)
uv run python examples/run_mujoco_task.py pendulum --viewer mjviser --render

# Both viewers simultaneously (native + browser panels)
uv run mjpython examples/run_mujoco_task.py walker --viewer both --render

# GO1 quadruped via mjlab model
uv run python examples/run_mujoco_task.py go1_walking --viewer mjviser --render

uv run python examples/run_all_envs.py      # synthetic convergence envs
```

## Architecture

### Primary stack (real MuJoCo physics)

1. **Tasks** (`src/mpc_warp/tasks/`) — `Task` subclasses implementing `running_cost(data, ctrl)` and `terminal_cost(data)` against `mujoco.MjData`.
   - Models in `src/mpc_warp/models/` (pendulum, cart_pole, double_cart_pole, particle, walker, crane, humanoid_standup/G1)
   - `Go1Walking` — Unitree GO1 quadruped built dynamically from `mjlab`'s robot XML + floor + torque actuators
   - `TrajectoryTask` — wraps any base task and adds a reference `(T, nq)` qpos tracking cost; call `.advance()` after each env step
   - Each task declares `trace_sites` (site names for 3-D trajectory overlay); stored as `self.trace_site_ids`

2. **MujocoTaskEnv** (`src/mpc_warp/envs/mujoco_env.py`) — wraps a `Task` into `reset/step`. Holds the live `mujoco.MjData` used by both viewers.

3. **WarpMPPISolver** (`src/mpc_warp/solvers/mppi_warp.py`) — primary MPPI solver using mujoco_warp batched rollouts.
   - Each `command(env.data)` call uploads state to N worlds, runs H steps in parallel, weights samples, updates nominal trajectory
   - Post-command diagnostics: `solver.planned_sites` `(H, n_sites, 3)`, `solver.last_cost`, `solver.cost_weights` `(N,)`
   - GPU (CUDA) when available, CPU Warp kernels otherwise

### Viewer stack

**`--viewer mujoco`** (default, requires `mjpython` on macOS):
- `mujoco.viewer.launch_passive` for 3-D physics view
- `TrajectoryViz` (`src/mpc_warp/viz/trajectory.py`) draws into `viewer.user_scn`:
  - Past trace: fading grey spheres
  - Planned trajectory: cyan spheres + lines (from `solver.planned_sites`)
  - Action bars: green/red `mjGEOM_BOX` per actuator
  - HUD text: cost + ESS as `mjGEOM_LABEL`

**`--viewer mjviser`** (browser, no `mjpython` needed):
- `mjviser.ViserMujocoScene` renders the 3-D physics in the browser
- `MppiPanel` (`src/mpc_warp/viz/mjviser_panel.py`) adds viser GUI tabs:
  - Cost history: `add_uplot` time-series chart (update via `handle.data = (x, y)`)
  - ESS: `add_progress_bar` + HTML label
  - Action bars: `add_html` with per-actuator centered bar chart
- Open `http://localhost:8080` (default viser port) in a browser

**`--viewer both`**: native viewer on main thread + viser server in background thread sharing the same `MppiPanel`.

### Synthetic stack (tests only)

- **Synthetic environments** (`src/mpc_warp/envs/task_envs.py`) — linear-dynamics stubs
- **MujocoWarpBackend** (`src/mpc_warp/backends/mujoco_warp_backend.py`) — fork-based wrapper for synthetic envs
- **MPPISolver** (`src/mpc_warp/solvers/mppi.py`) — sequential MPPI for synthetic envs
- **Registry** (`src/mpc_warp/envs/registry.py`) — `ENV_REGISTRY` iterated by tests and `run_all_envs.py`

### Key design notes

- `WarpMPPISolver.command` takes `mujoco.MjData` directly so cost functions access full computed quantities (`site_xpos`, `sensordata`)
- `mjviser.Viewer` owns its own loop; for `mjviser` mode we drive physics manually and call `scene.update_from_mjdata(data)` each step instead of using `Viewer.run()`
- `Go1Walking` builds its model at runtime via `mujoco.MjSpec` (adds floor + actuators to mjlab's bare robot XML); no scene.xml file
- `mjlab` provides robot XML paths via e.g. `mjlab.asset_zoo.robots.unitree_go1.go1_constants.GO1_XML`
