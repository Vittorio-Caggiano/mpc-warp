# mpc-warp
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1XRCW9FQON9Eb92qo3w1UtGy631qCdtKs?usp=sharing)

MPC toolkit using [mujoco_warp](https://github.com/google-deepmind/mujoco_warp) for batched parallel rollouts. All MPPI samples are stepped simultaneously in a single `mujoco_warp.step` kernel call — on GPU when CUDA is available, on CPU Warp kernels otherwise.

## Setup

```bash
uv sync
uv run pytest -q
```

## MuJoCo tasks

Model files are sourced from [vincekurtz/hydrax](https://github.com/vincekurtz/hydrax).

| Task | Description | Actuators |
|------|-------------|-----------|
| `pendulum` | Inverted pendulum swingup | 1 |
| `cart_pole` | Cart-pole swingup | 1 |
| `double_cart_pole` | Double pendulum on a cart | 1 |
| `particle` | Planar point mass chases a target | 2 |
| `walker` | Planar biped walking | 6 |
| `crane` | Luffing crane payload positioning | 3 |
| `humanoid_standup` | Unitree G1 standup (29-DOF, mesh assets) | 29 |

Run headless:

```bash
uv run python examples/run_mujoco_task.py pendulum
uv run python examples/run_mujoco_task.py walker --steps 200
```

Run with interactive viewer — use `mjpython` (bundled with `mujoco`) so the Cocoa event loop works on macOS:

```bash
uv run mjpython examples/run_mujoco_task.py pendulum --render
uv run mjpython examples/run_mujoco_task.py cart_pole --render
uv run mjpython examples/run_mujoco_task.py double_cart_pole --render
uv run mjpython examples/run_mujoco_task.py particle --render --steps 1000
uv run mjpython examples/run_mujoco_task.py walker --render --steps 1000
uv run mjpython examples/run_mujoco_task.py crane --render --steps 1000
uv run mjpython examples/run_mujoco_task.py humanoid_standup --render --steps 1000
```

`--render` automatically halves the sample count to keep the loop near real-time. Override with `--num-samples N`:

```bash
uv run mjpython examples/run_mujoco_task.py walker --render --num-samples 16
```

Close the viewer window (or press `Esc`) to stop early.

## Synthetic convergence tasks

Lightweight pure-Python linear-dynamics environments for fast MPC convergence tests (no mujoco_warp):

```bash
uv run python examples/run_all_envs.py
```

Tasks: `inverted_pendulum`, `ant`, `humanoid`. A task is solved when the terminal state norm drops below its registry threshold.

