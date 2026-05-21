# mpc-warp

MPC toolkit using [mujoco_warp](https://github.com/google-deepmind/mujoco_warp) for batched parallel rollouts. All MPPI samples are stepped simultaneously in a single `mujoco_warp.step` kernel call — on GPU when CUDA is available, on CPU Warp kernels otherwise.

<img width="1200" alt="mpc_warp_banner" src="https://github.com/user-attachments/assets/68757627-1f3e-4439-90c3-8054c3f7c3e9" />

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
| `g1_velocity` | Unitree G1 command-conditioned velocity tracking | 29 |
| `g1_mocap` | Unitree G1 motion-capture reference tracking | 29 |
| `go1_walking` | Unitree GO1 quadruped walking | 12 |
| `go1_trot` | GO1 trotting with physics-generated reference trajectory | 12 |

## Viewer

The default viewer is **mjviser** — a browser-based 3-D renderer with live MPPI panels (cost history, ESS, action horizon). No `mjpython` required.

```bash
uv run python examples/run_mujoco_task.py pendulum --render
uv run python examples/run_mujoco_task.py walker --render --steps 1000
uv run python examples/run_mujoco_task.py g1_velocity --render --steps 1000
uv run python examples/run_mujoco_task.py g1_mocap --render --steps 1000
uv run python examples/run_mujoco_task.py go1_walking --render --steps 1000
```

Open `http://localhost:8080` in a browser. The 3-D scene, cost chart, ESS bar, and action horizon plot update in real time.

`--render` automatically halves the sample count to keep the loop near real-time. Override with `--num-samples N`:

```bash
uv run python examples/run_mujoco_task.py walker --render --num-samples 16
```

**Ghost overlay** — for `g1_mocap`, a semi-transparent orange ghost is rendered at the current reference configuration alongside the live robot, matching the colour of the reference body-position dots.

### Motion-capture tracking (`g1_mocap`)

Reference data comes from the [LocoMuJoCo dataset](https://huggingface.co/datasets/robfiras/loco-mujoco-datasets) on HuggingFace and is downloaded automatically on first run (requires `huggingface_hub`, included in the project dependencies):

```bash
uv run python examples/run_mujoco_task.py g1_mocap --render
```

The default clip is `Lafan1/mocap/UnitreeG1/walk1_subject1.npz`. Pass a different clip via the task constructor:

```python
from mpc_warp.tasks import G1MocapTracking
task = G1MocapTracking(reference_filename="Lafan1/mocap/UnitreeG1/run1_subject1.npz")
```

Available clips are any `.npz` file under `Lafan1/mocap/UnitreeG1/` in the dataset repository.

### Native desktop viewer

For the native MuJoCo viewer (requires `mjpython` on macOS for the Cocoa event loop):

```bash
uv run mjpython examples/run_mujoco_task.py pendulum --viewer mujoco --render
uv run mjpython examples/run_mujoco_task.py walker --viewer mujoco --render --steps 1000
```

Both viewers simultaneously:

```bash
uv run mjpython examples/run_mujoco_task.py walker --viewer both --render
```

### Headless (no viewer)

```bash
uv run python examples/run_mujoco_task.py pendulum
uv run python examples/run_mujoco_task.py walker --steps 200
```

## Synthetic convergence tasks

Lightweight pure-Python linear-dynamics environments for fast MPC convergence tests (no mujoco_warp):

```bash
uv run python examples/run_all_envs.py
```

Tasks: `inverted_pendulum`, `ant`, `humanoid`. A task is solved when the terminal state norm drops below its registry threshold.


## Tutorials

Interactive walkthroughs for MPPI, mjviser, and MuJoCo tasks. Run locally with Jupyter or open in Colab (T4 GPU).

| Tutorial | Notebook | Colab |
|----------|----------|-------|
| MPPI + mjviser visualization | [`examples/tutorial_mppi_mjviser.ipynb`](examples/tutorial_mppi_mjviser.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1XRCW9FQON9Eb92qo3w1UtGy631qCdtKs?usp=sharing) |
| G1 motion-capture tracking | [`examples/tutorial_g1_mocap.ipynb`](examples/tutorial_g1_mocap.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1b-ONmaP3YNS8uTZCb4XAcrXKa7rufXTI?usp=sharing) |
