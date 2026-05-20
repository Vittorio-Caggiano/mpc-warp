"""Run MPPI on a real MuJoCo task using mujoco_warp batched rollouts.

Three viewer modes are supported:
  --viewer mujoco   (default) native 3-D desktop viewer + trajectory overlays
  --viewer mjviser  browser-based viser viewer + MPPI panels (cost, ESS, actions)
  --viewer both     native 3-D viewer + mjviser panels running side-by-side

Usage:
    uv run mjpython examples/run_mujoco_task.py pendulum --viewer mujoco --render
    uv run python   examples/run_mujoco_task.py pendulum --viewer mjviser --render
    uv run mjpython examples/run_mujoco_task.py walker   --viewer both   --render
    uv run python   examples/run_mujoco_task.py go1_walking --viewer mjviser --render
"""

from __future__ import annotations

import argparse
import time

import mujoco
import numpy as np

from mpc_warp.envs.mujoco_env import MujocoTaskEnv
from mpc_warp.solvers.mppi_warp import WarpMPPIConfig, WarpMPPISolver
from mpc_warp.tasks import (
    CartPole,
    Crane,
    DoubleCartPole,
    G1VelocityTracking,
    Go1Walking,
    HumanoidStandup,
    Particle,
    Pendulum,
    TrajectoryTask,
    Walker,
)
from mpc_warp.tasks.utils.go1_trot_reference import DEFAULT_OUT as _GO1_TROT_NPZ
from mpc_warp.tasks.utils.go1_trot_reference import JOINT_NAMES as _GO1_JOINT_NAMES
from mpc_warp.tasks.utils.go1_trot_reference import generate as _generate_go1_trot


def _make_go1_trajectory() -> TrajectoryTask:
    """Return a GO1 task that tracks a physics-generated trot reference.

    Generates the npz on first call and caches it at data/go1_trot.npz.
    """
    if not _GO1_TROT_NPZ.exists():
        print("Generating GO1 trot reference trajectory (one-time)…")
        _generate_go1_trot()
    base = Go1Walking()
    return TrajectoryTask.from_npz(
        base,
        npz_path=_GO1_TROT_NPZ,
        joint_names=_GO1_JOINT_NAMES,
        qpos_weight=3.0,
        qvel_weight=0.05,
        loop=True,
    )


TASKS = {
    "pendulum": Pendulum,
    "cart_pole": CartPole,
    "double_cart_pole": DoubleCartPole,
    "particle": Particle,
    "walker": Walker,
    "crane": Crane,
    "humanoid_standup": HumanoidStandup,
    "g1_velocity": G1VelocityTracking,
    "go1_walking": Go1Walking,
    "go1_trot": _make_go1_trajectory,  # factory, not a class
}

CONFIGS: dict[str, dict] = {
    "pendulum": {"horizon": 16, "num_samples": 128, "noise_sigma": 0.5, "temperature": 0.1},
    "cart_pole": {"horizon": 20, "num_samples": 128, "noise_sigma": 0.3, "temperature": 0.1},
    "double_cart_pole": {"horizon": 20, "num_samples": 128, "noise_sigma": 0.3, "temperature": 0.1},
    "particle": {"horizon": 12, "num_samples": 64, "noise_sigma": 0.3, "temperature": 0.5},
    "walker": {"horizon": 16, "num_samples": 128, "noise_sigma": 0.3, "temperature": 0.5},
    "crane": {"horizon": 16, "num_samples": 64, "noise_sigma": 0.05, "temperature": 0.5},
    "humanoid_standup": {"horizon": 20, "num_samples": 256, "noise_sigma": 0.3, "temperature": 0.1, "nominal_return": 0.1},
    "g1_velocity": {"horizon": 32, "num_samples": 256, "noise_sigma": 0.3, "temperature": 0.05, "nominal_return": 0.15},
    "go1_walking": {"horizon": 30, "num_samples": 256, "noise_sigma": 0.3, "temperature": 0.1},
    "go1_trot": {"horizon": 30, "num_samples": 256, "noise_sigma": 0.3, "temperature": 0.1},
}


# ── viewer mode helpers ────────────────────────────────────────────────────────


def _run_mujoco_viewer(
    task_name: str,
    max_steps: int,
    num_samples: int | None,
    mjviser_panel=None,  # optional MppiPanel for "both" mode
) -> None:
    """Native mujoco.viewer loop (requires mjpython on macOS with --render)."""
    import mujoco.viewer

    from mpc_warp.viz.trajectory import TrajectoryViz

    task = TASKS[task_name]()
    env = MujocoTaskEnv(task)
    env.reset()

    cfg = _make_cfg(task_name, num_samples, render=True)
    solver = WarpMPPISolver(task, WarpMPPIConfig(**cfg), seed=0)
    print(f"{task_name} [mujoco/{solver.device}]: horizon={cfg['horizon']} samples={cfg['num_samples']}")

    viewer_ctx = mujoco.viewer.launch_passive(env.model, env.data)
    viz = TrajectoryViz(task)

    total_cost = 0.0
    step_start = time.perf_counter()

    with viewer_ctx:
        for step_i in range(max_steps):
            if not viewer_ctx.is_running():
                break
            u = solver.command(env.data)
            _, reward, *_ = env.step(u)
            total_cost -= reward

            ref_pos = None
            if isinstance(task, TrajectoryTask):
                ref_pos = task.ref_positions_window(solver.cfg.horizon)
                task.advance()

            _vel_arrow = np.array([task.vx_cmd, task.vy_cmd]) if isinstance(task, G1VelocityTracking) else None
            with viewer_ctx.lock():
                viz.update_scene(
                    env.data,
                    solver.planned_sites,
                    ref_pos,
                    viewer_ctx.user_scn,
                    velocity_arrow=_vel_arrow,
                )
            figs = viz.build_figures(
                solver.last_cost,
                solver.cost_weights,
                u,
                cost_terms=solver.last_cost_terms,
                u_nominal=solver.u_nominal_snapshot,
            )
            viewer_ctx.set_figures(figs)
            _extra = _velocity_cmd_lines(task)
            viewer_ctx.set_texts(viz.build_texts(solver.last_cost, solver.cost_weights, extra_lines=_extra))
            viewer_ctx.sync()

            if mjviser_panel is not None:
                _act_names = [
                    mujoco.mj_id2name(task.mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or f"a{i}"
                    for i in range(task.mj_model.nu)
                ]
                mjviser_panel.update(
                    solver.last_cost,
                    solver.cost_weights,
                    u,
                    cost_terms=solver.last_cost_terms,
                    u_nominal=solver.u_nominal_snapshot,
                )

            elapsed = time.perf_counter() - step_start
            target = (step_i + 1) * task.dt
            if target > elapsed:
                time.sleep(target - elapsed)

            if (step_i + 1) % 50 == 0:
                print(f"  step {step_i + 1:4d}  cumulative_cost={total_cost:.3f}")

    print(f"{task_name}: done after {step_i + 1} steps, total_cost={total_cost:.3f}")


def _run_mjviser(task_name: str, max_steps: int, num_samples: int | None) -> None:
    """mjviser browser viewer loop — no native desktop window needed."""
    import viser
    from mjviser import ViserMujocoScene

    from mpc_warp.viz.mjviser_panel import MppiPanel

    task = TASKS[task_name]()
    env = MujocoTaskEnv(task)
    env.reset()

    cfg = _make_cfg(task_name, num_samples, render=True)
    solver = WarpMPPISolver(task, WarpMPPIConfig(**cfg), seed=0)
    print(f"{task_name} [mjviser/{solver.device}]: horizon={cfg['horizon']} samples={cfg['num_samples']}")

    server = viser.ViserServer()

    act_names = [mujoco.mj_id2name(task.mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or f"a{i}" for i in range(task.mj_model.nu)]
    panel = MppiPanel(server, task.mj_model.nu, act_names)

    scene = ViserMujocoScene(server, task.mj_model, num_envs=1)

    # Reference trajectory point cloud (only for TrajectoryTask with body positions).
    _ref_cloud = None
    if isinstance(task, TrajectoryTask) and task.ref_body_pos is not None:
        _dummy = np.zeros((1, 3), dtype=np.float32)
        _ref_cloud = server.scene.add_point_cloud(
            name="ref_trajectory",
            points=_dummy,
            colors=np.array([[1.0, 0.55, 0.0]], dtype=np.float32),
            point_size=0.04,
            point_shape="circle",
        )

    # Planned MPPI trajectory point cloud (cyan dots — updated each step).
    _plan_cloud = None
    if task.trace_site_ids:
        _dummy = np.zeros((1, 3), dtype=np.float32)
        _plan_cloud = server.scene.add_point_cloud(
            name="planned_trajectory",
            points=_dummy,
            colors=np.array([[0.0, 0.9, 0.9]], dtype=np.float32),
            point_size=0.03,
            point_shape="circle",
        )

    # Target-path strip for velocity-command tasks (green dots on ground plane).
    _target_path_cloud = None
    if isinstance(task, G1VelocityTracking):
        _dummy = np.zeros((1, 3), dtype=np.float32)
        _target_path_cloud = server.scene.add_point_cloud(
            name="target_path",
            points=_dummy,
            colors=np.array([[0.1, 0.9, 0.2]], dtype=np.float32),
            point_size=0.04,
            point_shape="circle",
        )

    total_cost = 0.0
    step_start = time.perf_counter()

    print(f"  Open browser: http://localhost:{server.get_port()}")
    for step_i in range(max_steps):
        u = solver.command(env.data)
        _, reward, *_ = env.step(u)
        total_cost -= reward

        if isinstance(task, TrajectoryTask):
            if _ref_cloud is not None:
                window = task.ref_positions_window(solver.cfg.horizon)
                if window is not None:
                    _ref_cloud.points = window.astype(np.float32)
                    _ref_cloud.colors = np.tile([1.0, 0.55, 0.0], (len(window), 1)).astype(np.float32)
            task.advance()

        # Update planned MPPI trajectory cloud.
        if _plan_cloud is not None and solver.planned_sites is not None:
            pts = solver.planned_sites[:, :, :].reshape(-1, 3)
            _plan_cloud.points = pts.astype(np.float32)
            _plan_cloud.colors = np.tile([0.0, 0.9, 0.9], (len(pts), 1)).astype(np.float32)

        # Update ground-plane target path for velocity commands.
        if _target_path_cloud is not None:
            root_xy = env.data.qpos[:2]
            speed = np.sqrt(task.vx_cmd**2 + task.vy_cmd**2)
            path_len = max(1.5, speed * 2.0)
            n_dots = 20
            ts = np.linspace(0.1, path_len, n_dots)
            if speed > 1e-3:
                dx = task.vx_cmd / speed
                dy = task.vy_cmd / speed
            else:
                dx, dy = 1.0, 0.0
            path_pts = np.stack([root_xy[0] + dx * ts, root_xy[1] + dy * ts, np.full(n_dots, 0.02)], axis=1)
            _target_path_cloud.points = path_pts.astype(np.float32)
            _target_path_cloud.colors = np.tile([0.1, 0.9, 0.2], (n_dots, 1)).astype(np.float32)

        scene.update_from_mjdata(env.data)
        panel.update(
            solver.last_cost,
            solver.cost_weights,
            u,
            cost_terms=solver.last_cost_terms,
            u_nominal=solver.u_nominal_snapshot,
        )

        elapsed = time.perf_counter() - step_start
        target = (step_i + 1) * task.dt
        if target > elapsed:
            time.sleep(target - elapsed)

        if (step_i + 1) % 50 == 0:
            print(f"  step {step_i + 1:4d}  cumulative_cost={total_cost:.3f}")

    server.stop()
    print(f"{task_name}: done after {step_i + 1} steps, total_cost={total_cost:.3f}")


def _run_both(task_name: str, max_steps: int, num_samples: int | None) -> None:
    """Native viewer (main thread) + mjviser panels (background thread)."""
    import viser

    from mpc_warp.viz.mjviser_panel import MppiPanel

    # Start viser server in background (non-blocking).
    server = viser.ViserServer()
    task_tmp = TASKS[task_name]()
    act_names = [
        mujoco.mj_id2name(task_tmp.mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or f"a{i}" for i in range(task_tmp.mj_model.nu)
    ]
    panel = MppiPanel(server, task_tmp.mj_model.nu, act_names)
    print(f"  Browser panels: http://localhost:{server.get_port()}")

    # Run native viewer in main thread (required for Cocoa on macOS).
    _run_mujoco_viewer(task_name, max_steps, num_samples, mjviser_panel=panel)
    server.stop()


# ── utilities ─────────────────────────────────────────────────────────────────


def _velocity_cmd_lines(task) -> list[tuple[str, str]] | None:
    if isinstance(task, G1VelocityTracking):
        return [(f"cmd  vx {task.vx_cmd:+.2f}  vy {task.vy_cmd:+.2f}", f"yaw {task.yaw_cmd:+.2f} rad/s")]
    return None


def _make_cfg(task_name: str, num_samples: int | None, render: bool) -> dict:
    cfg = dict(CONFIGS[task_name])
    if num_samples is not None:
        cfg["num_samples"] = num_samples
    elif render:
        cfg["num_samples"] = max(8, cfg["num_samples"] // 4)
    return cfg


class _nullctx:
    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass


# ── entry point ───────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("task", choices=list(TASKS))
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument(
        "--viewer",
        choices=["mujoco", "mjviser", "both"],
        default="mujoco",
        help=(
            "mujoco: native desktop viewer (needs mjpython on macOS); "
            "mjviser: browser-based viewer; "
            "both: native viewer + mjviser panels"
        ),
    )
    parser.add_argument("--render", action="store_true", help="Enable viewer (required for mujoco/both modes on macOS)")
    parser.add_argument("--num-samples", type=int, default=None, help="Override sample count")
    args = parser.parse_args()

    if args.viewer == "mjviser":
        _run_mjviser(args.task, args.steps, args.num_samples)
    elif args.viewer == "both":
        _run_both(args.task, args.steps, args.num_samples)
    else:
        # mujoco mode
        if args.render:
            _run_mujoco_viewer(args.task, args.steps, args.num_samples)
        else:
            # headless: fast loop, no viewer
            task = TASKS[args.task]()
            env = MujocoTaskEnv(task)
            env.reset()
            cfg = _make_cfg(args.task, args.num_samples, render=False)
            solver = WarpMPPISolver(task, WarpMPPIConfig(**cfg), seed=0)
            print(f"{args.task} [{solver.device}]: horizon={cfg['horizon']} samples={cfg['num_samples']}")
            total_cost = 0.0
            for step_i in range(args.steps):
                u = solver.command(env.data)
                _, reward, *_ = env.step(u)
                total_cost -= reward
                if (step_i + 1) % 50 == 0:
                    print(f"  step {step_i + 1:4d}  cumulative_cost={total_cost:.3f}")
            print(f"{args.task}: done after {args.steps} steps, total_cost={total_cost:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
