"""MPPI solver backed by mujoco_warp for fully parallel rollouts.

All ``num_samples`` trajectories are advanced simultaneously via a single
``mjw.step`` call on a batched ``nworld=num_samples`` Data object — GPU when
CUDA is available, compiled Warp CPU kernels otherwise.

Cost evaluation uses a single bulk GPU→CPU transfer per array per timestep
(via ``task.batch_running_cost``) instead of N individual ``get_data_into``
calls, which was the dominant bottleneck on GPU.
"""
from __future__ import annotations

from dataclasses import dataclass

import mujoco
import mujoco_warp as mjw
import numpy as np
import warp as wp

from mpc_warp.tasks.task_base import Task


@dataclass
class WarpMPPIConfig:
    horizon: int = 20
    num_samples: int = 64
    noise_sigma: float = 0.3
    temperature: float = 1.0


class WarpMPPISolver:
    """MPPI using mujoco_warp batched physics for parallel rollouts.

    Usage::

        task = Pendulum()
        env  = MujocoTaskEnv(task)
        cfg  = WarpMPPIConfig(horizon=16, num_samples=128)
        solver = WarpMPPISolver(task, cfg, seed=0)

        x = env.reset()[0]
        while True:
            u = solver.command(env)   # reads/writes env.data in place
            env.step(u)
    """

    def __init__(self, task: Task, cfg: WarpMPPIConfig, seed: int = 0) -> None:
        self.task = task
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)

        mjm = task.mj_model
        self.mjm = mjm
        self.nu = mjm.nu

        self.warp_model = mjw.put_model(mjm)
        # Allocate num_samples parallel worlds on the device (GPU or CPU).
        self.warp_data = mjw.make_data(mjm, nworld=cfg.num_samples)

        self.u_nominal = np.zeros((cfg.horizon, self.nu), dtype=np.float64)

        # Diagnostics updated each command() call.
        self.planned_sites: np.ndarray | None = None  # (H, n_sites, 3)
        self.last_cost: float = 0.0
        self.cost_weights: np.ndarray = np.zeros(cfg.num_samples)
        self.last_cost_terms: dict[str, float] = {}
        self.u_nominal_snapshot: np.ndarray = np.zeros((cfg.horizon, self.nu))

    def command(self, env_data: mujoco.MjData) -> np.ndarray:
        """Compute the next action from the current MjData state.

        Returns a 1-D array of shape ``(nu,)``.
        """
        cfg = self.cfg
        N, H, nu = cfg.num_samples, cfg.horizon, self.nu

        # Sample action perturbations: (N, H, nu)
        noise = self.rng.standard_normal((N, H, nu)) * cfg.noise_sigma
        perturbed = np.clip(
            self.u_nominal[None] + noise,
            self.task.u_min,
            self.task.u_max,
        ).astype(np.float32)

        # Upload the current state into all N worlds.
        self.warp_data = mjw.put_data(self.mjm, env_data, nworld=N)

        # Roll out H steps, accumulating cost for each sample.
        # One bulk GPU→CPU transfer per array per step instead of N individual
        # get_data_into calls — critical for GPU performance.
        costs = np.zeros(N, dtype=np.float64)
        for t in range(H):
            self.warp_data.ctrl.assign(perturbed[:, t, :])
            mjw.step(self.warp_model, self.warp_data)

            costs += self.task.batch_running_cost(
                qpos=self.warp_data.qpos.numpy(),
                qvel=self.warp_data.qvel.numpy(),
                ctrl=perturbed[:, t, :].astype(np.float64),
                sensordata=self.warp_data.sensordata.numpy(),
                site_xpos=self.warp_data.site_xpos.numpy(),
                mocap_pos=self.warp_data.mocap_pos.numpy(),
            )

        # MPPI importance weights.
        beta = costs.min()
        ws = np.exp(-(costs - beta) / max(cfg.temperature, 1e-8))
        ws /= ws.sum()

        # Update nominal trajectory.
        self.u_nominal += (ws[:, None, None] * noise).sum(axis=0)
        self.u_nominal = np.clip(self.u_nominal, self.task.u_min, self.task.u_max)

        u0 = self.u_nominal[0].copy()

        # Store diagnostics before shifting.
        self.last_cost = float(self.task.running_cost(env_data, u0))
        self.last_cost_terms = self.task.cost_terms(env_data, u0)
        self.cost_weights = ws
        self.u_nominal_snapshot = self.u_nominal.copy()

        # Forward pass to record planned site positions along nominal trajectory.
        if self.task.trace_site_ids:
            n_sites = len(self.task.trace_site_ids)
            planned = np.zeros((H, n_sites, 3))
            snap = mujoco.MjData(self.mjm)
            mujoco.mj_copyData(snap, self.mjm, env_data)
            for t in range(H):
                snap.ctrl[:] = np.clip(self.u_nominal[t], self.task.u_min, self.task.u_max)
                mujoco.mj_step(self.mjm, snap)
                for k, sid in enumerate(self.task.trace_site_ids):
                    planned[t, k] = snap.site_xpos[sid]
            self.planned_sites = planned
        else:
            self.planned_sites = None

        self.u_nominal = np.roll(self.u_nominal, -1, axis=0)
        self.u_nominal[-1] = 0.0
        return u0

    @property
    def device(self) -> str:
        """The Warp device being used (e.g. 'cuda:0' or 'cpu')."""
        return wp.get_device().alias
