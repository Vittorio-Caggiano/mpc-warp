from __future__ import annotations

import mujoco
import numpy as np

from mpc_warp.tasks.task_base import Task


class MujocoTaskEnv:
    """Wraps a Task into the reset/step/get_internal_state/set_internal_state interface.

    Holds the live ``mujoco.MjData`` (``self.data``) that ``mujoco.viewer`` reads
    for rendering and that ``WarpMPPISolver.command`` accepts directly.
    State vector is qpos concatenated with qvel.
    """

    def __init__(self, task: Task) -> None:
        self.task = task
        self.model = task.mj_model
        self.data = mujoco.MjData(self.model)
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        self.obs_dim = self.model.nq + self.model.nv
        self.act_dim = self.model.nu

    def _obs(self) -> list[float]:
        return list(self.data.qpos) + list(self.data.qvel)

    def reset(self, seed: int | None = None) -> tuple[list[float], dict]:
        self.task.reset_data(self.data)
        return self._obs(), {}

    def get_internal_state(self) -> dict:
        return {
            "qpos": list(self.data.qpos),
            "qvel": list(self.data.qvel),
            "mocap_pos": list(self.data.mocap_pos.flatten()),
            "mocap_quat": list(self.data.mocap_quat.flatten()),
        }

    def set_internal_state(self, snapshot: dict) -> list[float]:
        self.data.qpos[:] = snapshot["qpos"]
        self.data.qvel[:] = snapshot["qvel"]
        if "mocap_pos" in snapshot and self.model.nmocap > 0:
            self.data.mocap_pos[:] = np.array(snapshot["mocap_pos"]).reshape(self.model.nmocap, 3)
            self.data.mocap_quat[:] = np.array(snapshot["mocap_quat"]).reshape(self.model.nmocap, 4)
        mujoco.mj_forward(self.model, self.data)
        return self._obs()

    def step(self, action: list[float]) -> tuple[list[float], float, bool, bool, dict]:
        ctrl = np.clip(
            np.array(action, dtype=np.float64),
            self.task.u_min,
            self.task.u_max,
        )
        self.data.ctrl[:] = ctrl
        mujoco.mj_step(self.model, self.data)
        reward = -self.task.running_cost(self.data, ctrl)
        return self._obs(), float(reward), False, False, {}
