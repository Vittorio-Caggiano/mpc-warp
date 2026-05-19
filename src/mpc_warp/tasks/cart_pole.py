from __future__ import annotations

import math

import mujoco
import numpy as np

from .task_base import MODELS_DIR, Task


class CartPole(Task):
    """Cart-pole swingup."""

    def __init__(self) -> None:
        mj_model = mujoco.MjModel.from_xml_path(
            str(MODELS_DIR / "cart_pole" / "scene.xml")
        )
        super().__init__(mj_model, trace_sites=["tip"])

    def _distance_to_upright(self, data: mujoco.MjData) -> float:
        # qpos[0] = slider, qpos[1] = hinge
        theta = data.qpos[1] + math.pi
        return (math.cos(theta) - 1) ** 2 + math.sin(theta) ** 2

    def running_cost(self, data: mujoco.MjData, control: np.ndarray) -> float:
        theta_cost = self._distance_to_upright(data)
        centering_cost = float(data.qpos[0]) ** 2
        velocity_cost = 0.01 * float(np.sum(np.array(data.qvel) ** 2))
        control_cost = 0.01 * float(np.sum(control**2))
        return theta_cost + centering_cost + velocity_cost + control_cost

    def terminal_cost(self, data: mujoco.MjData) -> float:
        theta_cost = 10 * self._distance_to_upright(data)
        centering_cost = float(data.qpos[0]) ** 2
        velocity_cost = 0.01 * float(np.sum(np.array(data.qvel) ** 2))
        return theta_cost + centering_cost + velocity_cost

    def batch_running_cost(self, qpos, qvel, ctrl, sensordata, site_xpos, mocap_pos):
        qpos = qpos.astype(np.float64); qvel = qvel.astype(np.float64)
        theta = qpos[:, 1] + math.pi
        theta_cost     = (np.cos(theta) - 1.0) ** 2 + np.sin(theta) ** 2
        centering_cost = qpos[:, 0] ** 2
        vel_cost       = 0.01 * np.sum(qvel ** 2, axis=1)
        ctrl_cost      = 0.01 * np.sum(ctrl ** 2, axis=1)
        return theta_cost + centering_cost + vel_cost + ctrl_cost
