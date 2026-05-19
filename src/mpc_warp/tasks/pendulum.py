from __future__ import annotations

import math

import mujoco
import numpy as np

from .task_base import MODELS_DIR, Task


class Pendulum(Task):
    """Inverted pendulum swingup."""

    def __init__(self) -> None:
        mj_model = mujoco.MjModel.from_xml_path(
            str(MODELS_DIR / "pendulum" / "scene.xml")
        )
        super().__init__(mj_model, trace_sites=["tip"])

    def _distance_to_upright(self, data: mujoco.MjData) -> float:
        theta = data.qpos[0] - math.pi
        return (math.cos(theta) - 1) ** 2 + math.sin(theta) ** 2

    def running_cost(self, data: mujoco.MjData, control: np.ndarray) -> float:
        theta_cost = self._distance_to_upright(data)
        theta_dot_cost = 0.01 * float(data.qvel[0]) ** 2
        control_cost = 0.001 * float(np.sum(control**2))
        return theta_cost + theta_dot_cost + control_cost

    def cost_terms(self, data: mujoco.MjData, control: np.ndarray) -> dict[str, float]:
        return {
            "angle":   self._distance_to_upright(data),
            "ang_vel": 0.01 * float(data.qvel[0]) ** 2,
            "control": 0.001 * float(np.sum(control**2)),
        }

    def terminal_cost(self, data: mujoco.MjData) -> float:
        return self._distance_to_upright(data) + 0.01 * float(data.qvel[0]) ** 2

    def batch_running_cost(self, qpos, qvel, ctrl, sensordata, site_xpos, mocap_pos):
        theta = qpos[:, 0].astype(np.float64) - math.pi
        angle_cost = (np.cos(theta) - 1.0) ** 2 + np.sin(theta) ** 2
        vel_cost   = 0.01 * qvel[:, 0].astype(np.float64) ** 2
        ctrl_cost  = 0.001 * np.sum(ctrl ** 2, axis=1)
        return angle_cost + vel_cost + ctrl_cost
