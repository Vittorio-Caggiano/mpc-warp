from __future__ import annotations

import mujoco
import numpy as np

from .task_base import MODELS_DIR, Task


class DoubleCartPole(Task):
    """Swing-up task for a double pendulum on a cart."""

    def __init__(self) -> None:
        mj_model = mujoco.MjModel.from_xml_path(str(MODELS_DIR / "double_cart_pole" / "scene.xml"))
        super().__init__(mj_model, trace_sites=["tip"])
        self._tip_id = mj_model.site("tip").id

    def _distance_to_upright(self, data: mujoco.MjData) -> float:
        tip_z = float(data.site_xpos[self._tip_id, 2])
        tip_x = float(data.site_xpos[self._tip_id, 0])
        cart_x = float(data.qpos[0])
        return (tip_z - 4.0) ** 2 + (tip_x - cart_x) ** 2

    def running_cost(self, data: mujoco.MjData, control: np.ndarray) -> float:
        upright_cost = self._distance_to_upright(data)
        cart_cost = 0.1 * float(data.qpos[0]) ** 2
        velocity_cost = 0.1 * float(np.sum(np.array(data.qvel) ** 2))
        control_cost = 0.001 * float(np.sum(control**2))
        return upright_cost + cart_cost + velocity_cost + control_cost

    def terminal_cost(self, data: mujoco.MjData) -> float:
        upright_cost = 10 * self._distance_to_upright(data)
        centering_cost = 10 * float(data.qpos[0]) ** 2
        velocity_cost = float(np.sum(np.array(data.qvel) ** 2))
        return upright_cost + centering_cost + velocity_cost

    def batch_running_cost(self, qpos, qvel, ctrl, sensordata, site_xpos, mocap_pos):
        # site_xpos: (N, nsite, 3) — tip is self._tip_id
        tip = site_xpos[:, self._tip_id, :].astype(np.float64)  # (N, 3)
        cart_x = qpos[:, 0].astype(np.float64)
        upright_cost = (tip[:, 2] - 4.0) ** 2 + (tip[:, 0] - cart_x) ** 2
        cart_cost = 0.1 * cart_x**2
        vel_cost = 0.1 * np.sum(qvel.astype(np.float64) ** 2, axis=1)
        ctrl_cost = 0.001 * np.sum(ctrl**2, axis=1)
        return upright_cost + cart_cost + vel_cost + ctrl_cost
