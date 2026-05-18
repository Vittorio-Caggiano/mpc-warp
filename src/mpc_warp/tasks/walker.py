from __future__ import annotations

import mujoco
import numpy as np

from .task_base import MODELS_DIR, Task


class Walker(Task):
    """Planar biped walking forward."""

    def __init__(self) -> None:
        mj_model = mujoco.MjModel.from_xml_path(
            str(MODELS_DIR / "walker" / "scene.xml")
        )
        super().__init__(mj_model, trace_sites=["torso_site"])

        self._pos_adr = mj_model.sensor_adr[
            mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SENSOR, "torso_position")
        ]
        self._vel_adr = mj_model.sensor_adr[
            mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SENSOR, "torso_subtreelinvel")
        ]
        self._zax_adr = mj_model.sensor_adr[
            mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SENSOR, "torso_zaxis")
        ]
        self.target_velocity = 1.5
        self.target_height = 1.2

    def terminal_cost(self, data: mujoco.MjData) -> float:
        height = float(data.sensordata[self._pos_adr + 2])
        velocity = float(data.sensordata[self._vel_adr])
        zaxis_z = float(data.sensordata[self._zax_adr + 2])
        height_cost = (height - self.target_height) ** 2
        orientation_cost = (zaxis_z - 1.0) ** 2
        velocity_cost = (velocity - self.target_velocity) ** 2
        return 10.0 * height_cost + 3.0 * orientation_cost + 1.0 * velocity_cost

    def running_cost(self, data: mujoco.MjData, control: np.ndarray) -> float:
        return self.terminal_cost(data) + 0.1 * float(np.sum(control**2))
