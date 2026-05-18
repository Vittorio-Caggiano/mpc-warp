from __future__ import annotations

import mujoco
import numpy as np

from .task_base import MODELS_DIR, Task


def _quat_rotate(v: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Rotate vector v by unit quaternion q = [w, x, y, z].

    Uses the double-cross formula v' = v + 2w(k×v) + 2(k×(k×v)) where k=[x,y,z].
    """
    w, x, y, z = q
    t = 2.0 * np.cross(np.array([x, y, z]), v)
    return v + w * t + np.cross(np.array([x, y, z]), t)


class HumanoidStandup(Task):
    """Standup task for the Unitree G1 humanoid."""

    def __init__(self) -> None:
        mj_model = mujoco.MjModel.from_xml_path(
            str(MODELS_DIR / "g1" / "scene.xml")
        )
        super().__init__(mj_model, trace_sites=["imu_in_torso"])

        self._orient_adr = int(mj_model.sensor_adr[mj_model.sensor("imu_in_torso_quat").id])
        self._torso_id = mj_model.site("imu_in_torso").id
        self.target_height = 0.9
        self.qstand = mj_model.keyframe("stand").qpos.copy()

    def _torso_height(self, data: mujoco.MjData) -> float:
        return float(data.site_xpos[self._torso_id, 2])

    def _torso_orientation_cost(self, data: mujoco.MjData) -> float:
        quat = np.array(data.sensordata[self._orient_adr: self._orient_adr + 4])
        upright = np.array([0.0, 0.0, 1.0])
        rotated = _quat_rotate(upright, quat)
        return float(np.sum(rotated**2))

    def running_cost(self, data: mujoco.MjData, control: np.ndarray) -> float:
        orientation_cost = self._torso_orientation_cost(data)
        height_cost = (self._torso_height(data) - self.target_height) ** 2
        nominal_cost = float(np.sum((np.array(data.qpos[7:]) - self.qstand[7:]) ** 2))
        return 10.0 * orientation_cost + 10.0 * height_cost + 0.1 * nominal_cost

    def terminal_cost(self, data: mujoco.MjData) -> float:
        return self.running_cost(data, np.zeros(self.mj_model.nu))
