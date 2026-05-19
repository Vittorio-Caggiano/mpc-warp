from __future__ import annotations

import mujoco
import numpy as np

from .task_base import MODELS_DIR, Task


class Crane(Task):
    """Luffing crane moves a payload to a target position."""

    def __init__(self) -> None:
        mj_model = mujoco.MjModel.from_xml_path(str(MODELS_DIR / "crane" / "scene.xml"))
        super().__init__(mj_model, trace_sites=["payload_end"])
        self._pos_adr = int(mj_model.sensor_adr[mj_model.sensor("payload_pos").id])
        self._vel_adr = int(mj_model.sensor_adr[mj_model.sensor("payload_vel").id])

    def running_cost(self, data: mujoco.MjData, control: np.ndarray) -> float:
        pos = np.array(data.sensordata[self._pos_adr : self._pos_adr + 3])
        vel = np.array(data.sensordata[self._vel_adr : self._vel_adr + 3])
        return float(np.sum(pos**2)) + 0.1 * float(np.sum(vel**2)) + 0.01 * float(np.sum(control**2))

    def terminal_cost(self, data: mujoco.MjData) -> float:
        return self.running_cost(data, np.zeros(self.mj_model.nu))

    def batch_running_cost(self, qpos, qvel, ctrl, sensordata, site_xpos, mocap_pos):
        sd = sensordata.astype(np.float64)
        pos = sd[:, self._pos_adr : self._pos_adr + 3]
        vel = sd[:, self._vel_adr : self._vel_adr + 3]
        ctrl_cost = 0.01 * np.sum(ctrl.astype(np.float64) ** 2, axis=1)
        return np.sum(pos**2, axis=1) + 0.1 * np.sum(vel**2, axis=1) + ctrl_cost
