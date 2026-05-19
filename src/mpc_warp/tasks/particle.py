from __future__ import annotations

import mujoco
import numpy as np

from .task_base import MODELS_DIR, Task


class Particle(Task):
    """Planar point mass chases a mocap target."""

    def __init__(self) -> None:
        mj_model = mujoco.MjModel.from_xml_path(
            str(MODELS_DIR / "particle" / "scene.xml")
        )
        super().__init__(mj_model, trace_sites=["pointmass"])
        self.pointmass_id = mj_model.site("pointmass").id

    def terminal_cost(self, data: mujoco.MjData) -> float:
        pos = data.site_xpos[self.pointmass_id]
        target = data.mocap_pos[0]
        position_cost = float(np.sum((pos - target) ** 2))
        velocity_cost = float(np.sum(np.array(data.qvel) ** 2))
        return 5.0 * position_cost + 0.1 * velocity_cost

    def running_cost(self, data: mujoco.MjData, control: np.ndarray) -> float:
        return self.terminal_cost(data) + 0.1 * float(np.sum(control**2))

    def batch_running_cost(self, qpos, qvel, ctrl, sensordata, site_xpos, mocap_pos):
        # site_xpos: (N, nsite, 3), mocap_pos: (N, nmocap, 3)
        pos    = site_xpos[:, self.pointmass_id, :].astype(np.float64)   # (N, 3)
        target = mocap_pos[:, 0, :].astype(np.float64)                   # (N, 3)
        pos_cost = 5.0 * np.sum((pos - target) ** 2, axis=1)
        vel_cost = 0.1 * np.sum(qvel.astype(np.float64) ** 2, axis=1)
        ctrl_cost = 0.1 * np.sum(ctrl ** 2, axis=1)
        return pos_cost + vel_cost + ctrl_cost
