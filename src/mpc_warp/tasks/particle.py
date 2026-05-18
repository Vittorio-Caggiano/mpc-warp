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
