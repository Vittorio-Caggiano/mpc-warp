from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import mujoco
import numpy as np

MODELS_DIR = Path(__file__).parent.parent / "models"


class Task(ABC):
    """Abstract task: a MuJoCo model paired with running and terminal cost functions.

    Subclasses implement ``running_cost`` and ``terminal_cost`` against ``mujoco.MjData``.
    ``u_min``/``u_max`` are set automatically from the model's actuator control limits.
    """

    def __init__(self, mj_model: mujoco.MjModel, trace_sites: list[str] | None = None) -> None:
        self.mj_model = mj_model
        self.trace_site_ids: list[int] = [mj_model.site(name).id for name in (trace_sites or [])]
        self.u_min = np.where(
            mj_model.actuator_ctrllimited,
            mj_model.actuator_ctrlrange[:, 0],
            -np.inf,
        )
        self.u_max = np.where(
            mj_model.actuator_ctrllimited,
            mj_model.actuator_ctrlrange[:, 1],
            np.inf,
        )
        self.dt = mj_model.opt.timestep

    @abstractmethod
    def running_cost(self, data: mujoco.MjData, control: np.ndarray) -> float:
        """Step-wise cost ℓ(xₜ, uₜ)."""

    @abstractmethod
    def terminal_cost(self, data: mujoco.MjData) -> float:
        """Terminal cost ϕ(x_T)."""

    def cost_terms(self, data: mujoco.MjData, control: np.ndarray) -> dict[str, float]:
        """Named breakdown of running_cost.  Override to expose per-term values.

        Returns an ordered dict mapping term name → cost value.
        Default: single entry with the total cost.
        """
        return {"total": self.running_cost(data, control)}
