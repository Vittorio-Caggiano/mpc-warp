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

    def batch_running_cost(
        self,
        qpos: np.ndarray,       # (N, nq)  float32
        qvel: np.ndarray,       # (N, nv)  float32
        ctrl: np.ndarray,       # (N, nu)  float64 perturbed controls
        sensordata: np.ndarray, # (N, nsensordata) float32
        site_xpos: np.ndarray,  # (N, nsite, 3) float32
        mocap_pos: np.ndarray,  # (N, nmocap, 3) float32
    ) -> np.ndarray:            # (N,) float64
        """Vectorised running cost over N parallel worlds (bulk GPU→CPU transfer path).

        Default: slow loop over running_cost() via a shared MjData proxy.
        Override with vectorised numpy for full speed.
        Tasks that read sensordata or site_xpos MUST override this method,
        because the base proxy does not call mj_forward.
        """
        N = qpos.shape[0]
        costs = np.empty(N, dtype=np.float64)
        proxy = mujoco.MjData(self.mj_model)
        for i in range(N):
            proxy.qpos[:] = qpos[i].astype(np.float64)
            proxy.qvel[:] = qvel[i].astype(np.float64)
            costs[i] = self.running_cost(proxy, ctrl[i])
        return costs
