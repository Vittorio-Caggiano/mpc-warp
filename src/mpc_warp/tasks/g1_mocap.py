"""Unitree G1 23-DOF humanoid — motion-capture reference tracking.

Replicates the HumanoidMocap task from hydrax in the mpc-warp format:
https://github.com/vincekurtz/hydrax/blob/main/hydrax/tasks/humanoid_mocap.py

Reference data comes from the LocoMuJoCo dataset on HuggingFace:
https://huggingface.co/datasets/robfiras/loco-mujoco-datasets

Cost terms (exponential kernel, each in [0, 1], normalized sum):
  - body_position    : world-frame body positions  (weight 1.0)
  - body_orientation : world-frame body quaternions (weight 0.1)
  - configuration    : qpos tracking               (weight 0.1)
  - velocity         : qvel tracking               (weight 0.01)
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from .task_base import MODELS_DIR, Task

# Default reference file from the LocoMuJoCo dataset (repo_type="dataset"):
_DEFAULT_REF = "Lafan1/mocap/UnitreeG1/walk1_subject1.npz"


@dataclass
class G1MocapOptions:
    """Cost weights for G1MocapTracking (un-normalised; will be normalised in __init__)."""

    body_position_weight: float = 1.0
    body_orientation_weight: float = 0.1
    configuration_weight: float = 0.1
    velocity_weight: float = 0.01


class G1MocapTracking(Task):
    """G1 23-DOF humanoid tracks a LocoMuJoCo motion-capture reference.

    The reference is indexed by an internal counter advanced with ``advance()``.
    Call ``advance()`` once after each ``env.step()`` to keep the reference in sync.

    Args:
        reference_filename: Path within the LocoMuJoCo HuggingFace dataset.
        options: Cost weight configuration.
        loop: Whether to loop the reference at the end (default True).
    """

    # scene_23dof.xml has 10 sensors / 33 dims.  We append 30 framepos (3-dim) +
    # 30 framequat (4-dim) sensors for all non-world bodies → 70 sensors / 243 dims.
    _BODY_POS_SENSOR_ADR = 33  # start of framepos sensors  (33 + 30*3 = 123)
    _BODY_QUAT_SENSOR_ADR = 123  # start of framequat sensors (123 + 30*4 = 243)

    def __init__(
        self,
        reference_filename: str = _DEFAULT_REF,
        options: G1MocapOptions | None = None,
        loop: bool = True,
    ) -> None:
        if options is None:
            options = G1MocapOptions()

        # ── Build model — add body-pose sensors via MjSpec ─────────────────
        # We need body names before adding sensors, so do a quick compile first.
        spec = mujoco.MjSpec.from_file(str(MODELS_DIR / "g1" / "scene_23dof.xml"))
        spec.njmax = 200
        _names_model = spec.compile()
        body_names = [
            mujoco.mj_id2name(_names_model, mujoco.mjtObj.mjOBJ_BODY, i)
            for i in range(_names_model.nbody)
            if mujoco.mj_id2name(_names_model, mujoco.mjtObj.mjOBJ_BODY, i) != "world"
        ]
        self._nbody_tracked = len(body_names)  # 30

        # Re-parse the spec cleanly and add sensors, then do the final compile.
        spec2 = mujoco.MjSpec.from_file(str(MODELS_DIR / "g1" / "scene_23dof.xml"))
        spec2.njmax = 200
        for bname in body_names:
            sp = spec2.add_sensor()
            sp.type = mujoco.mjtSensor.mjSENS_FRAMEPOS
            sp.name = f"_bp_{bname}"
            sp.objname = bname
            sp.objtype = mujoco.mjtObj.mjOBJ_BODY
            sp.reftype = mujoco.mjtObj.mjOBJ_UNKNOWN  # world frame
        for bname in body_names:
            sq = spec2.add_sensor()
            sq.type = mujoco.mjtSensor.mjSENS_FRAMEQUAT
            sq.name = f"_bq_{bname}"
            sq.objname = bname
            sq.objtype = mujoco.mjtObj.mjOBJ_BODY
            sq.reftype = mujoco.mjtObj.mjOBJ_UNKNOWN

        mj_model = spec2.compile()
        super().__init__(
            mj_model,
            trace_sites=["imu_in_torso", "left_foot", "right_foot"],
        )

        # ── Load reference data ─────────────────────────────────────────────
        from huggingface_hub import hf_hub_download

        npz = np.load(
            hf_hub_download(
                repo_id="robfiras/loco-mujoco-datasets",
                filename=reference_filename,
                repo_type="dataset",
            )
        )

        ref_qpos = npz["qpos"].astype(np.float64)  # (T, nq)
        ref_qvel = npz["qvel"].astype(np.float64)  # (T, nv)
        self._ref_fps: float = float(npz["frequency"])
        self._loop = loop

        # Precompute reference body poses from sensordata to guarantee consistency
        # with what batch_running_cost sees during MPPI rollouts (npz xpos/xquat are empty).
        T = len(ref_qpos)
        _body_pos_len = self._nbody_tracked * 3  # 90
        _body_quat_len = self._nbody_tracked * 4  # 120
        ref_body_pos = np.zeros((T, self._nbody_tracked, 3))
        ref_body_quat = np.zeros((T, self._nbody_tracked, 4))
        ref_xipos = np.zeros((T, self._nbody_tracked, 3))  # for scalar cost
        ref_xquat_direct = np.zeros((T, self._nbody_tracked, 4))  # for scalar cost
        ref_xpos_viz = np.zeros((T, self._nbody_tracked, 3))  # body frame origins, for viz
        _tmp = mujoco.MjData(mj_model)
        for t in range(T):
            _tmp.qpos[:] = ref_qpos[t]
            _tmp.qvel[:] = ref_qvel[t]
            mujoco.mj_forward(mj_model, _tmp)
            # Read sensor values — these are what batch_running_cost will see.
            sd = _tmp.sensordata
            pos_adr = self._BODY_POS_SENSOR_ADR
            quat_adr = self._BODY_QUAT_SENSOR_ADR
            ref_body_pos[t] = sd[pos_adr : pos_adr + _body_pos_len].reshape(self._nbody_tracked, 3)
            ref_body_quat[t] = sd[quat_adr : quat_adr + _body_quat_len].reshape(self._nbody_tracked, 4)
            ref_xipos[t] = _tmp.xipos[1:]  # skip world; CoM positions for scalar cost
            ref_xquat_direct[t] = _tmp.xquat[1:]
            ref_xpos_viz[t] = _tmp.xpos[1:]  # body frame origins (joint positions) for viz

        self.ref_qpos = ref_qpos  # (T, nq)
        self.ref_qvel = ref_qvel  # (T, nv)
        self.ref_body_pos = ref_body_pos  # (T, 30, 3) — from sensor, for batch cost
        self.ref_body_quat = ref_body_quat  # (T, 30, 4) — from sensor, for batch cost
        self.ref_xpos = ref_xipos  # (T, 30, 3) — xipos (CoM), for scalar cost
        self.ref_xquat = ref_xquat_direct  # (T, 30, 4) — xquat, for scalar cost
        self.ref_xpos_viz = ref_xpos_viz  # (T, 30, 3) — body origins, for visualization
        self._n_frames = T
        self._ref_idx: int = 0

        # ── Normalised cost weights ─────────────────────────────────────────
        total = (
            options.body_position_weight
            + options.body_orientation_weight
            + options.configuration_weight
            + options.velocity_weight
        )
        self._w_xpos = options.body_position_weight / total
        self._w_xquat = options.body_orientation_weight / total
        self._w_qpos = options.configuration_weight / total
        self._w_qvel = options.velocity_weight / total

        # Cache nominal ctrl = zero (no stand keyframe in 23dof model).
        self._nominal = np.zeros(mj_model.nu)

    # ── Reference management ───────────────────────────────────────────────

    def advance(self) -> None:
        """Step the reference to the next frame.  Call once after each env.step()."""
        self._ref_idx += 1
        if self._loop:
            self._ref_idx %= self._n_frames
        else:
            self._ref_idx = min(self._ref_idx, self._n_frames - 1)

    # ── Task hooks ─────────────────────────────────────────────────────────

    def reset_data(self, data: mujoco.MjData) -> None:
        self._ref_idx = 0
        data.qpos[:] = self.ref_qpos[0]
        data.qvel[:] = self.ref_qvel[0]
        mujoco.mj_forward(self.mj_model, data)

    def nominal_ctrl(self) -> np.ndarray:
        return self._nominal.copy()

    def noise_sigma(self, cfg_sigma: float) -> np.ndarray:
        sigma = np.full(self.mj_model.nu, cfg_sigma)
        sigma[12:15] *= 0.5  # waist — limited range
        sigma[15:] *= 0.3  # arms — moderate exploration needed for mocap
        return sigma

    # ── Cost helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _exp_cost(sq_err: float) -> float:
        return 1.0 - float(np.exp(-sq_err))

    def _cost_components(self, data: mujoco.MjData) -> dict[str, float]:
        t = self._ref_idx

        qpos_err = np.sum((data.qpos - self.ref_qpos[t]) ** 2)
        qvel_err = np.sum((data.qvel - self.ref_qvel[t]) ** 2)
        xpos_err = np.sum((data.xipos[1:] - self.ref_xpos[t]) ** 2)  # skip world; xipos = CoM
        xquat_err = np.sum((data.xquat[1:] - self.ref_xquat[t]) ** 2)

        return {
            "body_position": self._w_xpos * self._exp_cost(xpos_err),
            "body_orientation": self._w_xquat * self._exp_cost(xquat_err),
            "configuration": self._w_qpos * self._exp_cost(qpos_err),
            "velocity": self._w_qvel * self._exp_cost(qvel_err),
        }

    def running_cost(self, data: mujoco.MjData, control: np.ndarray) -> float:
        return sum(self._cost_components(data).values())

    def terminal_cost(self, data: mujoco.MjData) -> float:
        return self.running_cost(data, self._nominal) * self.dt

    def cost_terms(self, data: mujoco.MjData, control: np.ndarray) -> dict[str, float]:
        return self._cost_components(data)

    def batch_running_cost(self, qpos, qvel, ctrl, sensordata, site_xpos, mocap_pos):
        """Vectorised cost over N parallel rollouts.

        Body positions and orientations come from the injected framepos/framequat
        sensors appended to the model.  Configuration and velocity come directly
        from qpos/qvel.
        """
        qpos = qpos.astype(np.float64)
        qvel = qvel.astype(np.float64)
        sd = sensordata.astype(np.float64)
        t = self._ref_idx

        # Body positions from FRAMEPOS sensors.
        body_pos_flat = sd[:, self._BODY_POS_SENSOR_ADR : self._BODY_POS_SENSOR_ADR + self._nbody_tracked * 3]
        body_pos = body_pos_flat.reshape(-1, self._nbody_tracked, 3)  # (N, 30, 3)
        xpos_err = np.sum((body_pos - self.ref_body_pos[t]) ** 2, axis=(1, 2))  # (N,)

        # Body orientations from FRAMEQUAT sensors.
        body_quat_flat = sd[:, self._BODY_QUAT_SENSOR_ADR : self._BODY_QUAT_SENSOR_ADR + self._nbody_tracked * 4]
        body_quat = body_quat_flat.reshape(-1, self._nbody_tracked, 4)  # (N, 30, 4)
        xquat_err = np.sum((body_quat - self.ref_body_quat[t]) ** 2, axis=(1, 2))  # (N,)

        # Configuration and velocity
        qpos_err = np.sum((qpos - self.ref_qpos[t]) ** 2, axis=1)  # (N,)
        qvel_err = np.sum((qvel - self.ref_qvel[t]) ** 2, axis=1)  # (N,)

        # Exponential kernel: 1 - exp(-squared_error)
        xpos_cost = 1.0 - np.exp(-xpos_err)
        xquat_cost = 1.0 - np.exp(-xquat_err)
        qpos_cost = 1.0 - np.exp(-qpos_err)
        qvel_cost = 1.0 - np.exp(-qvel_err)

        return self._w_xpos * xpos_cost + self._w_xquat * xquat_cost + self._w_qpos * qpos_cost + self._w_qvel * qvel_cost
