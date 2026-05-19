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
        mj_model = mujoco.MjModel.from_xml_path(str(MODELS_DIR / "g1" / "scene.xml"))
        super().__init__(mj_model, trace_sites=["imu_in_torso"])

        self._orient_adr = int(mj_model.sensor_adr[mj_model.sensor("imu_in_torso_quat").id])
        self._torso_id = mj_model.site("imu_in_torso").id
        self.qstand = mj_model.keyframe("stand").qpos.copy()

        # Compute target height from the stand keyframe rather than hard-coding.
        _kd = mujoco.MjData(mj_model)
        mujoco.mj_resetDataKeyframe(mj_model, _kd, mj_model.keyframe("stand").id)
        mujoco.mj_forward(mj_model, _kd)
        self.target_height = float(_kd.site_xpos[self._torso_id, 2])

        # The G1 uses position actuators (ctrl = target joint angle); nominal is stand pose.
        act_qadr = np.array([mj_model.jnt_qposadr[mj_model.actuator_trnid[i, 0]] for i in range(mj_model.nu)], dtype=int)
        self._stand_ctrl = np.clip(self.qstand[act_qadr], self.u_min, self.u_max)

    def reset_data(self, data: mujoco.MjData) -> None:
        mujoco.mj_resetDataKeyframe(self.mj_model, data, self.mj_model.keyframe("stand").id)
        mujoco.mj_forward(self.mj_model, data)

    def nominal_ctrl(self) -> np.ndarray:
        """Stand-pose joint angles for the position-controlled G1."""
        return self._stand_ctrl.copy()

    def noise_sigma(self, cfg_sigma: float) -> np.ndarray:
        """Position-controlled joints: sigma in radians, scaled down to avoid saturating PD."""
        return np.full(self.mj_model.nu, cfg_sigma * 0.2)

    def _torso_height(self, data: mujoco.MjData) -> float:
        return float(data.site_xpos[self._torso_id, 2])

    def _torso_orientation_cost(self, data: mujoco.MjData) -> float:
        quat = np.array(data.sensordata[self._orient_adr : self._orient_adr + 4])
        upright = np.array([0.0, 0.0, 1.0])
        rotated = _quat_rotate(upright, quat)
        # Measures how far the body z-axis deviates from world-up; 0 when upright.
        return float((rotated[2] - 1.0) ** 2)

    def running_cost(self, data: mujoco.MjData, control: np.ndarray) -> float:
        orientation_cost = self._torso_orientation_cost(data)
        height_cost = (self._torso_height(data) - self.target_height) ** 2
        nominal_cost = float(np.sum((np.array(data.qpos[7:]) - self.qstand[7:]) ** 2))
        # Penalise deviation from stand-pose targets to prevent nominal trajectory drift.
        ctrl_reg = float(np.sum((control - self._stand_ctrl) ** 2))
        return 10.0 * orientation_cost + 10.0 * height_cost + 1.0 * nominal_cost + 0.5 * ctrl_reg

    def terminal_cost(self, data: mujoco.MjData) -> float:
        return self.running_cost(data, np.zeros(self.mj_model.nu))

    def batch_running_cost(self, qpos, qvel, ctrl, sensordata, site_xpos, mocap_pos):
        # Vectorised quat rotation: q (N,4), v (3,) -> (N,3)
        q = sensordata[:, self._orient_adr : self._orient_adr + 4].astype(np.float64)
        upright = np.array([0.0, 0.0, 1.0])
        k = q[:, 1:]  # (N, 3) [x,y,z]
        t = 2.0 * np.cross(k, upright[None])  # (N, 3)
        rotated = upright[None] + q[:, :1] * t + np.cross(k, t)
        orientation_cost = (rotated[:, 2] - 1.0) ** 2

        height = site_xpos[:, self._torso_id, 2].astype(np.float64)
        height_cost = (height - self.target_height) ** 2

        joints = qpos[:, 7:].astype(np.float64)
        nominal_cost = np.sum((joints - self.qstand[7:]) ** 2, axis=1)

        ctrl_reg = 0.5 * np.sum((ctrl.astype(np.float64) - self._stand_ctrl) ** 2, axis=1)

        return 10.0 * orientation_cost + 10.0 * height_cost + 1.0 * nominal_cost + ctrl_reg
