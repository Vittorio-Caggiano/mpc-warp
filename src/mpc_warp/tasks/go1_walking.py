"""Unitree GO1 quadruped walking task."""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from .task_base import Task

_GO1_XML = Path(__file__).parent.parent / "models" / "go1" / "go1.xml"

# Joint name order matches the floating-base XML (first joint is free).
_HINGE_JOINTS = [
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint",
    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",
    "RR_hip_joint",
    "RR_thigh_joint",
    "RR_calf_joint",
    "RL_hip_joint",
    "RL_thigh_joint",
    "RL_calf_joint",
]

# Default standing pose (qpos indices 7..18, matching _HINGE_JOINTS order).
_STAND_QPOS = np.array(
    [
        -0.1,
        0.9,
        -1.8,  # FR: hip, thigh, calf
        0.1,
        0.9,
        -1.8,  # FL
        -0.1,
        0.9,
        -1.8,  # RR
        0.1,
        0.9,
        -1.8,  # RL
    ],
    dtype=np.float64,
)


def _build_go1_model() -> mujoco.MjModel:
    """Build a GO1 model with floor and torque actuators."""
    spec = mujoco.MjSpec.from_file(str(_GO1_XML))

    # Add floor.
    floor = spec.worldbody.add_geom()
    floor.type = mujoco.mjtGeom.mjGEOM_PLANE
    floor.size[:] = [0.0, 0.0, 0.05]
    floor.rgba[:] = [0.5, 0.5, 0.5, 1.0]

    # Add one torque actuator per hinge joint.
    for jnt in spec.joints:
        if jnt.type == mujoco.mjtJoint.mjJNT_HINGE:
            act = spec.add_actuator()
            act.name = jnt.name + "_act"
            act.trntype = mujoco.mjtTrn.mjTRN_JOINT
            act.target = jnt.name
            # gain = 1 (direct torque)
            gainp = np.zeros(10)
            gainp[0] = 1.0
            act.gainprm = gainp
            act.biastype = mujoco.mjtBias.mjBIAS_NONE
            # Torque limits ±35 Nm (conservative)
            act.ctrllimited = True
            act.ctrlrange[:] = [-35.0, 35.0]

    # mujoco_warp pre-allocates njmax slots; 4 feet × ~17 contacts each needs headroom.
    spec.njmax = 100

    return spec.compile()


class Go1Walking(Task):
    """Unitree GO1 quadruped — walk forward at target velocity."""

    def __init__(self, target_velocity: float = 0.5) -> None:
        mj_model = _build_go1_model()
        super().__init__(mj_model, trace_sites=["imu"])

        self._trunk_id = mj_model.body("trunk").id
        self._imu_id = mj_model.site("imu").id
        self.target_velocity = target_velocity
        self.target_height = 0.278  # default standing height
        # Map joint name → qpos index.
        self._joint_qadr = np.array([int(mj_model.joint(name).qposadr[0]) for name in _HINGE_JOINTS], dtype=int)

        # Pre-compute gravity-compensation torques at the stand pose.
        _kd = mujoco.MjData(mj_model)
        self._reset_to_stand(_kd)
        _kd.qacc[:] = 0.0
        mujoco.mj_inverse(mj_model, _kd)
        act_qadr = np.array([mj_model.jnt_dofadr[mj_model.actuator_trnid[i, 0]] for i in range(mj_model.nu)], dtype=int)
        grav = _kd.qfrc_inverse[act_qadr]
        self._grav_comp = np.clip(grav, self.u_min, self.u_max)

    def _reset_to_stand(self, data: mujoco.MjData) -> None:
        """Set data to a stable standing configuration."""
        mujoco.mj_resetData(self.mj_model, data)
        data.qpos[0:3] = [0.0, 0.0, 0.278]
        data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]  # identity quaternion
        data.qpos[self._joint_qadr] = _STAND_QPOS
        mujoco.mj_forward(self.mj_model, data)

    def reset_data(self, data: mujoco.MjData) -> None:
        self._reset_to_stand(data)

    def nominal_ctrl(self) -> np.ndarray:
        """Gravity-compensation torques at the stand pose."""
        return self._grav_comp.copy()

    def noise_sigma(self, cfg_sigma: float) -> np.ndarray:
        """Scale noise by actuator range: torque actuators need physical-unit exploration."""
        u_range = np.where(np.isfinite(self.u_max - self.u_min), self.u_max - self.u_min, 2.0)
        return cfg_sigma * u_range

    def _trunk_height(self, data: mujoco.MjData) -> float:
        return float(data.qpos[2])

    def _trunk_velocity_x(self, data: mujoco.MjData) -> float:
        return float(data.qvel[0])

    def _upright_cost(self, data: mujoco.MjData) -> float:
        # quat = [w, x, y, z]; cost = 1 - w² penalises tilt
        w = float(data.qpos[3])
        return 1.0 - w * w

    def _cost_components(self, data: mujoco.MjData, control: np.ndarray) -> dict[str, float]:
        return {
            "velocity": 1.0 * (self._trunk_velocity_x(data) - self.target_velocity) ** 2,
            "lateral": 0.5 * float(data.qvel[1]) ** 2,
            "yaw": 0.5 * float(data.qvel[5]) ** 2,
            "height": 5.0 * (self._trunk_height(data) - self.target_height) ** 2,
            "orient": 3.0 * self._upright_cost(data),
            "pose": 0.1 * float(np.sum((data.qpos[self._joint_qadr] - _STAND_QPOS) ** 2)),
            "control": 1e-3 * float(np.sum(control**2)),
        }

    def running_cost(self, data: mujoco.MjData, control: np.ndarray) -> float:
        return sum(self._cost_components(data, control).values())

    def cost_terms(self, data: mujoco.MjData, control: np.ndarray) -> dict[str, float]:
        return self._cost_components(data, control)

    def terminal_cost(self, data: mujoco.MjData) -> float:
        vel_cost = (self._trunk_velocity_x(data) - self.target_velocity) ** 2
        height_cost = (self._trunk_height(data) - self.target_height) ** 2
        orient_cost = self._upright_cost(data)
        return 2.0 * vel_cost + 10.0 * height_cost + 5.0 * orient_cost

    def batch_running_cost(self, qpos, qvel, ctrl, sensordata, site_xpos, mocap_pos):
        qpos = qpos.astype(np.float64)
        qvel = qvel.astype(np.float64)
        vel_cost = 1.0 * (qvel[:, 0] - self.target_velocity) ** 2
        lateral_cost = 0.5 * qvel[:, 1] ** 2
        yaw_cost = 0.5 * qvel[:, 5] ** 2
        height_cost = 5.0 * (qpos[:, 2] - self.target_height) ** 2
        orient_cost = 3.0 * (1.0 - qpos[:, 3] ** 2)  # 1 - w²
        pose_cost = 0.1 * np.sum((qpos[:, self._joint_qadr] - _STAND_QPOS) ** 2, axis=1)
        ctrl_cost = 1e-3 * np.sum(ctrl**2, axis=1)
        return vel_cost + lateral_cost + yaw_cost + height_cost + orient_cost + pose_cost + ctrl_cost
