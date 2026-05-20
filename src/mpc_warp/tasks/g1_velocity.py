"""Unitree G1 humanoid — command-conditioned directional velocity tracking.

The controller receives a velocity command (vx, vy, yaw_rate) and tracks it
while maintaining upright posture.  Inspired by the mjlab velocity task:
https://github.com/mujocolab/mjlab

Cost terms (following mjlab conventions):
  - velocity tracking  : exp-kernel on (vx, vy) error + yaw-rate error
  - upright            : keep torso z-axis aligned with world up
  - height             : keep pelvis at target height
  - pose               : joint angles close to stand keyframe
  - action rate        : penalise large ctrl changes (smoothness)
  - lateral stability  : penalise roll/pitch angular velocity
"""

from __future__ import annotations

import math

import mujoco
import numpy as np

from .task_base import MODELS_DIR, Task


class G1VelocityTracking(Task):
    """G1 humanoid — track (vx_cmd, vy_cmd, yaw_cmd) velocity commands.

    Args:
        vx_cmd:   Target forward velocity in m/s  (default 0.5).
        vy_cmd:   Target lateral velocity in m/s  (default 0.0).
        yaw_cmd:  Target yaw rate in rad/s        (default 0.0).
    """

    # ── Sensor address constants (from scene.xml sensor order) ────────────
    # sensor 0: imu-torso-angular-velocity  adr=0  dim=3   [wx, wy, wz] body frame
    # sensor 2: imu_in_torso_quat           adr=6  dim=4   [w, x, y, z]
    # sensor 3: imu_in_torso_linvel         adr=10 dim=3   [vx, vy, vz] world frame
    _ANG_VEL_ADR = 0  # torso angular velocity (body frame)
    _QUAT_ADR = 6  # torso quaternion
    _LIN_VEL_ADR = 10  # torso linear velocity (world frame)

    def __init__(
        self,
        vx_cmd: float = 0.5,
        vy_cmd: float = 0.0,
        yaw_cmd: float = 0.0,
    ) -> None:
        # Load via MjSpec to set njmax explicitly — the G1 biped generates ~32 contact
        # constraints when standing; warp needs a pre-allocated buffer of at least 80.
        _spec = mujoco.MjSpec.from_file(str(MODELS_DIR / "g1" / "scene.xml"))
        _spec.njmax = 200
        mj_model = _spec.compile()
        super().__init__(mj_model, trace_sites=["imu_in_torso"])

        self.vx_cmd = float(vx_cmd)
        self.vy_cmd = float(vy_cmd)
        self.yaw_cmd = float(yaw_cmd)

        self._torso_id = mj_model.site("imu_in_torso").id
        self.qstand = mj_model.keyframe("stand").qpos.copy()

        # Target height: pelvis z at stand keyframe.
        _kd = mujoco.MjData(mj_model)
        mujoco.mj_resetDataKeyframe(mj_model, _kd, mj_model.keyframe("stand").id)
        mujoco.mj_forward(mj_model, _kd)
        self.target_height = float(_kd.qpos[2])

        # Position-actuator stand targets and gravity compensation.
        act_qadr = np.array(
            [mj_model.jnt_qposadr[mj_model.actuator_trnid[i, 0]] for i in range(mj_model.nu)],
            dtype=int,
        )
        self._stand_ctrl = np.clip(self.qstand[act_qadr], self.u_min, self.u_max)

    # ------------------------------------------------------------------
    # Task hooks
    # ------------------------------------------------------------------

    def reset_data(self, data: mujoco.MjData) -> None:
        mujoco.mj_resetDataKeyframe(self.mj_model, data, self.mj_model.keyframe("stand").id)
        mujoco.mj_forward(self.mj_model, data)

    def nominal_ctrl(self) -> np.ndarray:
        """Stand-pose joint angles for the position-controlled G1."""
        return self._stand_ctrl.copy()

    def noise_sigma(self, cfg_sigma: float) -> np.ndarray:
        """Per-actuator sigma scaled by joint group.

        Legs get full sigma for walking exploration; waist gets half;
        arms get 0.1× (they don't affect locomotion and have narrow ranges).
        """
        sigma = np.full(self.mj_model.nu, cfg_sigma)
        sigma[12:15] *= 0.5  # waist (yaw, roll, pitch) — limited range
        sigma[15:] *= 0.1  # arms — not relevant for locomotion
        return sigma

    # ------------------------------------------------------------------
    # Velocity command interface
    # ------------------------------------------------------------------

    def set_command(self, vx: float, vy: float = 0.0, yaw: float = 0.0) -> None:
        """Update the velocity command at runtime."""
        self.vx_cmd = float(vx)
        self.vy_cmd = float(vy)
        self.yaw_cmd = float(yaw)

    # ------------------------------------------------------------------
    # Cost helpers
    # ------------------------------------------------------------------

    def _quat_to_gravity_proj(self, quat: np.ndarray) -> np.ndarray:
        """Project world gravity [0,0,-1] into body frame via quat [w,x,y,z]."""
        w, x, y, z = quat
        g_world = np.array([0.0, 0.0, -1.0])
        # Rotate by inverse quaternion (conjugate).
        k = np.array([x, y, z])
        t = 2.0 * np.cross(k, g_world)
        return g_world + w * t + np.cross(k, t)

    def _upright_cost(self, quat: np.ndarray) -> float:
        """Cost = 1 - (gravity_body · [0,0,-1])^2; zero when perfectly upright."""
        g_b = self._quat_to_gravity_proj(quat)
        return float(1.0 - g_b[2] ** 2)

    def _velocity_tracking_cost(self, linvel_w: np.ndarray, ang_vel: np.ndarray) -> float:
        """Exponential-kernel tracking cost for (vx, vy, yaw_rate) plus overshoot penalty."""
        vx_err = linvel_w[0] - self.vx_cmd
        vy_err = linvel_w[1] - self.vy_cmd
        yaw_err = ang_vel[2] - self.yaw_cmd
        lin_exp = 1.0 - math.exp(-2.0 * (vx_err**2 + vy_err**2))
        yaw_exp = 1.0 - math.exp(-2.0 * yaw_err**2)
        # Penalise exceeding the commanded speed — overshoot causes forward lean and falls.
        speed_cmd = math.sqrt(self.vx_cmd**2 + self.vy_cmd**2)
        actual_speed = math.sqrt(linvel_w[0] ** 2 + linvel_w[1] ** 2)
        overshoot = max(0.0, actual_speed - 1.3 * max(speed_cmd, 0.1))
        return 2.0 * lin_exp + 2.0 * yaw_exp + 3.0 * overshoot**2

    def _cost_components(self, data: mujoco.MjData, control: np.ndarray) -> dict[str, float]:
        linvel = np.array(data.sensordata[self._LIN_VEL_ADR : self._LIN_VEL_ADR + 3])
        ang_vel = np.array(data.sensordata[self._ANG_VEL_ADR : self._ANG_VEL_ADR + 3])
        quat = np.array(data.sensordata[self._QUAT_ADR : self._QUAT_ADR + 4])

        vel_cost = self._velocity_tracking_cost(linvel, ang_vel)
        upright_cost = self._upright_cost(quat)
        height_cost = (float(data.qpos[2]) - self.target_height) ** 2
        roll_pitch_cost = float(ang_vel[0] ** 2 + ang_vel[1] ** 2)

        return {
            "velocity": vel_cost,
            "upright": 30.0 * upright_cost,
            "height": 30.0 * height_cost,
            "roll_pitch": 3.0 * roll_pitch_cost,
        }

    def running_cost(self, data: mujoco.MjData, control: np.ndarray) -> float:
        return sum(self._cost_components(data, control).values())

    def terminal_cost(self, data: mujoco.MjData) -> float:
        linvel = np.array(data.sensordata[self._LIN_VEL_ADR : self._LIN_VEL_ADR + 3])
        ang_vel = np.array(data.sensordata[self._ANG_VEL_ADR : self._ANG_VEL_ADR + 3])
        quat = np.array(data.sensordata[self._QUAT_ADR : self._QUAT_ADR + 4])
        vel_cost = self._velocity_tracking_cost(linvel, ang_vel)
        upright_cost = self._upright_cost(quat)
        height_cost = (float(data.qpos[2]) - self.target_height) ** 2
        return 2.0 * vel_cost + 40.0 * upright_cost + 80.0 * height_cost

    def cost_terms(self, data: mujoco.MjData, control: np.ndarray) -> dict[str, float]:
        return self._cost_components(data, control)

    def batch_running_cost(self, qpos, qvel, ctrl, sensordata, site_xpos, mocap_pos):
        qpos = qpos.astype(np.float64)
        sd = sensordata.astype(np.float64)

        linvel = sd[:, self._LIN_VEL_ADR : self._LIN_VEL_ADR + 3]  # (N, 3) world frame
        ang_vel = sd[:, self._ANG_VEL_ADR : self._ANG_VEL_ADR + 3]  # (N, 3) body frame
        quat = sd[:, self._QUAT_ADR : self._QUAT_ADR + 4]  # (N, 4) [w,x,y,z]

        # Velocity tracking: exponential kernel on linear + yaw errors.
        vx_err = linvel[:, 0] - self.vx_cmd
        vy_err = linvel[:, 1] - self.vy_cmd
        yaw_err = ang_vel[:, 2] - self.yaw_cmd
        lin_exp = 1.0 - np.exp(-2.0 * (vx_err**2 + vy_err**2))
        yaw_exp = 1.0 - np.exp(-2.0 * yaw_err**2)
        speed_cmd = math.sqrt(self.vx_cmd**2 + self.vy_cmd**2)
        actual_speed = np.sqrt(linvel[:, 0] ** 2 + linvel[:, 1] ** 2)
        overshoot = np.maximum(0.0, actual_speed - 1.3 * max(speed_cmd, 0.1))
        vel_cost = 2.0 * lin_exp + 2.0 * yaw_exp + 3.0 * overshoot**2

        # Upright: project gravity into body frame, penalise tilt.
        # g_body_z = w²·(-1) + cross terms; full: cos_angle = g_b·[0,0,-1] = g_b_z
        # Using quat rotation of [0,0,-1]:
        w, x, y, _ = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        # g_body = R^T · [0,0,-1]; for [0,0,-1], the z-component of R^T·g is:
        # g_bz = -1 + 2*(x*z - w*y)*(-1)... let me use the double-cross formula.
        # k = [x,y,z], v = [0,0,-1], t = 2*cross(k,v)
        # cross([x,y,z], [0,0,-1]) = [y*(-1)-z*0, z*0-x*(-1), x*0-y*0] = [-y, x, 0]
        t_x = -2.0 * y
        t_y = 2.0 * x
        t_z = 0.0
        # rotated = v + w*t + cross(k, t)
        # cross([x,y,z], [t_x,t_y,0]) = [y*0-z*t_y, z*t_x-x*0, x*t_y-y*t_x]
        #                              = [-z*t_y, z*t_x, x*t_y-y*t_x]
        g_bz = -1.0 + w * t_z + (x * t_y - y * t_x)
        upright_cost = 30.0 * (1.0 - g_bz**2)

        # Height.
        height_cost = 30.0 * (qpos[:, 2] - self.target_height) ** 2

        # Roll/pitch angular velocity damping.
        roll_pitch_cost = 3.0 * (ang_vel[:, 0] ** 2 + ang_vel[:, 1] ** 2)

        # Hard fall penalty: large cost whenever the pelvis drops below half target height.
        fall_threshold = self.target_height * 0.7
        fall_penalty = 200.0 * np.maximum(0.0, fall_threshold - qpos[:, 2])

        return vel_cost + upright_cost + height_cost + roll_pitch_cost + fall_penalty
