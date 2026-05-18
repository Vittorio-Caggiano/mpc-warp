"""Generate a GO1 trot-gait reference trajectory and save as npz.

The reference is produced by driving the GO1's joint angles through a CPG
(central pattern generator) sinusoidal trot pattern inside MuJoCo physics,
then recording the resulting body and joint states.

Usage (standalone):
    uv run python -m mpc_warp.tasks.utils.go1_trot_reference
    # writes: data/go1_trot.npz
"""
from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

# Default output path (relative to repo root).
DEFAULT_OUT = Path(__file__).parents[5] / "data" / "go1_trot.npz"

# Joint order in the GO1 model (hinge joints only, after floating-base joint).
JOINT_NAMES = [
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
]

# Default standing pose for each joint.
_STAND = np.array([
    -0.1,  0.9, -1.8,   # FR
     0.1,  0.9, -1.8,   # FL
    -0.1,  0.9, -1.8,   # RR
     0.1,  0.9, -1.8,   # RL
])

# CPG parameters for trot gait.
# Trot: (FR, RL) and (FL, RR) swing in phase.
_FREQ       = 1.5   # gait cycles per second
_HIP_AMP    = 0.05  # hip lateral swing (rad)
_THIGH_AMP  = 0.35  # thigh lift amplitude (rad)
_CALF_AMP   = 0.40  # calf flexion amplitude (rad)


def _trot_joint_targets(t: float) -> np.ndarray:
    """Return 12 joint position targets for trot gait at time t."""
    phi = 2.0 * np.pi * _FREQ * t
    # FR and RL are in-phase; FL and RR are anti-phase.
    swing_a = np.sin(phi)        # FR, RL
    swing_b = np.sin(phi + np.pi)  # FL, RR

    q = _STAND.copy()
    # FR (indices 0-2): hip, thigh, calf
    q[0] += _HIP_AMP * swing_a
    q[1] += _THIGH_AMP * max(swing_a, 0.0)
    q[2] += -_CALF_AMP * max(swing_a, 0.0)
    # FL (indices 3-5)
    q[3] += -_HIP_AMP * swing_b
    q[4] += _THIGH_AMP * max(swing_b, 0.0)
    q[5] += -_CALF_AMP * max(swing_b, 0.0)
    # RR (indices 6-8)
    q[6] += _HIP_AMP * swing_b
    q[7] += _THIGH_AMP * max(swing_b, 0.0)
    q[8] += -_CALF_AMP * max(swing_b, 0.0)
    # RL (indices 9-11)
    q[9]  += -_HIP_AMP * swing_a
    q[10] += _THIGH_AMP * max(swing_a, 0.0)
    q[11] += -_CALF_AMP * max(swing_a, 0.0)
    return q


def generate(
    n_cycles: int = 4,
    output_path: Path | str | None = None,
    kp: float = 80.0,   # position gain for the PD tracking controller
    kd: float = 5.0,
) -> Path:
    """Simulate GO1 following a CPG trot reference and record the trajectory.

    Returns the path to the saved npz file.
    """
    from mpc_warp.tasks.go1_walking import _build_go1_model

    mjm = _build_go1_model()
    mjd = mujoco.MjData(mjm)

    # Build joint index arrays.
    joint_qadr = np.array([int(mjm.joint(n).qposadr[0]) for n in JOINT_NAMES], dtype=int)
    joint_dadr = np.array([int(mjm.joint(n).dofadr[0])  for n in JOINT_NAMES], dtype=int)
    act_ids    = np.arange(mjm.nu)   # one actuator per hinge joint (same order)

    # Initial pose.
    mjd.qpos[2] = 0.278
    mjd.qpos[3] = 1.0
    mjd.qpos[joint_qadr] = _STAND
    mujoco.mj_forward(mjm, mjd)

    dt         = float(mjm.opt.timestep)
    duration   = float(n_cycles) / _FREQ
    n_steps    = int(duration / dt)

    # Pre-allocate log arrays.
    joint_pos     = np.zeros((n_steps, len(JOINT_NAMES)))
    joint_vel     = np.zeros((n_steps, len(JOINT_NAMES)))
    body_pos_w    = np.zeros((n_steps, mjm.nbody, 3))
    body_quat_w   = np.zeros((n_steps, mjm.nbody, 4))
    body_lin_vel_w = np.zeros((n_steps, mjm.nbody, 3))
    body_ang_vel_w = np.zeros((n_steps, mjm.nbody, 3))

    for i in range(n_steps):
        t = i * dt
        q_ref = _trot_joint_targets(t)
        q_cur = mjd.qpos[joint_qadr]
        v_cur = mjd.qvel[joint_dadr]

        # PD torque command.
        tau = kp * (q_ref - q_cur) - kd * v_cur
        tau = np.clip(tau, -35.0, 35.0)
        mjd.ctrl[act_ids] = tau

        mujoco.mj_step(mjm, mjd)

        # Record.
        joint_pos[i]      = mjd.qpos[joint_qadr]
        joint_vel[i]      = mjd.qvel[joint_dadr]
        body_pos_w[i]     = mjd.xpos          # (nbody, 3)
        body_quat_w[i]    = mjd.xquat         # (nbody, 4) [w,x,y,z]
        # body velocities: cvel is in local frame; use subtree_linvel for world-frame
        for b in range(mjm.nbody):
            body_lin_vel_w[i, b] = mjd.cvel[b, 3:6]   # translational (world-frame)
            body_ang_vel_w[i, b] = mjd.cvel[b, 0:3]   # rotational (world-frame)

    out = Path(output_path) if output_path else DEFAULT_OUT
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        fps=np.array([1.0 / dt]),
        joint_pos=joint_pos,
        joint_vel=joint_vel,
        body_pos_w=body_pos_w,
        body_quat_w=body_quat_w,
        body_lin_vel_w=body_lin_vel_w,
        body_ang_vel_w=body_ang_vel_w,
    )
    print(f"Saved {n_steps} frames ({duration:.1f}s) to {out}")
    return out


if __name__ == "__main__":
    generate()
