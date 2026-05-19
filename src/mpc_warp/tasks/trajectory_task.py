"""Trajectory-following task wrapper.

Wraps any base Task and adds a reference-tracking cost term.  References can
be loaded from:
  - a raw numpy qpos array  (``TrajectoryTask(base, ref_qpos)``)
  - a ``.npz`` file          (``TrajectoryTask.from_npz(base, path)``)

The npz format stores:
  joint_pos      (T, nj)       hinge joint positions
  joint_vel      (T, nj)       hinge joint velocities
  body_pos_w     (T, nb, 3)    world-frame body positions  (optional)
  body_quat_w    (T, nb, 4)    world-frame body orientations [w,x,y,z]  (optional)
"""
from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from .task_base import Task


class TrajectoryTask(Task):
    """Base task augmented with a qpos/qvel reference trajectory.

    Args:
        base_task:    Task whose model and base cost are reused.
        ref_qpos:     ``(T, nq)`` reference joint positions.
        ref_qvel:     ``(T, nv)`` reference velocities (optional).
        joint_qadr:   Indices of the tracked joints inside ``qpos``.
                      If None, the full qpos is tracked.
        qpos_weight:  Weight on joint-position tracking error.
        qvel_weight:  Weight on joint-velocity tracking error.
        loop:         If True the trajectory repeats cyclically.
    """

    def __init__(
        self,
        base_task: Task,
        ref_qpos: np.ndarray,
        ref_qvel: np.ndarray | None = None,
        joint_qadr: np.ndarray | None = None,
        qpos_weight: float = 1.0,
        qvel_weight: float = 0.1,
        loop: bool = True,
    ) -> None:
        # Share attributes from base task (no super().__init__ re-compile).
        self.mj_model        = base_task.mj_model
        self.u_min           = base_task.u_min
        self.u_max           = base_task.u_max
        self.dt              = base_task.dt
        self.trace_site_ids  = list(base_task.trace_site_ids)

        self._base     = base_task
        self._ref_qpos = np.asarray(ref_qpos, dtype=np.float64)   # (T, nq_tracked)
        self._ref_qvel = (
            np.asarray(ref_qvel, dtype=np.float64) if ref_qvel is not None else None
        )
        self._joint_qadr = joint_qadr   # which qpos indices to track
        self._T        = self._ref_qpos.shape[0]
        self._qpos_w   = float(qpos_weight)
        self._qvel_w   = float(qvel_weight)
        self._loop     = loop
        self._step_idx: int = 0

        # Optional 3-D body positions for visualization (from body_pos_w in npz).
        # Shape (T, 3).  Set by from_npz when body_pos_w is available.
        self.ref_body_pos: np.ndarray | None = None  # (T, 3) anchor body in world frame

    # ------------------------------------------------------------------
    # Factory: load from mjlab-format npz
    # ------------------------------------------------------------------

    @classmethod
    def from_npz(
        cls,
        base_task: Task,
        npz_path: str | Path,
        joint_names: list[str] | None = None,
        qpos_weight: float = 2.0,
        qvel_weight: float = 0.1,
        loop: bool = True,
    ) -> "TrajectoryTask":
        """Load a trajectory from a mjlab-format npz file.

        The npz must contain at least ``joint_pos`` and ``joint_vel``.
        If ``joint_names`` is given, only those joints are tracked;
        otherwise all joints in the npz are tracked in order.
        """
        data  = np.load(npz_path)
        mjm   = base_task.mj_model

        joint_pos = np.array(data["joint_pos"], dtype=np.float64)  # (T, nj)
        joint_vel = np.array(data["joint_vel"], dtype=np.float64)

        T, nj = joint_pos.shape

        # Resolve which qpos addresses to track.
        if joint_names is not None:
            qadr = np.array([int(mjm.joint(n).qposadr[0]) for n in joint_names], dtype=int)
            dadr = np.array([int(mjm.joint(n).dofadr[0])  for n in joint_names], dtype=int)
        else:
            # Assume the npz columns map to hinge joints in model order.
            hinge_joints = [
                mujoco.mj_id2name(mjm, mujoco.mjtObj.mjOBJ_JOINT, i)
                for i in range(mjm.njnt)
                if mjm.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE
            ]
            hinge_joints = [n for n in hinge_joints if n]
            use = hinge_joints[:nj]
            qadr = np.array([int(mjm.joint(n).qposadr[0]) for n in use], dtype=int)
            dadr = np.array([int(mjm.joint(n).dofadr[0])  for n in use], dtype=int)

        task = cls(
            base_task,
            ref_qpos=joint_pos,
            ref_qvel=joint_vel,
            joint_qadr=qadr,
            qpos_weight=qpos_weight,
            qvel_weight=qvel_weight,
            loop=loop,
        )

        # Store anchor-body world positions for visualization if available.
        if "body_pos_w" in data:
            body_pos_w = np.array(data["body_pos_w"], dtype=np.float64)  # (T, nb, 3)
            # Use body index 1 (first non-world body = trunk/root) as the anchor.
            anchor_idx = 1 if body_pos_w.shape[1] > 1 else 0
            task.ref_body_pos = body_pos_w[:, anchor_idx, :]  # (T, 3)

        return task

    # ------------------------------------------------------------------
    # Trajectory state management
    # ------------------------------------------------------------------

    def ref_positions_window(self, n: int) -> np.ndarray | None:
        """Return the next ``n`` reference anchor-body positions as ``(n, 3)``.

        Returns None if ``ref_body_pos`` is not set.
        Wraps cyclically when ``loop=True``.
        """
        if self.ref_body_pos is None:
            return None
        indices = [(self._step_idx + i) % self._T for i in range(n)]
        return self.ref_body_pos[indices]

    def advance(self) -> None:
        """Advance the trajectory index by one step.  Call after each env.step()."""
        if self._loop:
            self._step_idx = (self._step_idx + 1) % self._T
        else:
            self._step_idx = min(self._step_idx + 1, self._T - 1)

    def reset(self) -> None:
        self._step_idx = 0

    @property
    def ref_qpos_now(self) -> np.ndarray:
        return self._ref_qpos[self._step_idx]

    @property
    def ref_qvel_now(self) -> np.ndarray | None:
        if self._ref_qvel is None:
            return None
        return self._ref_qvel[self._step_idx]

    # ------------------------------------------------------------------
    # Cost
    # ------------------------------------------------------------------

    def _tracking_cost(self, data: mujoco.MjData) -> float:
        ref_q = self.ref_qpos_now
        if self._joint_qadr is not None:
            cur_q = data.qpos[self._joint_qadr]
        else:
            cur_q = data.qpos[: len(ref_q)]
        qpos_err = float(np.sum((cur_q - ref_q) ** 2))
        cost = self._qpos_w * qpos_err

        if self._ref_qvel is not None and self.ref_qvel_now is not None:
            ref_v = self.ref_qvel_now
            cur_v = data.qvel[: len(ref_v)]
            cost += self._qvel_w * float(np.sum((cur_v - ref_v) ** 2))
        return cost

    def running_cost(self, data: mujoco.MjData, control: np.ndarray) -> float:
        return self._base.running_cost(data, control) + self._tracking_cost(data)

    def terminal_cost(self, data: mujoco.MjData) -> float:
        return self._base.terminal_cost(data) + 5.0 * self._tracking_cost(data)

    def batch_running_cost(self, qpos, qvel, ctrl, sensordata, site_xpos, mocap_pos):
        base_costs = self._base.batch_running_cost(
            qpos, qvel, ctrl, sensordata, site_xpos, mocap_pos
        )
        ref_q = self.ref_qpos_now.astype(np.float64)
        if self._joint_qadr is not None:
            cur_q = qpos[:, self._joint_qadr].astype(np.float64)
        else:
            cur_q = qpos[:, : len(ref_q)].astype(np.float64)
        tracking = self._qpos_w * np.sum((cur_q - ref_q) ** 2, axis=1)
        if self._ref_qvel is not None and self.ref_qvel_now is not None:
            ref_v = self.ref_qvel_now.astype(np.float64)
            cur_v = qvel[:, : len(ref_v)].astype(np.float64)
            tracking += self._qvel_w * np.sum((cur_v - ref_v) ** 2, axis=1)
        return base_costs + tracking
