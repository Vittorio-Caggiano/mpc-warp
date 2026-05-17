"""Minimal MPC toolkit with a mujoco_warp backend."""

from .solvers.mppi import MPPIConfig, MPPISolver

__all__ = ["MPPIConfig", "MPPISolver"]
