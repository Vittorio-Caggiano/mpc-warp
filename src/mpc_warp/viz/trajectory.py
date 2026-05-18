"""MJPC-style viewer overlays: past trace, planned trajectory, and figure panels.

3-D geometry (traces, planned path, reference markers) goes into viewer.user_scn.
Scalar panels (cost history, cost terms bar chart, action horizon, ESS) use
viewer.set_figures() with MjvFigure — the same mechanism as mujoco_mpc's simulate.
Scalar HUD text (cost, ESS) uses viewer.set_texts().
"""
from __future__ import annotations

from collections import deque

import mujoco
import numpy as np

from mpc_warp.tasks.task_base import Task

# Geometry constants
_SPHERE = mujoco.mjtGeom.mjGEOM_SPHERE
_LINE   = mujoco.mjtGeom.mjGEOM_LINE
_BOX    = mujoco.mjtGeom.mjGEOM_BOX

_IDENTITY_MAT = np.eye(3).flatten()

# Maximum data points per MjvFigure line.
_FIGURE_MAXPNT = 1000

# Panel colours (RGB, 0-1)
_COLORS = [
    (0.9, 0.3, 0.3),   # red
    (0.3, 0.6, 0.9),   # blue
    (0.3, 0.9, 0.4),   # green
    (0.9, 0.7, 0.2),   # amber
    (0.8, 0.3, 0.9),   # purple
    (0.9, 0.6, 0.3),   # orange
    (0.2, 0.9, 0.8),   # teal
    (0.9, 0.9, 0.3),   # yellow
]


def _add_sphere(scn: mujoco.MjvScene, pos: np.ndarray, radius: float, rgba: np.ndarray) -> bool:
    if scn.ngeom >= scn.maxgeom:
        return False
    g = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(g, _SPHERE, np.array([radius, radius, radius]),
                        pos.astype(np.float64), _IDENTITY_MAT, rgba.astype(np.float32))
    scn.ngeom += 1
    return True


def _add_line(scn: mujoco.MjvScene, a: np.ndarray, b: np.ndarray,
              width: float, rgba: np.ndarray) -> bool:
    if scn.ngeom >= scn.maxgeom:
        return False
    g = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(g, _LINE, np.zeros(3), np.zeros(3), _IDENTITY_MAT,
                        rgba.astype(np.float32))
    mujoco.mjv_connector(g, _LINE, width,
                         a.astype(np.float64), b.astype(np.float64))
    scn.ngeom += 1
    return True


def _make_figure(title: str, xlabel: str = "step") -> mujoco.MjvFigure:
    fig = mujoco.MjvFigure()
    mujoco.mjv_defaultFigure(fig)
    fig.title = title
    fig.xlabel = xlabel
    fig.flg_legend = 1
    fig.flg_extend = 1
    fig.flg_ticklabel = 1
    return fig


def _figure_append(fig: mujoco.MjvFigure, line_idx: int,
                   x: float, y: float) -> None:
    """Append a single (x, y) point to a figure line (circular buffer)."""
    n = int(fig.linepnt[line_idx])
    if n >= _FIGURE_MAXPNT:
        # Shift left by one to make room.
        fig.linedata[line_idx, :2 * (_FIGURE_MAXPNT - 1)] = \
            fig.linedata[line_idx, 2:2 * _FIGURE_MAXPNT]
        n = _FIGURE_MAXPNT - 1
    fig.linedata[line_idx, 2 * n]     = x
    fig.linedata[line_idx, 2 * n + 1] = y
    fig.linepnt[line_idx] = n + 1


def _figure_set_line(fig: mujoco.MjvFigure, line_idx: int,
                     xs: np.ndarray, ys: np.ndarray) -> None:
    """Replace a figure line with new data arrays."""
    n = min(len(xs), _FIGURE_MAXPNT)
    for i in range(n):
        fig.linedata[line_idx, 2 * i]     = float(xs[i])
        fig.linedata[line_idx, 2 * i + 1] = float(ys[i])
    fig.linepnt[line_idx] = n


class TrajectoryViz:
    """MJPC-style overlays for the mujoco passive viewer.

    3-D overlays are written into ``viewer.user_scn`` (inside ``viewer.lock()``).
    Panels (cost history, cost terms, action horizon) are passed to
    ``viewer.set_figures()`` each step.
    HUD text (cost, ESS) is passed to ``viewer.set_texts()``.

    Usage::

        viz = TrajectoryViz(task)

        with viewer_ctx.lock():
            viz.update_scene(env.data, planned_sites, ref_positions, viewer_ctx.user_scn)

        figs = viz.build_figures(last_cost, cost_weights, u, cost_terms, u_nominal)
        viewer_ctx.set_figures(figs)

        viewer_ctx.set_texts(viz.build_texts(last_cost, cost_weights))
    """

    def __init__(self, task: Task, max_trace_len: int = 200,
                 max_cost_history: int = 500) -> None:
        self._task = task
        n = len(task.trace_site_ids)
        self._traces: list[deque[np.ndarray]] = [
            deque(maxlen=max_trace_len) for _ in range(n)
        ]
        self._step = 0

        # ── Cost history figure (top-left panel) ─────────────────────────
        self._fig_cost = _make_figure("Running cost", xlabel="step")
        self._fig_cost.linergb[0] = [0.9, 0.3, 0.3]
        self._fig_cost.linename[0] = b"cost"
        self._fig_cost.linewidth = 2.0

        # ── Cost terms bar chart (top-right panel) ────────────────────────
        self._fig_terms = _make_figure("Cost terms")
        self._fig_terms.flg_barplot = 1

        # ── Action horizon figure (bottom-left panel) ─────────────────────
        self._fig_actions = _make_figure("Nominal actions", xlabel="horizon step")

        # Actuator names (populated on first update).
        self._act_names_set = False

    # ------------------------------------------------------------------
    # Main entry points
    # ------------------------------------------------------------------

    def update_scene(
        self,
        data: mujoco.MjData,
        planned_sites: np.ndarray | None,
        ref_positions: np.ndarray | None,
        viewer_scn: mujoco.MjvScene,
    ) -> None:
        """Update 3-D user_scn overlays.  Call inside viewer.lock()."""
        for k, sid in enumerate(self._task.trace_site_ids):
            self._traces[k].append(data.site_xpos[sid].copy())

        viewer_scn.ngeom = 0

        if ref_positions is not None:
            self._draw_reference(ref_positions, viewer_scn)
        self._draw_traces(viewer_scn)
        if planned_sites is not None:
            self._draw_plan(planned_sites, viewer_scn)

    def build_figures(
        self,
        last_cost: float,
        cost_weights: np.ndarray,
        u: np.ndarray,
        cost_terms: dict[str, float] | None = None,
        u_nominal: np.ndarray | None = None,
        window_width: int = 1200,
    ) -> list[tuple[mujoco.MjrRect, mujoco.MjvFigure]]:
        """Return a list of (viewport, figure) pairs for viewer.set_figures().

        Panels are laid out in the bottom-right corner of the window.
        """
        self._step += 1

        # ── 1. Cost history ───────────────────────────────────────────────
        _figure_append(self._fig_cost, 0, float(self._step), float(last_cost))

        # ── 2. Cost terms bar chart ───────────────────────────────────────
        if cost_terms:
            n_terms = len(cost_terms)
            xs = np.arange(n_terms, dtype=np.float64)
            ys = np.array(list(cost_terms.values()), dtype=np.float64)
            for ci, (name, val) in enumerate(cost_terms.items()):
                rgb = _COLORS[ci % len(_COLORS)]
                self._fig_terms.linergb[ci] = list(rgb)
                self._fig_terms.linename[ci] = name.encode()[:36]
                _figure_set_line(self._fig_terms, ci,
                                 np.array([float(ci)]), np.array([float(val)]))

        # ── 3. Action horizon ─────────────────────────────────────────────
        if u_nominal is not None:
            H, nu = u_nominal.shape
            if not self._act_names_set:
                mjm = self._task.mj_model
                for ai in range(nu):
                    name = mujoco.mj_id2name(mjm, mujoco.mjtObj.mjOBJ_ACTUATOR, ai) or f"a{ai}"
                    rgb = _COLORS[ai % len(_COLORS)]
                    self._fig_actions.linergb[ai] = list(rgb)
                    self._fig_actions.linename[ai] = name.encode()[:36]
                self._act_names_set = True
            t_axis = np.arange(H, dtype=np.float64)
            for ai in range(nu):
                _figure_set_line(self._fig_actions, ai, t_axis, u_nominal[:, ai])

        # ── Layout: panels stacked on the RIGHT side, clear of the left menu ──
        # MuJoCo's left menu is ~224 px wide; place panels on the right edge.
        # Assume a typical window width of ~1200 px; panels are 280 px wide.
        # Stack bottom-up: actions at bottom, terms in middle, cost at top.
        PW, PH = 280, 160
        RIGHT_EDGE = window_width
        gap = 4
        left = RIGHT_EDGE - PW - 6
        figs_ordered = [self._fig_actions, self._fig_terms, self._fig_cost]
        panels = []
        for row, fig in enumerate(figs_ordered):
            bottom = 6 + row * (PH + gap)
            panels.append((mujoco.MjrRect(left, bottom, PW, PH), fig))
        return panels

    def build_texts(
        self,
        last_cost: float,
        cost_weights: np.ndarray,
    ) -> list[tuple[int | None, int | None, str | None, str | None]]:
        """Return text entries for viewer.set_texts().

        Each entry: (font, gridpos, left_text, right_text).
        gridpos=0 → top-left corner of the viewer.
        """
        N = len(cost_weights)
        ess = 1.0 / float(np.sum(cost_weights ** 2)) if N > 0 else 0.0
        return [
            (None, 0, f"cost  {last_cost:.4f}", f"ESS {ess:.0f}/{N}"),
        ]

    # ------------------------------------------------------------------
    # Legacy single-call update (kept for backwards compat)
    # ------------------------------------------------------------------

    def update(
        self,
        data: mujoco.MjData,
        planned_sites: np.ndarray | None,
        last_cost: float,
        cost_weights: np.ndarray,
        u_nominal_0: np.ndarray,
        viewer_scn: mujoco.MjvScene,
        ref_positions: np.ndarray | None = None,
        cost_terms: dict[str, float] | None = None,
        u_nominal: np.ndarray | None = None,
    ) -> None:
        """Combined update — call inside viewer.lock(), then call set_figures/set_texts yourself."""
        self.update_scene(data, planned_sites, ref_positions, viewer_scn)

    # ------------------------------------------------------------------
    def _draw_traces(self, scn: mujoco.MjvScene) -> None:
        for trace in self._traces:
            n = len(trace)
            buf = list(trace)
            for i, pos in enumerate(buf):
                age = i / max(n - 1, 1)
                alpha = 0.1 + 0.5 * age
                rgba = np.array([0.6, 0.6, 0.6, alpha])
                _add_sphere(scn, pos, 0.015, rgba)

    def _draw_reference(self, ref_positions: np.ndarray, scn: mujoco.MjvScene) -> None:
        n = len(ref_positions)
        orange     = np.array([1.0, 0.55, 0.0, 0.9])
        orange_dim = np.array([1.0, 0.55, 0.0, 0.35])
        for t in range(n):
            is_current = (t == 0)
            r    = 0.035 if is_current else 0.02
            rgba = orange if is_current else orange_dim
            _add_sphere(scn, ref_positions[t], r, rgba)
            if t > 0:
                _add_line(scn, ref_positions[t - 1], ref_positions[t],
                          0.004, orange_dim)

    def _draw_plan(self, planned_sites: np.ndarray, scn: mujoco.MjvScene) -> None:
        H, n_sites, _ = planned_sites.shape
        cyan = np.array([0.0, 0.9, 0.9, 0.8])
        for k in range(n_sites):
            for t in range(H):
                _add_sphere(scn, planned_sites[t, k], 0.02, cyan)
                if t > 0:
                    _add_line(scn, planned_sites[t - 1, k], planned_sites[t, k],
                              0.003, cyan)
