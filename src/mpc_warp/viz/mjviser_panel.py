"""MJPC-style real-time panels rendered inside a viser browser window."""
from __future__ import annotations

from collections import deque

import numpy as np
import viser
import viser.uplot as uplot


class MppiPanel:
    """Cost history, named cost terms, ESS, action horizon and action bar panels.

    Call ``update()`` after each MPPI step.
    """

    def __init__(
        self,
        server: viser.ViserServer,
        nu: int,
        actuator_names: list[str] | None = None,
        max_history: int = 300,
    ) -> None:
        self._nu = nu
        self._names = actuator_names or [f"a{i}" for i in range(nu)]
        self._step = 0
        self._cost_x: deque[float] = deque(maxlen=max_history)
        self._cost_y: deque[float] = deque(maxlen=max_history)

        # Per-term history (populated on first update once term names are known).
        self._term_names: list[str] = []
        self._term_histories: dict[str, deque[float]] = {}
        self._term_plot: viser.GuiUplotHandle | None = None

        with server.gui.add_folder("MPC · Objective"):
            _x0 = np.array([0.0, 1.0])
            _y0 = np.zeros(2)
            self._cost_plot = server.gui.add_uplot(
                data=(_x0, _y0),
                series=(
                    uplot.Series(label="step"),
                    uplot.Series(label="total cost", stroke="#e74c3c", width=2),
                ),
                title="Running cost",
            )
            with server.gui.add_folder("Cost terms"):
                self._terms_html = server.gui.add_html(
                    "<div style='color:#aaa;font-size:0.8em'>waiting…</div>"
                )

        with server.gui.add_folder("MPC · Planner"):
            self._ess_bar = server.gui.add_progress_bar(value=0.0, animated=False)
            self._ess_html = server.gui.add_html(
                "<div style='font-size:0.8em'>ESS: —</div>"
            )

        self._horizon_plot: viser.GuiUplotHandle | None = None
        self._server = server

        with server.gui.add_folder("MPC · Actions (current)"):
            self._action_html = server.gui.add_html(
                self._render_action_bars(np.zeros(nu))
            )

    # ------------------------------------------------------------------
    def update(
        self,
        last_cost: float,
        cost_weights: np.ndarray,       # (N,) softmin weights
        u: np.ndarray,                  # (nu,) current action
        cost_terms: dict[str, float] | None = None,  # named breakdown
        u_nominal: np.ndarray | None = None,          # (H, nu) horizon
    ) -> None:
        self._step += 1
        step = float(self._step)

        # ── total cost history ─────────────────────────────────────────
        self._cost_x.append(step)
        self._cost_y.append(float(last_cost))
        x = np.array(self._cost_x, dtype=np.float64)
        y = np.array(self._cost_y, dtype=np.float64)
        self._cost_plot.data = (x, y)

        # ── named cost terms ───────────────────────────────────────────
        if cost_terms:
            self._terms_html.content = self._render_cost_terms(cost_terms)

        # ── ESS ────────────────────────────────────────────────────────
        N = len(cost_weights)
        sq = float(np.sum(cost_weights ** 2))
        ess = float(1.0 / sq) if sq > 0 else 0.0
        frac = float(np.clip(ess / max(N, 1), 0.0, 1.0))
        self._ess_bar.value = frac
        self._ess_html.content = (
            f"<div style='font-size:0.8em'>ESS: {ess:.0f} / {N}"
            f"  ({100 * frac:.0f}%)</div>"
        )

        # ── action horizon uplot (built once, updated each step) ───────
        if u_nominal is not None:
            self._update_horizon_plot(u_nominal)

        # ── current action bars ────────────────────────────────────────
        self._action_html.content = self._render_action_bars(u)

    # ------------------------------------------------------------------
    def _update_horizon_plot(self, u_nominal: np.ndarray) -> None:
        H, nu = u_nominal.shape
        t_axis = np.arange(H, dtype=np.float64)

        if self._horizon_plot is None:
            # Build the plot on first call.
            _stroke_colors = ["#2ecc71", "#e74c3c", "#3498db", "#f39c12",
                               "#9b59b6", "#1abc9c"]
            series_list = [uplot.Series(label="step")]
            for i in range(nu):
                name = self._names[i] if i < len(self._names) else f"a{i}"
                color = _stroke_colors[i % len(_stroke_colors)]
                series_list.append(
                    uplot.Series(label=name, stroke=color, width=1.5)
                )
            with self._server.gui.add_folder("MPC · Actions (horizon)"):
                self._horizon_plot = self._server.gui.add_uplot(
                    data=(t_axis, *[u_nominal[:, i] for i in range(nu)]),
                    series=tuple(series_list),
                    title="Nominal actions over horizon",
                )
        else:
            self._horizon_plot.data = (
                t_axis,
                *[u_nominal[:, i] for i in range(nu)],
            )

    # ------------------------------------------------------------------
    def _render_cost_terms(self, terms: dict[str, float]) -> str:
        total = sum(terms.values()) or 1.0
        stroke_colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6", "#1abc9c"]
        rows = []
        for ci, (name, val) in enumerate(terms.items()):
            frac = float(val) / total
            color = stroke_colors[ci % len(stroke_colors)]
            pct = frac * 100.0
            rows.append(
                f"<div style='display:flex;align-items:center;margin:2px 0;gap:4px;'>"
                f"<span style='width:5em;font-size:0.7em;color:{color};'>{name}</span>"
                f"<div style='flex:1;height:8px;background:#333;border-radius:2px;'>"
                f"<div style='width:{pct:.1f}%;height:100%;background:{color};"
                f"border-radius:2px;'></div></div>"
                f"<span style='width:4em;font-size:0.7em;text-align:right;color:#ccc;'>"
                f"{val:.3f}</span></div>"
            )
        return "<div style='padding:4px;'>" + "".join(rows) + "</div>"

    # ------------------------------------------------------------------
    def _render_action_bars(self, u: np.ndarray) -> str:
        rows = []
        for i, ui in enumerate(u):
            ui_c = float(np.clip(ui, -1.0, 1.0))
            name = self._names[i] if i < len(self._names) else f"a{i}"
            pct = abs(ui_c) * 50
            color = "#2ecc71" if ui_c >= 0 else "#e74c3c"
            if ui_c >= 0:
                bar = (
                    f"<div style='margin-left:50%;width:{pct:.1f}%;height:100%;"
                    f"background:{color};border-radius:2px;'></div>"
                )
            else:
                bar = (
                    f"<div style='margin-left:{50 - pct:.1f}%;width:{pct:.1f}%;height:100%;"
                    f"background:{color};border-radius:2px;'></div>"
                )
            rows.append(
                f"<div style='display:flex;align-items:center;margin:2px 0;gap:4px;'>"
                f"<span style='width:5em;font-size:0.7em;overflow:hidden;white-space:nowrap;"
                f"color:#aaa;'>{name}</span>"
                f"<div style='flex:1;height:8px;background:#333;border-radius:2px;'>"
                f"{bar}</div>"
                f"<span style='width:4em;font-size:0.7em;text-align:right;color:#ccc;'>"
                f"{ui_c:+.2f}</span></div>"
            )
        return "<div style='padding:4px;'>" + "".join(rows) + "</div>"
