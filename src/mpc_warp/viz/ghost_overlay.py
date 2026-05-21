"""Semi-transparent ghost overlay for mjviser — shows a reference MuJoCo pose.

Usage::

    ghost = GhostOverlay(server, task.mj_model, alpha=0.35, color=(0.4, 0.7, 1.0))
    # each frame:
    ghost.update(ref_qpos)          # numpy (nq,)
    ghost.update(ref_qpos, visible=True)
"""

from __future__ import annotations

import copy

import mujoco
import numpy as np
import viser


class GhostOverlay:
    """Render a semi-transparent copy of a MuJoCo model at an arbitrary qpos.

    Geometry is pre-built as trimesh handles under a ``/ghost/`` viser prefix.
    Collision-only geoms (contype/conaffinity != 0 and group >= 3) are hidden.
    Each frame, calling :meth:`update` runs forward kinematics and repositions
    body frames so the ghost tracks the reference configuration.
    """

    def __init__(
        self,
        server: viser.ViserServer,
        mj_model: mujoco.MjModel,
        *,
        alpha: float = 0.35,
        color: tuple[float, float, float] = (0.4, 0.7, 1.0),
        prefix: str = "/ghost",
    ) -> None:
        try:
            import trimesh
            from mjviser.conversions import (
                get_body_name,
                group_geoms_by_visual_compat,
                merge_geoms,
            )
        except ImportError as e:
            raise ImportError("GhostOverlay requires mjviser and trimesh") from e

        self._server = server
        self._prefix = prefix

        # Build a ghost model: hide collision geoms, recolour visual geoms.
        # mjviser's _resolve_flat_rgba gives priority to mat_rgba over geom_rgba,
        # so we must override mat_rgba too. We strip the material from every geom
        # (geom_matid = -1) so geom_rgba is always the authority, then set geom_rgba
        # to the ghost colour for visual geoms and alpha=0 for collision-only geoms.
        ghost_model = copy.deepcopy(mj_model)
        ghost_model.geom_matid[:] = -1
        for gi in range(ghost_model.ngeom):
            is_collision = ghost_model.geom_contype[gi] != 0 or ghost_model.geom_conaffinity[gi] != 0
            if is_collision and ghost_model.geom_group[gi] >= 3:
                ghost_model.geom_rgba[gi, 3] = 0.0
            else:
                ghost_model.geom_rgba[gi, :3] = color
                ghost_model.geom_rgba[gi, 3] = alpha

        self._ghost_model = ghost_model
        self._ghost_data = mujoco.MjData(ghost_model)

        # body_id -> viser FrameHandle (we move it each step)
        self._body_frames: dict[int, viser.FrameHandle] = {}

        # Pre-build trimesh handles grouped by (body, texture-compat subgroup).
        mujoco.mj_kinematics(ghost_model, self._ghost_data)

        body_group_geoms: dict[tuple[int, int], list[int]] = {}
        for gi in range(ghost_model.ngeom):
            if ghost_model.geom_rgba[gi, 3] == 0.0:
                continue
            if ghost_model.geom_bodyid[gi] == 0:  # world body — skip floor/ground
                continue
            if ghost_model.geom_type[gi] == mujoco.mjtGeom.mjGEOM_PLANE:
                continue
            body_id = ghost_model.geom_bodyid[gi]
            group_id = int(ghost_model.geom_group[gi])
            body_group_geoms.setdefault((body_id, group_id), []).append(gi)

        for (body_id, group_id), geom_ids in body_group_geoms.items():
            body_name = get_body_name(ghost_model, body_id)
            frame_path = f"{prefix}/{body_name}"

            if frame_path not in self._body_frames:
                frame = server.scene.add_frame(frame_path, show_axes=False)
                self._body_frames[body_id] = frame

            subgroups = group_geoms_by_visual_compat(ghost_model, geom_ids)
            for sub_idx, sub_geom_ids in enumerate(subgroups):
                suffix = f"/sub{sub_idx}" if len(subgroups) > 1 else ""
                mesh = merge_geoms(ghost_model, sub_geom_ids)

                # Vertex-color alpha alone is not enough for transparency in
                # three.js/viser — the GLB material must declare alphaMode BLEND.
                mesh.visual = trimesh.visual.TextureVisuals(
                    material=trimesh.visual.material.PBRMaterial(
                        baseColorFactor=[*color, alpha],
                        alphaMode="BLEND",
                    )
                )
                server.scene.add_mesh_trimesh(
                    f"{frame_path}/group{group_id}{suffix}",
                    mesh,
                    cast_shadow=False,
                )

        self._visible = True

    # ------------------------------------------------------------------

    def update(
        self,
        qpos: np.ndarray,
        *,
        visible: bool = True,
        scene_offset: np.ndarray | None = None,
    ) -> None:
        """Reposition the ghost to the given joint configuration.

        Args:
            qpos: Reference joint positions, shape ``(nq,)``.
            visible: Whether to show the ghost this frame.
            scene_offset: World-space offset applied by the ViserMujocoScene
                (``scene._scene_offset``). Pass this so the ghost aligns with
                the rendered robot when the scene is recentred on the tracked
                body. If omitted, no offset correction is applied.
        """
        if visible != self._visible:
            self._set_visible(visible)
            self._visible = visible

        if not visible:
            return

        self._ghost_data.qpos[:] = qpos
        mujoco.mj_kinematics(self._ghost_model, self._ghost_data)

        offset = scene_offset if scene_offset is not None else np.zeros(3)
        for body_id, frame in self._body_frames.items():
            frame.position = self._ghost_data.xpos[body_id] + offset
            frame.wxyz = self._ghost_data.xquat[body_id]

    def set_visible(self, visible: bool) -> None:
        if visible != self._visible:
            self._set_visible(visible)
            self._visible = visible

    def _set_visible(self, visible: bool) -> None:
        with self._server.scene.atomic():
            for frame in self._body_frames.values():
                frame.visible = visible
