"""DumpPcdForGgx action.

Debug/validation action: dumps the target object's stored point-cloud data
from WorldState to .npy files for handoff to an external process. Used to
validate GraspGenX against real observed point clouds without wiring a new
grasp-estimation module into the action layer yet.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from src.core.action import AbstractAction, ActionResult
from src.core.context import Context
import src.core.logger as log

DEFAULT_OUTPUT_DIR = "/home/orchid/orchid_server/submodules/GGX/GraspGenX/scratch_handoff"


class DumpPcdForGgx(AbstractAction):
    """Dump ``<target>_object_pcd`` / ``<target>_surround_pcd`` to .npy files.

    Use this after an observation action (e.g. ``ObserveScene``) has stored
    point-cloud data for the target object in ``WorldState``. Writes
    ``<target>_object.npy`` and ``<target>_surround.npy`` (when available)
    into ``output_dir`` for an external process to pick up via a shared
    bind mount.

    Params:
        target: str - Object identifier used to read WorldState point clouds.
        output_dir: str - Directory to write the .npy files into.
    """

    required_modules = []

    def __init__(self, target: str='', output_dir: str=DEFAULT_OUTPUT_DIR) -> None:
        self.target = target
        self.output_dir = output_dir
        super().__init__()

    def execute(self, context: Context) -> ActionResult:
        ws = context.world_state
        if ws is None:
            return ActionResult(status="failure", reason="missing_world_state")

        object_pcd = ws.get(f"{self.target}_object_pcd")
        surround_pcd = ws.get(f"{self.target}_surround_pcd")
        if object_pcd is None:
            return ActionResult(
                status="needs_replan",
                reason="missing_object_pointcloud",
                suggested_replan_context={
                    "action": "DumpPcdForGgx",
                    "target": self.target,
                },
            )

        os.makedirs(self.output_dir, exist_ok=True)

        object_path = os.path.join(self.output_dir, f"{self.target}_object.npy")
        np.save(object_path, np.asarray(object_pcd, dtype=np.float32))
        log.info(f"DumpPcdForGgx: wrote {object_path}")

        surround_path = None
        if surround_pcd is not None:
            surround_path = os.path.join(self.output_dir, f"{self.target}_surround.npy")
            np.save(surround_path, np.asarray(surround_pcd, dtype=np.float32))
            log.info(f"DumpPcdForGgx: wrote {surround_path}")

        return ActionResult(
            status="success",
            payload={"object_path": object_path, "surround_path": surround_path},
        )
