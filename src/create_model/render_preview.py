#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Intermediate: offscreen renders of the built model (no display needed).

    python src/create_model/render_preview.py
    python src/create_model/render_preview.py --pose crouch --out docs/preview.png
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_XML = PROJECT_DIR / "model" / "w2l_12.xml"
DEFAULT_OUT = PROJECT_DIR / "src" / "create_model" / "renders"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

LEGS = ("FL", "FR", "RL", "RR")


def apply_pose(model, data, pose: str) -> None:
    """Poses are absolute joint angles in the robot convention:
    roll 0 = thigh horizontal outwards / 90 = straight down,
    hip 0 = thigh straight down (negative = backwards),
    knee 0 = calf in line with the thigh (positive = forwards)."""
    adr = {mujoco_name(model, i): int(model.jnt_qposadr[i]) for i in range(model.njnt)}

    def setj(name: str, deg: float) -> None:
        if name in adr:
            data.qpos[adr[name]] = np.radians(deg)

    if pose == "crouch":
        for leg in LEGS:
            setj(f"roll_{leg}", 0.0)
            setj(f"hip_{leg}", -52.0)
            setj(f"knee_{leg}", 118.0)
    elif pose == "extend":
        for leg in LEGS:
            setj(f"roll_{leg}", 0.0)
            setj(f"hip_{leg}", 0.0)
            setj(f"knee_{leg}", 0.0)
    elif pose == "splay":
        for leg in LEGS:
            setj(f"roll_{leg}", 24.0)
    elif pose == "asym":
        for i, leg in enumerate(LEGS):
            setj(f"roll_{leg}", 6.0 * (1 if i % 2 else -1))
            setj(f"hip_{leg}", -30.0 - 20.0 * i + 18.0)
            setj(f"knee_{leg}", 80.0 + 22.0 * i - 12.0)
    ground(model, data)


def ground(model, data) -> None:
    """Drop the robot until its lowest collision point sits on the floor plane."""
    mujoco.mj_kinematics(model, data)
    low = math.inf
    for gid in range(model.ngeom):
        if model.geom_contype[gid] == 0 or model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_PLANE:
            continue
        pos = data.geom_xpos[gid]
        mat = data.geom_xmat[gid].reshape(3, 3)
        size = model.geom_size[gid]
        gtype = model.geom_type[gid]
        if gtype == mujoco.mjtGeom.mjGEOM_MESH:
            adr = int(model.geom_dataid[gid])
            a0, n = int(model.mesh_vertadr[adr]), int(model.mesh_vertnum[adr])
            v = np.asarray(model.mesh_vert[a0:a0 + n], dtype=np.float64)
            z = float((v @ mat[2]).min()) + pos[2]
        elif gtype == mujoco.mjtGeom.mjGEOM_BOX:
            corners = np.array([[sx, sy, sz] for sx in (-1.0, 1.0)
                                for sy in (-1.0, 1.0) for sz in (-1.0, 1.0)]) * size
            z = float((corners @ mat[2]).min()) + pos[2]
        elif gtype == mujoco.mjtGeom.mjGEOM_CYLINDER:
            c = abs(float(mat[2, 2]))
            z = pos[2] - (size[1] * c + size[0] * math.sqrt(max(0.0, 1.0 - c * c)))
        else:
            z = pos[2] - float(model.geom_rbound[gid])
        low = min(low, z)
    if math.isfinite(low):
        data.qpos[2] -= low
        mujoco.mj_kinematics(model, data)


def mujoco_name(model, i: int) -> str:
    import mujoco
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) or ""


def main() -> int:
    global mujoco
    import mujoco
    from PIL import Image

    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", type=Path, default=DEFAULT_XML)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--pose", default="zero",
                    choices=["zero", "crouch", "extend", "splay", "asym"])
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    model = mujoco.MjModel.from_xml_path(str(args.xml))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    apply_pose(model, data, args.pose)
    mujoco.mj_forward(model, data)

    args.out.mkdir(parents=True, exist_ok=True)
    renderer = mujoco.Renderer(model, 800, 1100)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0.0, 0.0, 0.10]
    cam.distance = 0.62
    views = {"iso": (135, -18), "front": (90, -4), "side": (180, -4),
             "top": (135, -78), "back": (-40, -12)}
    tag = args.tag or args.pose
    for name, (az, el) in views.items():
        cam.azimuth, cam.elevation = az, el
        renderer.update_scene(data, cam)
        Image.fromarray(renderer.render()).save(args.out / f"{tag}_{name}.png")
        print(f"  {args.out / f'{tag}_{name}.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
