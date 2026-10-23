#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Intermediate tool: render the raw CAD assembly (per-part meshes placed by the
SolidWorks component transforms) with MuJoCo, to validate the export and to see
the mechanism.  Not part of the final model.

    python src/create_model/preview_assembly.py
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[2]
EXPORT_JSON = PROJECT_DIR / "src" / "create_model" / "cad_export" / "cad_export.json"
MESH_DIR = PROJECT_DIR / "model" / "meshes"
OUT_DIR = PROJECT_DIR / "src" / "create_model" / "cad_export"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PALETTE = {
    "body": "#4c78a8", "ld_hip": "#f58518", "rd_hip": "#e45756",
    "left_thigh": "#72b7b2", "right_thigh": "#54a24b",
    "calf": "#b279a2", "wheel": "#9d755d",
}


def mat_to_quat(m: np.ndarray) -> np.ndarray:
    tr = np.trace(m)
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        q = np.array([0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s])
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        q = np.array([(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s,
                      (m[0, 2] + m[2, 0]) / s])
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        q = np.array([(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s,
                      (m[1, 2] + m[2, 1]) / s])
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        q = np.array([(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s,
                      (m[1, 2] + m[2, 1]) / s, 0.25 * s])
    q /= np.linalg.norm(q)
    return q if q[0] >= 0 else -q


def fmt(v, nd=6) -> str:
    out = []
    for x in v:
        s = f"{x:.{nd}f}".rstrip("0").rstrip(".")
        out.append("0" if s in ("", "-0") else s)
    return " ".join(out)


def build_xml(instances, group_of) -> str:
    lines = ['<mujoco model="w2l_12_assembly_preview">',
             '  <compiler angle="radian" meshdir="meshes" autolimits="true"/>',
             '  <option timestep="0.002"/>',
             '  <visual><global offwidth="1600" offheight="1200"/>',
             '    <headlight ambient="0.45 0.45 0.47" diffuse="0.6 0.6 0.6" specular="0.1 0.1 0.1"/></visual>',
             '  <asset>']
    for key, col in PALETTE.items():
        rgba = tuple(int(col[i:i + 2], 16) / 255 for i in (1, 3, 5))
        lines.append(f'    <material name="mat_{key}" rgba="{rgba[0]:.3f} {rgba[1]:.3f} {rgba[2]:.3f} 1"/>')
    names = sorted({i["mesh"] for i in instances if "mesh" in i})
    for name in names:
        lines.append(f'    <mesh name="{name[:-4]}" file="{name}"/>')
    lines.append("  </asset>")
    lines.append("  <worldbody>")
    lines.append('    <light pos="0.5 0.4 1.2" dir="-0.3 -0.3 -1" directional="true"/>')
    lines.append('    <body name="assembly" pos="0 0 0">')
    for inst in instances:
        if "mesh" not in inst:
            continue
        tf = np.asarray(inst["transform"], float)
        key = group_of(inst["group"])
        q = mat_to_quat(tf[:3, :3])
        lines.append(f'      <geom type="mesh" mesh="{inst["mesh"][:-4]}" material="mat_{key}" '
                     f'pos="{fmt(tf[:3, 3])}" quat="{fmt(q)}" contype="0" conaffinity="0"/>')
    lines.append("    </body>")
    lines.append("  </worldbody>")
    lines.append("</mujoco>")
    return "\n".join(lines)


def read_stl_bbox(path: Path) -> tuple[np.ndarray, np.ndarray]:
    raw = path.read_bytes()
    n = int(np.frombuffer(raw[80:84], np.uint32)[0])
    rec = np.frombuffer(raw[84:84 + n * 50],
                        dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", 9), ("a", "<u2")]))
    v = rec["v"].astype(np.float64).reshape(-1, 3)
    return v.min(0), v.max(0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="cad")
    args = ap.parse_args()

    import mujoco
    from PIL import Image

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = json.loads(EXPORT_JSON.read_text(encoding="utf-8"))
    instances, meshes = data["instances"], data["meshes"]

    # part bboxes in world, for the report
    pts_all = []
    for inst in instances:
        if "mesh" not in inst:
            continue
        info = meshes[inst["doc"]]
        if "bbox_min_m" in info:
            lo = np.array(info["bbox_min_m"])
            hi = np.array(info["bbox_max_m"])
        else:
            lo, hi = read_stl_bbox(MESH_DIR / info["mesh"])
        corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
        tf = np.asarray(inst["transform"], float)
        pts_all.append(corners @ tf[:3, :3].T + tf[:3, 3])
    pts = np.vstack(pts_all)
    print(f"assembly bbox (mm): {np.round(pts.min(0)*1e3,1)} .. {np.round(pts.max(0)*1e3,1)}  "
          f"size={np.round((pts.max(0)-pts.min(0))*1e3,1)}")

    def group_of(name: str) -> str:
        return name.split("-")[0]

    xml = build_xml(instances, group_of)
    # the throw-away MJCF lives in the system temp dir, the meshes are addressed
    # by absolute path so nothing is written into model/
    xml = xml.replace('meshdir="meshes"', f'meshdir="{MESH_DIR.as_posix()}"')
    tmp = Path(tempfile.gettempdir()) / "w2l_12_assembly_preview.xml"
    tmp.write_text(xml, encoding="utf-8")
    model = mujoco.MjModel.from_xml_path(str(tmp))
    d = mujoco.MjData(model)
    mujoco.mj_forward(model, d)
    print(f"model ok: ngeom={model.ngeom} nbody={model.nbody}")

    center = (pts.min(0) + pts.max(0)) / 2
    extent = float(np.linalg.norm(pts.max(0) - pts.min(0)))
    renderer = mujoco.Renderer(model, 800, 1100)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = center
    cam.distance = extent * 1.25
    views = {"iso": (135, -20), "front": (90, 0), "side": (180, 0), "top": (135, -80), "rear": (-40, -20)}
    for name, (az, el) in views.items():
        cam.azimuth, cam.elevation = az, el
        renderer.update_scene(d, cam)
        img = renderer.render()
        Image.fromarray(img).save(OUT_DIR / f"{args.tag}_{name}.png")
        print(f"  saved {args.tag}_{name}.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
