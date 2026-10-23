#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Export the SolidWorks assembly of the w2l_12 wheel-legged robot to
per-part STL meshes + a JSON instance tree (the CAD -> MuJoCo intermediate).

Why not "Save as STL" from the GUI: a robot needs *one mesh per part* plus the
exact placement of every instance, otherwise the links cannot be articulated.
This script therefore reads the assembly tree through the SolidWorks COM API,
tessellates every unique part once (in its own part coordinate system) and
records the component -> assembly transform of every instance.

Outputs
-------
<mesh-dir>/part_XXX.stl      one binary STL per unique part (metres, part frame)
<json>                       instance tree, transforms, mesh bookkeeping

Usage (conda env ms, SolidWorks 2024 must be installed):
    python src/create_model/export_cad.py
    python src/create_model/export_cad.py --list-only
    python src/create_model/export_cad.py --quality 1.0
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[2]      # src/create_model/x.py -> w2l_12/
DEFAULT_ASM = Path(r"D:\project\robot_leg\model\w2l_12\w2l_12.SLDASM")
DEFAULT_MESH_DIR = PROJECT_DIR / "model" / "meshes"
DEFAULT_JSON = PROJECT_DIR / "src" / "create_model" / "cad_export" / "cad_export.json"

SW_DOC_PART = 1
SW_DOC_ASSEMBLY = 2
SW_ALL_BODIES = 0
SW_SUPPRESSED = 0

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


# --------------------------------------------------------------------------
# SolidWorks COM helpers
# --------------------------------------------------------------------------
# The SW typelib is dual-interface and pywin32's late binding exposes a method
# as a plain property whenever the only argument is the return value. `_get`
# therefore tries the call form and falls back to the plain property value.
def _is_com(v) -> bool:
    return v is not None and not isinstance(v, (str, int, float, bool, tuple, list, dict, bytes))


def _get(obj, name, *args):
    v = getattr(obj, name)
    if args:
        return v(*args)
    if not _is_com(v):
        return v
    try:
        return v()
    except Exception:  # noqa: BLE001 - the property *is* the value
        return v


def _soft(obj, name, *args, default=None):
    if obj is None:
        return default
    try:
        return _get(obj, name, *args)
    except Exception:  # noqa: BLE001
        return default


@dataclass
class Instance:
    """One component occurrence in the top assembly."""
    path: str                      # "body-1/sts3032_单轴-4"
    name: str                      # "sts3032_单轴-4"
    group: str                     # top-level ancestor name: "body-1"
    doc_path: str = ""             # referenced .SLDPRT / .SLDASM
    doc_type: int = -1
    transform: np.ndarray = field(default_factory=lambda: np.eye(4))  # part -> top assembly
    mass: float | None = None      # kg, from the part's own mass properties
    com_local: list | None = None  # part frame, m
    children: list["Instance"] = field(default_factory=list)


def xform_from_array(arr) -> np.ndarray:
    """SW MathTransform.ArrayData -> 4x4 (part frame -> top assembly frame).

    The 16 doubles hold the rotation **column-major** in [0:9], the translation in
    [9:12] and a scale/trailing block in [12:16].  `IComponent2::Transform2` is
    already absolute: for a part inside a sub-assembly it is the full part ->
    top-assembly transform, so nested components must NOT be multiplied again.
    Both facts were verified against SolidWorks' own assembly mass properties
    (composing the part centres of mass reproduces them to machine precision;
    row-major or re-multiplied nesting is off by tens of millimetres).
    """
    a = np.asarray(arr, dtype=np.float64)
    m = np.eye(4)
    m[:3, :3] = a[:9].reshape(3, 3).T
    m[:3, 3] = a[9:12]
    return m


def mass_property(doc):
    """SI-unit mass properties of a document (kg, metres)."""
    ext = _soft(doc, "Extension", default=None)
    mp = _soft(ext, "CreateMassProperty", default=None)
    if mp is None:
        return None
    try:
        mp.UseSystemUnits = True
    except Exception:  # noqa: BLE001
        pass
    return mp


def walk_component(comp, group: str, prefix: str, instances: list[Instance]) -> None:
    name = _soft(comp, "Name2", default="?")
    doc_path = _soft(comp, "GetPathName", default="") or ""
    suppression = _soft(comp, "GetSuppression", default=2)
    if suppression == SW_SUPPRESSED:
        return
    local = _soft(comp, "Transform2", default=None)
    tf = xform_from_array(local.ArrayData) if local is not None else np.eye(4)
    path = f"{prefix}{name}"
    inst = Instance(path=path, name=name, group=group, doc_path=str(doc_path), transform=tf)
    doc = _soft(comp, "GetModelDoc2", default=None)
    inst.doc_type = int(_soft(doc, "GetType", default=-1) if doc is not None else -1)
    if doc is not None and inst.doc_type == SW_DOC_PART:
        mp = mass_property(doc)
        if mp is not None:
            inst.mass = float(_soft(mp, "Mass", default=0.0))
            inst.com_local = [float(v) for v in (_soft(mp, "CenterOfMass", default=None) or [])]
    instances.append(inst)

    kids = _soft(comp, "GetChildren", default=None)
    if kids:
        for kid in kids:
            walk_component(kid, group, path + "/", instances)


# --------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------
def part_triangles(doc) -> np.ndarray:
    """All triangles of every solid body of a part, in the part frame (metres)."""
    bodies = _soft(doc, "GetBodies2", SW_ALL_BODIES, False, default=None)
    if not bodies:
        return np.zeros((0, 3, 3))
    chunks = []
    for body in bodies:
        faces = _soft(body, "GetFaces", default=None)
        if not faces:
            continue
        for face in faces:
            arr = _soft(face, "GetTessTriangles", True, default=None)
            if not arr:
                continue
            chunks.append(np.asarray(arr, dtype=np.float64).reshape(-1, 3, 3))
    return np.vstack(chunks) if chunks else np.zeros((0, 3, 3))


def write_binary_stl(path: Path, tris: np.ndarray) -> None:
    n = np.array([np.cross(t[1] - t[0], t[2] - t[0]) for t in tris], dtype=np.float64)
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    n = np.divide(n, ln, out=np.zeros_like(n), where=ln > 1e-20)
    rec = np.zeros(len(tris), dtype=[("n", "<f4", 3), ("v", "<f4", 9), ("a", "<u2")])
    rec["n"] = n.astype("<f4")
    rec["v"] = tris.astype("<f4").reshape(len(tris), 9)
    header = b"w2l_12 cad export (SolidWorks tessellation)"
    with path.open("wb") as fh:
        fh.write(header.ljust(80, b" "))
        fh.write(np.uint32(len(tris)).tobytes())
        fh.write(rec.tobytes())


def mesh_volume(tris: np.ndarray) -> float:
    a, b, c = tris[:, 0], tris[:, 1], tris[:, 2]
    return float(abs(np.einsum("ij,ij->i", a, np.cross(b, c)).sum()) / 6.0)


def sanitize(text: str) -> str:
    out = []
    for ch in text:
        if ch.isascii() and (ch.isalnum() or ch in "-_."):
            out.append(ch)
        else:
            out.append("_")
    s = "".join(out).strip("_")
    while "__" in s:
        s = s.replace("__", "_")
    return s or "part"


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--assembly", type=Path, default=DEFAULT_ASM)
    ap.add_argument("--mesh-dir", type=Path, default=DEFAULT_MESH_DIR)
    ap.add_argument("--json", type=Path, default=DEFAULT_JSON)
    ap.add_argument("--list-only", action="store_true", help="only print the component tree")
    ap.add_argument("--keep-open", action="store_true", help="leave SolidWorks open when done")
    ap.add_argument("--phase", choices=("all", "tree", "parts"), default="all",
                    help="tree: only the instance tree; parts: only the meshes (resumable)")
    ap.add_argument("--only", default="", help="comma separated part-name substrings to export")
    args = ap.parse_args()

    import win32com.client.dynamic as dyn

    print(f"[sw] connecting ...", flush=True)
    app = dyn.Dispatch("SldWorks.Application.32")
    print(f"[sw] revision {_soft(app, 'RevisionNumber', default='?')}", flush=True)
    try:
        app.Visible = False
    except Exception:  # noqa: BLE001
        pass
    _soft(app, "CloseAllDocuments", False)

    if args.phase in ("all", "tree"):
        return export_tree(app, args)
    return export_parts(app, args)


def verify_placement(doc, parts: list[Instance]) -> None:
    """Cross-check the composed part placements against SolidWorks' own mass
    properties of the whole assembly (mass + centre of mass)."""
    mp = mass_property(doc)
    if mp is None or not parts:
        return
    com_sw = np.asarray(_soft(mp, "CenterOfMass", default=None) or [], dtype=float)
    mass_sw = float(_soft(mp, "Mass", default=0.0))
    tot = 0.0
    acc = np.zeros(3)
    for inst in parts:
        if inst.mass is None or not inst.com_local:
            continue
        c = np.asarray(inst.com_local, dtype=float)
        tot += inst.mass
        acc += inst.mass * (inst.transform[:3, :3] @ c + inst.transform[:3, 3])
    if tot <= 0 or com_sw.size != 3:
        return
    com = acc / tot
    err = np.linalg.norm(com - com_sw) * 1e3
    flag = "OK " if err < 0.5 else "MISMATCH"
    print(f"[check] {flag} mass {tot:.6f} vs {mass_sw:.6f} kg | "
          f"com err {err:.3f} mm  ({np.round(com * 1e3, 2)})", flush=True)


def export_tree(app, args) -> int:
    """Open the assembly once, dump the component tree, close it again."""
    t0 = time.perf_counter()
    doc = _soft(app, "OpenDoc", str(args.assembly), SW_DOC_ASSEMBLY, default=None)
    if doc is None:
        print(f"[error] cannot open {args.assembly}", file=sys.stderr)
        return 1
    print(f"[sw] opened {args.assembly.name} in {time.perf_counter() - t0:.1f}s", flush=True)

    comps = _soft(doc, "GetComponents", True, default=None) or ()
    print(f"[sw] {len(comps)} top-level components", flush=True)

    instances: list[Instance] = []
    for comp in comps:
        name = _soft(comp, "Name2", default="?")
        walk_component(comp, group=name, prefix="", instances=instances)

    dots = [c for c in instances if c.doc_type == SW_DOC_PART]
    print(f"[sw] {len(instances)} component instances, {len(dots)} part instances", flush=True)

    verify_placement(doc, dots)

    # unique part documents (by resolved path)
    unique: dict[str, dict] = {}
    for inst in dots:
        if inst.doc_path and inst.doc_path not in unique:
            unique[inst.doc_path] = {"parts": []}
        if inst.doc_path:
            unique[inst.doc_path]["parts"].append(inst)

    print(f"[sw] {len(unique)} unique part files", flush=True)

    if args.list_only:
        for inst in instances:
            depth = inst.path.count("/")
            print(f"{'  ' * depth}{inst.name:24s} tf=({inst.transform[0,3]*1e3:+7.1f},"
                  f"{inst.transform[1,3]*1e3:+7.1f},{inst.transform[2,3]*1e3:+7.1f})mm  {Path(inst.doc_path).name}")
        _soft(app, "CloseAllDocuments", False)
        return 0

    args.json.parent.mkdir(parents=True, exist_ok=True)

    # part index: sorted unique paths, stable across runs
    meta: dict[str, dict] = {}
    for index, doc_path in enumerate(sorted(unique), start=1):
        stem = Path(doc_path).stem
        meta[doc_path] = {
            "index": index,
            "name": stem,
            "mesh": f"part_{index:03d}_{sanitize(stem)}.stl",
        }

    out_instances = []
    for inst in instances:
        entry = {
            "path": inst.path,
            "name": inst.name,
            "group": inst.group,
            "doc_type": inst.doc_type,
            "doc": inst.doc_path,
            "transform": [[round(float(v), 9) for v in row] for row in inst.transform],
        }
        if inst.doc_path in meta:
            entry["mesh"] = meta[inst.doc_path]["mesh"]
        out_instances.append(entry)

    payload = {
        "assembly": str(args.assembly),
        "exported_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "length_unit": "m",
        "meshes": meta,
        "instances": out_instances,
    }
    args.json.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {args.json}  ({len(meta)} parts, {len(out_instances)} instances)")

    _soft(app, "CloseAllDocuments", False)
    if args.phase == "tree":
        return 0
    return export_parts(app, args)


def export_parts(app, args) -> int:
    """Tessellate every part of the tree file; resumable, skips finished meshes."""
    if not args.json.exists():
        print(f"[error] run --phase tree first ({args.json} missing)", file=sys.stderr)
        return 1
    payload = json.loads(args.json.read_text(encoding="utf-8"))
    meta: dict[str, dict] = payload["meshes"]
    args.mesh_dir.mkdir(parents=True, exist_ok=True)

    wanted = [s for s in args.only.split(",") if s]
    todo = []
    for doc_path, info in sorted(meta.items(), key=lambda kv: kv[1]["index"]):
        if wanted and not any(w in info["name"] for w in wanted):
            continue
        if (args.mesh_dir / info["mesh"]).exists():
            continue
        todo.append((doc_path, info))
    print(f"[sw] {len(todo)} part(s) to tessellate", flush=True)

    for doc_path, info in todo:
        t_part = time.perf_counter()
        part_doc = _soft(app, "OpenDoc", doc_path, SW_DOC_PART, default=None)
        if part_doc is None:
            print(f"  [warn] cannot open {doc_path}", file=sys.stderr)
            continue
        tris = part_triangles(part_doc)
        if len(tris) == 0:
            print(f"  [warn] no triangles for {info['name']}", file=sys.stderr)
            continue
        write_binary_stl(args.mesh_dir / info["mesh"], tris)
        vol = mesh_volume(tris)
        lo = tris.reshape(-1, 3).min(axis=0)
        hi = tris.reshape(-1, 3).max(axis=0)
        info.update({
            "triangles": int(len(tris)),
            "volume_m3": vol,
            "bbox_min_m": lo.round(6).tolist(),
            "bbox_max_m": hi.round(6).tolist(),
        })
        print(f"  [{info['index']:3d}/{len(meta)}] {info['name']:24s} tris={len(tris):7d} "
              f"vol={vol * 1e6:8.2f}cm3  {time.perf_counter() - t_part:5.1f}s", flush=True)
        # checkpoint after every part so an interrupted run resumes cleanly
        payload["meshes"] = meta
        args.json.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
        _soft(app, "CloseDoc", _soft(part_doc, "GetTitle", default=""))

    done = sum(1 for i in meta.values() if (args.mesh_dir / i["mesh"]).exists())
    print(f"\n{done}/{len(meta)} meshes present in {args.mesh_dir}")
    return 0 if done == len(meta) else 2


if __name__ == "__main__":
    raise SystemExit(main())
