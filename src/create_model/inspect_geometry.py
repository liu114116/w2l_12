#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Intermediate: print geometric facts about the exported CAD assembly
(part sizes, centres, principal axes) used to derive the kinematic tree.

    python src/create_model/inspect_geometry.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[2]
EXPORT_JSON = PROJECT_DIR / "src" / "create_model" / "cad_export" / "cad_export.json"
MESH_DIR = PROJECT_DIR / "model" / "meshes"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def read_stl(path: Path) -> np.ndarray:
    raw = path.read_bytes()
    n = int(np.frombuffer(raw[80:84], np.uint32)[0])
    rec = np.frombuffer(raw[84:84 + n * 50],
                        dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", 9), ("a", "<u2")]))
    return rec["v"].astype(np.float64).reshape(-1, 3, 3)


_cache: dict[str, np.ndarray] = {}


def mesh(name: str) -> np.ndarray:
    if name not in _cache:
        _cache[name] = read_stl(MESH_DIR / name)
    return _cache[name]


def pca_axes(verts: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    c = verts.mean(0)
    cov = np.cov((verts - c).T)
    w, v = np.linalg.eigh(cov)
    return c, v, np.sqrt(np.maximum(w, 0.0))   # ascending: v[:,0] = smallest spread


def main() -> int:
    data = json.loads(EXPORT_JSON.read_text(encoding="utf-8"))
    insts = [i for i in data["instances"] if "mesh" in i]
    print(f"{len(insts)} part instances\n")

    rows = []
    for inst in insts:
        tf = np.asarray(inst["transform"], float)
        tris = mesh(inst["mesh"])
        w = tris @ tf[:3, :3].T + tf[:3, 3]
        v = w.reshape(-1, 3)
        lo, hi = v.min(0) * 1e3, v.max(0) * 1e3
        c, axes, spread = pca_axes(v * 1e3)
        rows.append((inst["group"], inst["name"], (lo + hi) / 2, hi - lo, spread, axes))

    for group in sorted({r[0] for r in rows}):
        print(f"--- {group} ---")
        for g, name, c, size, spread, axes in rows:
            if g != group:
                continue
            ax = axes[:, 0]     # smallest-spread direction
            ax = ax if ax[np.argmax(np.abs(ax))] > 0 else -ax
            print(f"  {name:22s} c=({c[0]:+7.1f},{c[1]:+7.1f},{c[2]:+7.1f}) "
                  f"size=({size[0]:6.1f},{size[1]:6.1f},{size[2]:6.1f}) "
                  f"thin_axis=({ax[0]:+.2f},{ax[1]:+.2f},{ax[2]:+.2f}) "
                  f"spread=({spread[0]:5.1f},{spread[1]:5.1f},{spread[2]:5.1f})")
        print()

    # global extents
    allv = np.vstack([(mesh(i["mesh"]).reshape(-1, 3) @ np.asarray(i["transform"], float)[:3, :3].T
                       + np.asarray(i["transform"], float)[:3, 3]) for i in insts]) * 1e3
    print(f"TOTAL bbox {np.round(allv.min(0),1)} .. {np.round(allv.max(0),1)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
