#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Intermediate: derive the 4-leg kinematic tree of the CAD assembly.

The SolidWorks assembly has no usable mates (the parts are placed freely), so the
kinematics is reconstructed from the *mating features*:

  * every servo has a rotation axis that is exactly coaxial (0.03 mm) with the
    bearing of the link it drives,
  * every wheel is coaxial (0.00 mm) with the motor + encoder inside the calf,
  * the knee axis is the calf servo axis, which is parallel to the thigh bearing
    axis and passes through the thigh's distal bore.

Part-frame feature definitions (verified against the SolidWorks mate `重合1`,
which joins a body servo face at part (183.1, 167.8, 9.1) mm with the hip
bearing, so the servo axis is the part +Z line through (183.1, 167.8) mm):

    sts3032_*      axis = part +Z through (183.1, 167.8) mm
    单轴轴承        axis = part +Y through (0, 0)
    wheel          axis = part +X through (0, 0)
    电机            axis = part +Z through (0, 0)
    AS5600编码器    axis = part +Y through (0, 0)

Output: src/create_model/cad_export/kinematics.json

    python src/create_model/derive_kinematics.py
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[2]
EXPORT_JSON = PROJECT_DIR / "src" / "create_model" / "cad_export" / "cad_export.json"
MESH_DIR = PROJECT_DIR / "model" / "meshes"
OUT_JSON = PROJECT_DIR / "src" / "create_model" / "cad_export" / "kinematics.json"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# CAD -> robot frame: the CAD model has the chassis plane in x-z, i.e. the robot
# is "lying down"; z_robot = y_cad, y_robot = -z_cad, x_robot = x_cad.
R_CAD_TO_ROBOT = np.array([[1.0, 0.0, 0.0],
                           [0.0, 0.0, -1.0],
                           [0.0, 1.0, 0.0]])

# part stem -> (point in part frame [m], axis direction in part frame)
FEATURES = {
    "sts3032_单轴": (np.array([0.1831, 0.1678, 0.0]), np.array([0.0, 0.0, 1.0])),
    "sts3032_双轴": (np.array([0.1831, 0.1678, 0.0]), np.array([0.0, 0.0, 1.0])),
    "单轴轴承": (np.zeros(3), np.array([0.0, 1.0, 0.0])),
    "wheel": (np.zeros(3), np.array([1.0, 0.0, 0.0])),
    "电机": (np.zeros(3), np.array([0.0, 0.0, 1.0])),
    "AS5600编码器": (np.zeros(3), np.array([0.0, 1.0, 0.0])),
}


@dataclass
class Feature:
    inst: dict
    stem: str
    point: np.ndarray          # on the axis line, CAD/world frame
    axis: np.ndarray           # unit
    kind: str

    @property
    def name(self) -> str:
        return self.inst["name"]

    @property
    def group(self) -> str:
        return self.inst["group"]

    @property
    def tf(self) -> np.ndarray:
        return np.asarray(self.inst["transform"], float)


def kind_of(stem: str) -> str:
    if "sts3032" in stem:
        return "servo"
    if "轴承" in stem:
        return "bearing"
    if stem == "wheel":
        return "wheel"
    if stem == "电机":
        return "motor"
    if "编码器" in stem:
        return "encoder"
    return "other"


def load_features() -> tuple[list[Feature], dict]:
    data = json.loads(EXPORT_JSON.read_text(encoding="utf-8"))
    stems = {info["mesh"]: info["name"] for info in data["meshes"].values()}
    feats = []
    for inst in data["instances"]:
        if "mesh" not in inst:
            continue
        stem = stems[inst["mesh"]]
        if stem not in FEATURES:
            continue
        p_local, a_local = FEATURES[stem]
        tf = np.asarray(inst["transform"], float)
        point = tf[:3, :3] @ p_local + tf[:3, 3]
        axis = tf[:3, :3] @ a_local
        axis = axis / np.linalg.norm(axis)
        if axis[np.argmax(np.abs(axis))] < 0:
            axis = -axis
        feats.append(Feature(inst=inst, stem=stem, point=point, axis=axis, kind=kind_of(stem)))
    return feats, data


def line_metrics(p1, a1, p2, a2) -> tuple[float, float]:
    """(angle between lines [deg], shortest distance between them [m])."""
    cross = np.cross(a1, a2)
    s = float(np.linalg.norm(cross))
    angle = float(np.degrees(np.arctan2(s, abs(float(np.dot(a1, a2))))))
    if s < 1e-9:
        w = p2 - p1
        return angle, float(np.linalg.norm(w - np.dot(w, a1) * a1))
    return angle, abs(float(np.dot(p2 - p1, cross / s)))


def point_line_distance(p, p0, a) -> float:
    w = p - p0
    return float(np.linalg.norm(w - np.dot(w, a) * a))


def main() -> int:
    feats, data = load_features()
    servos = [f for f in feats if f.kind == "servo"]
    bearings = [f for f in feats if f.kind == "bearing"]
    wheels = [f for f in feats if f.kind == "wheel"]
    motors = [f for f in feats if f.kind == "motor"]

    # --- J1: body servo <-> hip bearing (coaxial) ---------------------------
    j1 = {}
    for bearing in bearings:
        if not bearing.group.startswith(("ld_hip", "rd_hip")):
            continue
        best = None
        for servo in servos:
            if servo.group != "body-1":
                continue
            ang, dist = line_metrics(servo.point, servo.axis, bearing.point, bearing.axis)
            if ang < 2.0 and dist < 1.0e-3 and (best is None or dist < best[1]):
                best = (servo, dist)
        j1[bearing.group] = {"bearing": bearing, "servo": best[0] if best else None}

    # --- J2: hip servo <-> thigh bearing (coaxial) --------------------------
    j2 = {}
    for bearing in bearings:
        if not (bearing.group.startswith("left_thigh") or bearing.group.startswith("right_thigh")):
            continue
        best = None
        for servo in servos:
            if not servo.group.startswith(("ld_hip", "rd_hip")):
                continue
            ang, dist = line_metrics(servo.point, servo.axis, bearing.point, bearing.axis)
            if ang < 2.0 and dist < 1.0e-3 and (best is None or dist < best[1]):
                best = (servo, dist)
        j2[bearing.group] = {"bearing": bearing, "servo": best[0] if best else None}

    # --- J4: wheel <-> calf motor (coaxial) --------------------------------
    j4 = {}
    for wheel in wheels:
        best = None
        for motor in motors:
            ang, dist = line_metrics(motor.point, motor.axis, wheel.point, wheel.axis)
            if ang < 2.0 and dist < 1.0e-3 and (best is None or dist < best[1]):
                best = (motor, dist)
        j4[wheel.group] = {"wheel": wheel, "motor": best[0] if best else None}

    # --- J3: calf servo <-> thigh (parallel + passes through the thigh) ----
    calf_servos = [s for s in servos if s.group.startswith("calf")]
    thigh_groups = [b.group for b in bearings
                    if b.group.startswith(("left_thigh", "right_thigh"))]
    thigh_tris: dict[str, np.ndarray] = {}
    inst_by_group = {}
    for inst in data["instances"]:
        if "mesh" in inst:
            inst_by_group.setdefault(inst["group"], []).append(inst)

    def group_points(group: str) -> np.ndarray:
        pts = []
        for inst in inst_by_group.get(group, []):
            raw = (MESH_DIR / inst["mesh"]).read_bytes()
            n = int(np.frombuffer(raw[80:84], np.uint32)[0])
            rec = np.frombuffer(raw[84:84 + n * 50],
                                dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", 9), ("a", "<u2")]))
            v = rec["v"].astype(np.float64).reshape(-1, 3)
            tf = np.asarray(inst["transform"], float)
            pts.append(v @ tf[:3, :3].T + tf[:3, 3])
        return np.vstack(pts)

    j3 = {}
    for servo in calf_servos:
        best = None
        for group in thigh_groups:
            bearing = j2[group]["bearing"]
            ang, _ = line_metrics(servo.point, servo.axis, bearing.point, bearing.axis)
            if ang > 3.0:
                continue
            if group not in thigh_tris:
                thigh_tris[group] = group_points(group)
            w = thigh_tris[group] - servo.point
            d = np.linalg.norm(w - np.outer(w @ servo.axis, servo.axis), axis=1)
            dmin = float(d.min())
            if best is None or dmin < best[2]:
                best = (group, bearing, dmin)
        j3[servo.group] = {"servo": servo, "thigh": best[0] if best else None,
                           "bearing": best[1] if best else None,
                           "surface_gap_mm": best[2] * 1e3 if best else None}

    # --- assemble the four chains -----------------------------------------
    legs = []
    for hip_group, j1i in j1.items():
        servo1 = j1i["servo"]
        bearing1 = j1i["bearing"]
        # the servo inside this hip bracket drives the thigh
        servo2 = next((f for f in servos if f.group == hip_group), None)
        thigh_group = next((g for g, v in j2.items() if v["servo"] is servo2), None)
        if thigh_group is None:
            print(f"[warn] no thigh for {hip_group}")
            continue
        bearing2 = j2[thigh_group]["bearing"]
        calf_group = next((g for g, v in j3.items() if v["thigh"] == thigh_group), None)
        if calf_group is None:
            print(f"[warn] no calf for {thigh_group}")
            continue
        servo3 = j3[calf_group]["servo"]
        wheel_group = next((g for g, v in j4.items()
                            if v["motor"] is not None and v["motor"].group == calf_group), None)
        wheel = j4[wheel_group]["wheel"]

        axis1 = bearing1.axis                     # hip roll  (cad x)
        axis2 = bearing2.axis                     # hip pitch (cad z)
        axis3 = servo3.axis                       # knee      (cad z, posed)
        axis4 = wheel.axis                        # wheel

        entry = {
            "name": hip_group.split("-")[0].replace("_hip", "") + hip_group.split("-")[-1],
            "hips": hip_group,
            "thigh": thigh_group,
            "calf": calf_group,
            "wheel": wheel_group,
            "joints": [
                {"kind": "hip_roll", "axis": axis1.tolist(), "point": bearing1.point.tolist()},
                {"kind": "hip_pitch", "axis": axis2.tolist(), "point": bearing2.point.tolist()},
                {"kind": "knee", "axis": axis3.tolist(), "point": servo3.point.tolist()},
                {"kind": "wheel", "axis": axis4.tolist(), "point": wheel.point.tolist()},
            ],
        }
        legs.append(entry)

    legs.sort(key=lambda e: e["name"])
    print(f"{len(legs)} legs assembled\n")
    for leg in legs:
        print(f"--- {leg['name']}: {leg['hips']} / {leg['thigh']} / {leg['calf']} / {leg['wheel']}")
        for j in leg["joints"]:
            p = np.array(j["point"]) * 1e3
            a = np.array(j["axis"])
            print(f"    {j['kind']:10s} p=({p[0]:+7.1f},{p[1]:+7.1f},{p[2]:+7.1f}) "
                  f"a=({a[0]:+.3f},{a[1]:+.3f},{a[2]:+.3f})")

    print("\nnote: the CAD stores the four legs at four arbitrary angles; the common\n"
          "      stance is solved geometrically by build_model.py, not from these axes.")

    payload = {"legs": legs, "cad_to_robot_rotation": R_CAD_TO_ROBOT.tolist()}
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nwrote {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
