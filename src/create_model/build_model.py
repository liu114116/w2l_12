#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build the MuJoCo (MJCF) model of the w2l_12 wheel-legged robot from the CAD.

Inputs (produced by export_cad.py / derive_kinematics.py)
--------------------------------------------------------
  model/meshes/part_XXX_*.stl                  one binary STL per unique CAD part
  src/create_model/cad_export/cad_export.json  instance tree + part -> world transforms
  src/create_model/cad_export/kinematics.json  the 4-leg kinematic tree

What the script does
--------------------
  1. converts the CAD frame into a z-up robot frame (the CAD model is exported
     lying down:  x_robot = x_cad, y_robot = -z_cad, z_robot = y_cad),
  2. splits every CAD part into one of the 20 links (base + 4 x (hip bracket,
     thigh, calf, wheel)),
  3. re-poses the four legs, which the CAD stores at four arbitrary angles, into
     the **joint-zero pose** and fixes the joint coordinate conventions:
         roll  0 deg = thigh horizontal, pointing outwards (mirrored L/R axis),
               +90 = leg straight down,               range   0 .. 180 deg
         hip   0 deg = thigh straight down, + = forwards,  range -180 .. 180 deg
         knee  0 deg = calf in line with the thigh, + = forwards, range -135..135
  4. bakes the standing pose (--thigh-deg / --calf-deg) into the geometry and
     stores its joint angles in `ref`, so that qpos0 is the standing pose while
     qpos still follows the conventions above,
  5. puts each link frame on its joint axis, so every joint is a plain hinge
     about the CAD-measured axis,
  6. estimates the link masses / inertias from the part volumes with a density
     and a realistic-mass table (SolidWorks' own densities are placeholders),
  7. adds collision geometry (chassis box, thigh convex hulls, wheel cylinders)
     and 16 actuators (12 position-controlled servos, 4 wheel velocity drives),
  8. writes model/w2l_12.xml and drops the robot onto the floor plane.

Usage (conda env ms):
    python src/create_model/build_model.py
    python src/create_model/build_model.py --thigh-deg 25 --calf-deg 45
    python src/create_model/build_model.py --cad-pose          # keep the CAD pose
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[2]
EXPORT_JSON = PROJECT_DIR / "src" / "create_model" / "cad_export" / "cad_export.json"
KINEMATICS_JSON = PROJECT_DIR / "src" / "create_model" / "cad_export" / "kinematics.json"
MESH_DIR = PROJECT_DIR / "model" / "meshes"
DEFAULT_OUT = PROJECT_DIR / "model" / "w2l_12.xml"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

EPS = 1e-12
G = 9.81

# --------------------------------------------------------------------------
# robot constants
# --------------------------------------------------------------------------
LEG_ORDER = ("FL", "FR", "RL", "RR")
LEG_LABEL = {"FL": "左前", "FR": "右前", "RL": "左后", "RR": "右后"}

DEFAULT_THIGH_DEG = 30.0     # stance: hip -> knee leans backwards (-x) from vertical
DEFAULT_CALF_DEG = 50.0      # stance: knee -> wheel leans forwards (+x)

SERVO_TORQUE = 0.98          # N*m, STS3032 stall torque
WHEEL_TORQUE = 0.29          # N*m, 2208 hub motor peak
WHEEL_VMAX = 125.0           # rad/s, ~1200 rpm at 12 V

# joint ranges, in the zero convention below
#   roll: 0 = leg straight down (sagittal plane); positive = outwards (abduction)
#   hip : 0 = thigh straight down;    positive = thigh swings towards -x
#   knee: 0 = calf collinear with the thigh; positive = calf swings towards -x
ROLL_RANGE = (-math.pi / 2.0, math.pi / 2.0)
HIP_RANGE = (-math.pi, math.pi)
KNEE_RANGE = (-math.radians(135.0), math.radians(135.0))

# part name keyword -> density [kg/m^3] (used for the volume based mass)
DENSITY = {
    "电池": 2100.0,
    "舵机": 1200.0,
    "轮胎": 1100.0,
    "wheel": 1100.0,
    "轴承": 7800.0,
    "M3": 7800.0,
    "编码器": 1400.0,
    "主板": 1200.0,
    "驱动板": 1200.0,
}
DEFAULT_DENSITY = 1200.0     # printed / carbon parts

# part name keyword -> fixed mass [kg] (electronics are not solid printed parts)
FIXED_MASS = {
    "sts3032": 0.055,        # STS3032 bus servo
    "电机": 0.035,            # 2208 brushless hub motor
    "AS5600": 0.004,
    "MCU": 0.025,
    "电机驱动板": 0.030,
}
# keys are matched as substrings of the part name, longest first
FIXED_MASS_ORDER = ("电机驱动板", "sts3032", "AS5600", "MCU", "电机")

COLORS = {
    "tire": ((0.09, 0.09, 0.10, 1.0), "mat_tire"),
    "servo": ((0.14, 0.15, 0.17, 1.0), "mat_servo"),
    "motor": ((0.28, 0.29, 0.32, 1.0), "mat_motor"),
    "battery": ((0.10, 0.22, 0.55, 1.0), "mat_battery"),
    "pcb": ((0.08, 0.34, 0.16, 1.0), "mat_pcb"),
    "metal": ((0.62, 0.64, 0.68, 1.0), "mat_metal"),
    "printed": ((0.83, 0.84, 0.86, 1.0), "mat_printed"),
    "carbon": ((0.16, 0.17, 0.19, 1.0), "mat_carbon"),
}
COLLISION_RGBA = (0.95, 0.45, 0.10, 0.45)


# --------------------------------------------------------------------------
# small math helpers (quaternions are w, x, y, z - MuJoCo convention)
# --------------------------------------------------------------------------
def mat_to_quat(m: np.ndarray) -> np.ndarray:
    tr = float(np.trace(m))
    if tr > 0.0:
        s = math.sqrt(tr + 1.0) * 2.0
        q = np.array([0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s,
                      (m[1, 0] - m[0, 1]) / s])
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        q = np.array([(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s,
                      (m[0, 2] + m[2, 0]) / s])
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        q = np.array([(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s,
                      (m[1, 2] + m[2, 1]) / s])
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        q = np.array([(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s,
                      (m[1, 2] + m[2, 1]) / s, 0.25 * s])
    q /= np.linalg.norm(q)
    return q if q[0] >= 0 else -q


def axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, float)
    axis = axis / (np.linalg.norm(axis) + EPS)
    k = np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]])
    return np.eye(3) + math.sin(angle) * k + (1.0 - math.cos(angle)) * (k @ k)


def screw(axis: np.ndarray, point: np.ndarray, angle: float) -> np.ndarray:
    """4x4 rotation of `angle` about the line (point, axis)."""
    T = np.eye(4)
    T[:3, :3] = axis_angle(axis, angle)
    T[:3, 3] = point - T[:3, :3] @ point
    return T


def signed_angle(u: np.ndarray, v: np.ndarray, axis: np.ndarray) -> float:
    """Signed angle that rotates u onto v about `axis` (u, v perpendicular to it)."""
    axis = axis / np.linalg.norm(axis)
    return float(math.atan2(np.dot(np.cross(u, v), axis), np.dot(u, v)))


def roll_axis_angle(roll_axis: np.ndarray, pitch_axis: np.ndarray) -> float:
    """Abduction angle that rotates `pitch_axis` about `roll_axis` until the
    pitch axis has no vertical component, i.e. the leg plane is sagittal."""
    r = np.asarray(roll_axis, float)
    r = r / np.linalg.norm(r)
    u = np.asarray(pitch_axis, float)
    u = u / np.linalg.norm(u)
    along = float(np.dot(u, r))
    a = u - along * r                     # part of u that turns with the roll
    w = np.cross(r, u)                    # its derivative
    # z(rotated u) = along*r_z + A cos(d) + B sin(d) = 0
    A, B, C = a[2], w[2], along * r[2]
    norm = math.hypot(A, B)
    if norm < 1e-9:
        return 0.0
    phi = math.atan2(B, A)
    delta = math.acos(max(-1.0, min(1.0, -C / norm)))
    c1, c2 = phi + delta, phi - delta
    wrap = lambda x: (x + math.pi) % (2 * math.pi) - math.pi   # noqa: E731
    return c1 if abs(wrap(c1)) <= abs(wrap(c2)) else c2


def link_direction(p_from: np.ndarray, p_to: np.ndarray, axis: np.ndarray) -> np.ndarray:
    """Unit direction between two hinge axes, perpendicular to the distal axis."""
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    d = np.asarray(p_to, float) - np.asarray(p_from, float)
    d = d - np.dot(d, a) * a
    return d / (np.linalg.norm(d) + EPS)


def fmt(values, nd: int = 6) -> str:
    out = []
    for v in np.atleast_1d(np.asarray(values, float)):
        if abs(v) < 1e-12:
            v = 0.0
        s = f"{v:.{nd}f}".rstrip("0").rstrip(".")
        out.append(s if s not in ("", "-0") else "0")
    return " ".join(out)


# --------------------------------------------------------------------------
# CAD data
# --------------------------------------------------------------------------
def read_stl(path: Path) -> np.ndarray:
    raw = path.read_bytes()
    n = int(np.frombuffer(raw[80:84], np.uint32)[0])
    rec = np.frombuffer(raw[84:84 + n * 50],
                        dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", 9), ("a", "<u2")]))
    return rec["v"].astype(np.float64).reshape(-1, 3, 3)


@dataclass
class Part:
    inst: dict
    name: str
    stem: str
    mesh: str
    tris: np.ndarray              # part frame
    tf: np.ndarray = field(default_factory=lambda: np.eye(4))   # part -> model frame
    mass: float = 0.0
    volume: float = 0.0
    com: np.ndarray = field(default_factory=lambda: np.zeros(3))  # part frame
    inertia: np.ndarray = field(default_factory=lambda: np.zeros((3, 3)))  # part frame, about com

    @property
    def material(self) -> str:
        return choose_material(self.name, self.stem)

    @property
    def com_world(self) -> np.ndarray:
        return self.tf[:3, :3] @ self.com + self.tf[:3, 3]


@dataclass
class Link:
    name: str
    parts: list[Part] = field(default_factory=list)
    frame: np.ndarray = field(default_factory=lambda: np.eye(4))   # link frame in model frame
    mass: float = 0.0
    com: np.ndarray = field(default_factory=lambda: np.zeros(3))   # link frame
    inertia: np.ndarray = field(default_factory=lambda: np.zeros((3, 3)))
    geoms: list[str] = field(default_factory=list)
    joints: list[str] = field(default_factory=list)
    children: list["Link"] = field(default_factory=list)
    parent: "Link | None" = None
    pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    quat: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0]))


@dataclass
class Joint:
    name: str
    kind: str
    axis: np.ndarray
    point: np.ndarray
    parent: str
    child: str
    value: float = 0.0            # joint angle that was baked into the geometry
    ref: float = 0.0              # joint angle of the standing pose (qpos0)
    rng: tuple[float, float] = (-math.pi, math.pi)


def fixed_mass(name: str) -> float | None:
    for key in FIXED_MASS_ORDER:
        if key in name:
            return FIXED_MASS[key]
    return None


def density_of(name: str) -> float:
    for key, rho in DENSITY.items():
        if key in name:
            return rho
    return DEFAULT_DENSITY


def choose_material(name: str, stem: str) -> str:
    if stem in ("wheel",) or "轮胎" in name:
        return "mat_tire"
    if "sts3032" in stem:
        return "mat_servo"
    if "AS5600" in stem or "MCU" in stem or "驱动板" in stem:
        return "mat_pcb"
    if stem == "电机":
        return "mat_motor"
    if "电池" in stem:
        return "mat_battery"
    if "轴承" in stem or "M3" in stem or "螺母" in stem:
        return "mat_metal"
    return "mat_printed"


def mesh_inertia(tris: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """(volume, centre of mass, inertia about the centre of mass) of a watertight
    triangle soup, with unit density and the mesh in metres."""
    v = tris.reshape(-1, 3)
    uniq, inverse = np.unique(np.round(v, 9), axis=0, return_inverse=True)
    faces = inverse.reshape(-1, 3)
    if len(uniq) < 4 or len(faces) < 4:
        return 0.0, v.mean(0), np.zeros((3, 3))
    a, b, c = uniq[faces[:, 0]], uniq[faces[:, 1]], uniq[faces[:, 2]]
    vols = np.einsum("ij,ij->i", a, np.cross(b, c)) / 6.0
    vol = float(vols.sum())
    if abs(vol) < 1e-15:
        return 0.0, v.mean(0), np.zeros((3, 3))
    com = ((a + b + c) / 4.0 * vols[:, None]).sum(0) / vol
    # second moment about the origin (tetrahedra from the origin)
    a0, b0, c0 = a - com, b - com, c - com
    vv = np.einsum("ij,ij->i", a0, np.cross(b0, c0)) / 6.0
    aa = np.einsum("ij,ik->ijk", a0, a0)
    bb = np.einsum("ij,ik->ijk", b0, b0)
    cc = np.einsum("ij,ik->ijk", c0, c0)
    ab = np.einsum("ij,ik->ijk", a0, b0)
    bc = np.einsum("ij,ik->ijk", b0, c0)
    ca = np.einsum("ij,ik->ijk", c0, a0)
    # inertia of a tetrahedron (origin, a, b, c) with unit density
    C = (aa + bb + cc + ab + ab + bc + bc + ca + ca) / 120.0
    C = 0.5 * (C + C.transpose(0, 2, 1))
    Itet = np.trace(C, axis1=1, axis2=2)[:, None, None] * np.eye(3) - C
    I = (Itet * vv[:, None, None]).sum(0)
    I = 0.5 * (I + I.T)
    return abs(vol) * np.sign(vol), com, I


class CadModel:
    def __init__(self) -> None:
        export = json.loads(EXPORT_JSON.read_text(encoding="utf-8"))
        kin = json.loads(KINEMATICS_JSON.read_text(encoding="utf-8"))
        self.export = export
        self.kin = kin
        self.stems = {info["mesh"]: info["name"] for info in export["meshes"].values()}
        self.R = np.asarray(kin["cad_to_robot_rotation"], float)
        self.parts: list[Part] = []
        cache: dict[str, np.ndarray] = {}
        for inst in export["instances"]:
            if "mesh" not in inst:
                continue
            mesh = inst["mesh"]
            if mesh not in cache:
                cache[mesh] = read_stl(MESH_DIR / mesh)
            tris = cache[mesh]
            stem = self.stems[mesh]
            m = fixed_mass(stem)
            if m is None:
                vol, com, inertia = mesh_inertia(tris)
                m = max(vol, 0.0) * density_of(stem)
            else:
                vol, com, inertia = mesh_inertia(tris)
                if abs(vol) > 1e-12:
                    inertia = inertia * (m / abs(vol))
            tf = np.asarray(inst["transform"], float).copy()
            tf[:3, :3] = self.R @ tf[:3, :3]
            tf[:3, 3] = self.R @ tf[:3, 3]
            self.parts.append(Part(inst=inst, name=inst["name"], stem=stem, mesh=mesh,
                                   tris=tris, tf=tf, mass=float(m), volume=abs(vol),
                                   com=np.asarray(com, float), inertia=np.asarray(inertia, float)))
        self.center_on_chassis()

    def center_on_chassis(self) -> None:
        """Put the model origin under the middle of the chassis (CAD origin is
        somewhere beside the robot)."""
        base = [p for p in self.parts if p.inst["group"] == "body-1"]
        pts = np.vstack([p.tris.reshape(-1, 3) @ p.tf[:3, :3].T + p.tf[:3, 3] for p in base])
        shift = np.array([(pts[:, 0].min() + pts[:, 0].max()) / 2,
                          (pts[:, 1].min() + pts[:, 1].max()) / 2,
                          0.0])
        for part in self.parts:
            part.tf[:3, 3] -= shift
        self.shift = shift

    def by_group(self, group: str) -> list[Part]:
        return [p for p in self.parts if p.inst["group"] == group]


# --------------------------------------------------------------------------
# kinematics
# --------------------------------------------------------------------------
def leg_chain(joints: dict[str, Joint], tag: str,
              q_roll: float, q_hip: float, q_knee: float) -> tuple[np.ndarray, ...]:
    """Cumulative transforms of the three leg joints from the zero pose."""
    jr, jh, jk = joints[f"roll_{tag}"], joints[f"hip_{tag}"], joints[f"knee_{tag}"]
    M_roll = screw(jr.axis, jr.point, q_roll)
    M_hip = M_roll @ screw(jh.axis, jh.point, q_hip)
    M_knee = M_hip @ screw(jk.axis, jk.point, q_knee)
    return M_roll, M_hip, M_knee


def apply_leg_pose(links: dict[str, Link], joints: dict[str, Joint], tag: str,
                   q_roll: float, q_hip: float, q_knee: float) -> None:
    """Bake a leg pose into the part transforms and the joint frames."""
    M_roll, M_hip, M_knee = leg_chain(joints, tag, q_roll, q_hip, q_knee)
    for name, M in (("hip", M_roll), ("thigh", M_hip),
                    ("calf", M_knee), ("wheel", M_knee)):
        for part in links[f"{name}_{tag}"].parts:
            part.tf = M @ part.tf
    j_hip, j_knee = joints[f"hip_{tag}"], joints[f"knee_{tag}"]
    j_wheel = joints[f"wheel_{tag}"]
    for joint, M in ((j_hip, M_roll), (j_knee, M_hip), (j_wheel, M_knee)):
        joint.point = M[:3, :3] @ joint.point + M[:3, 3]
        joint.axis = M[:3, :3] @ joint.axis


def rotate_about(joint: Joint, angle: float) -> np.ndarray:
    return screw(joint.axis, joint.point, angle)


def flip_to(joint: Joint, test) -> None:
    """Make `test(axis)` true by flipping the joint axis sign (only the sign of
    the joint coordinate changes, the geometry is untouched)."""
    if not test(joint.axis):
        joint.axis = -joint.axis


def bake_zero_pose(links: dict[str, Link], joints: dict[str, Joint]) -> None:
    """Re-pose every leg into the joint-zero pose:

      * roll  = 0,  the leg plane is the sagittal plane (leg hangs vertically)
      * hip   = 0,  the thigh points straight down
      * knee  = 0,  the calf is collinear with the thigh (straight leg)
    """
    for tag in LEG_ORDER:
        j_roll = joints[f"roll_{tag}"]
        j_hip = joints[f"hip_{tag}"]
        j_knee = joints[f"knee_{tag}"]
        j_wheel = joints[f"wheel_{tag}"]

        thigh_dir = link_direction(j_hip.point, j_knee.point, j_knee.axis)
        calf_dir = link_direction(j_knee.point, j_wheel.point, j_wheel.axis)

        # 1) roll: rotate the leg plane back into the sagittal plane, i.e. make
        #    the hip pitch axis exactly lateral again
        d_roll = roll_axis_angle(j_roll.axis, j_hip.axis)
        M_roll = rotate_about(j_roll, d_roll)

        # 2) hip: thigh to straight down (the pitch axis is lateral by now, so
        #    the rotation stays in the x-z plane)
        cur = M_roll[:3, :3] @ thigh_dir
        tgt = np.array([0.0, 0.0, -1.0])
        d_hip = signed_angle(cur, tgt, M_roll[:3, :3] @ j_hip.axis)
        M_hip = M_roll @ rotate_about(j_hip, d_hip)

        # 3) knee: calf collinear with the thigh
        cur_calf = M_hip[:3, :3] @ calf_dir
        tgt_calf = M_hip[:3, :3] @ thigh_dir
        d_knee = signed_angle(cur_calf, tgt_calf, M_hip[:3, :3] @ j_knee.axis)

        apply_leg_pose(links, joints, tag, d_roll, d_hip, d_knee)
        j_roll.value, j_hip.value, j_knee.value = d_roll, d_hip, d_knee


def orient_axes(links: dict[str, Link], joints: dict[str, Joint]) -> None:
    """Fix the sign of every joint axis so that the joint coordinate means what
    the robot convention says, independent of how the CAD axis happened to point:

        roll  > 0  ->  the leg swings outwards (abduction)
        hip   > 0  ->  the thigh swings towards -x
        knee  > 0  ->  the calf swings towards -x relative to the thigh

    The tests are evaluated in the joint-zero pose (legs straight down).
    """
    probe = 0.05
    for tag in LEG_ORDER:
        j_roll, j_hip = joints[f"roll_{tag}"], joints[f"hip_{tag}"]
        j_knee, j_wheel = joints[f"knee_{tag}"], joints[f"wheel_{tag}"]
        side = 1.0 if tag[1] == "L" else -1.0
        thigh_dir = link_direction(j_hip.point, j_knee.point, j_knee.axis)
        calf_dir = link_direction(j_knee.point, j_wheel.point, j_wheel.axis)

        # roll: positive must move the thigh outwards (away from the centre
        # line). The left leg sits at +y, the right one at -y, and the axes are
        # mirrored, so both sides abduct with the same positive value.
        def roll_ok(axis, j=j_roll, d=thigh_dir):
            moved = screw(axis, j.point, probe)[:3, :3] @ d
            return side * moved[1] > side * d[1]
        flip_to(j_roll, roll_ok)

        # hip: positive must move the thigh towards -x
        def hip_ok(axis, j=j_hip, d=thigh_dir):
            moved = screw(axis, j.point, probe)[:3, :3] @ d
            return moved[0] < d[0]
        flip_to(j_hip, hip_ok)

        # knee: positive must move the calf towards -x relative to the thigh
        def knee_ok(axis, j=j_knee, d=calf_dir):
            moved = screw(axis, j.point, probe)[:3, :3] @ d
            return moved[0] < d[0]
        flip_to(j_knee, knee_ok)

        # the wheel has no preferred sign; keep it parallel to the knee axis so
        # that all the leg joints are consistent
        if float(np.dot(j_wheel.axis, j_knee.axis)) < 0.0:
            j_wheel.axis = -j_wheel.axis


def solve_stance(joints: dict[str, Joint], links: dict[str, Link],
                 thigh_deg: float, calf_deg: float) -> dict[str, dict[str, float]]:
    """Joint angles (the `ref` values) of the standing pose, expressed in the
    zero convention: roll 0 / hip 0 = leg straight down, knee 0 = straight leg."""
    out: dict[str, dict[str, float]] = {}
    for tag in LEG_ORDER:
        j_roll, j_hip = joints[f"roll_{tag}"], joints[f"hip_{tag}"]
        j_knee, j_wheel = joints[f"knee_{tag}"], joints[f"wheel_{tag}"]
        thigh_dir = link_direction(j_hip.point, j_knee.point, j_knee.axis)
        calf_dir = link_direction(j_knee.point, j_wheel.point, j_wheel.axis)

        # 1) roll stays at 0: the standing leg plane is the sagittal plane
        q_roll = 0.0
        M_roll = np.eye(4)

        # 2) hip: thigh from straight down to `thigh_deg` towards +x
        tgt_thigh = np.array([math.sin(math.radians(thigh_deg)), 0.0,
                              -math.cos(math.radians(thigh_deg))])
        cur = M_roll[:3, :3] @ thigh_dir
        q_hip = signed_angle(cur, tgt_thigh, M_roll[:3, :3] @ j_hip.axis)
        M_hip = leg_chain(joints, tag, q_roll, q_hip, 0.0)[1]

        # 3) knee: calf to `calf_deg` towards -x (forwards)
        tgt_calf = np.array([-math.sin(math.radians(calf_deg)), 0.0,
                             -math.cos(math.radians(calf_deg))])
        cur_calf = M_hip[:3, :3] @ calf_dir
        q_knee = signed_angle(cur_calf, tgt_calf, M_hip[:3, :3] @ j_knee.axis)
        out[tag] = {"roll": q_roll, "hip": q_hip, "knee": q_knee}
    return out


def build_tree(cad: CadModel, thigh_deg: float, calf_deg: float,
               cad_pose: bool) -> tuple[Link, dict[str, Joint], dict[str, Link]]:
    legs = cad.kin["legs"]
    base = Link("base", parts=list(cad.by_group("body-1")))
    links: dict[str, Link] = {"base": base}
    joints: dict[str, Joint] = {}

    # name the legs FL / FR / RL / RR from the hip positions in the robot frame
    def leg_tag(leg: dict) -> str:
        hip_point = np.asarray(leg["joints"][0]["point"], float)   # the roll bearing
        c = cad.R @ hip_point - cad.shift
        front = c[0] < 0.0          # the robot's front is -x
        left = c[1] > 0.0
        return ("F" if front else "R") + ("L" if left else "R")

    for leg in legs:
        tag = leg_tag(leg)
        while tag in [k.split("_")[1] for k in joints if k.startswith("roll_")]:
            tag = tag[0] + ("R" if tag[1] == "L" else "L")     # pathological fallback
        hip = Link(f"hip_{tag}", parts=list(cad.by_group(leg["hips"])))
        thigh = Link(f"thigh_{tag}", parts=list(cad.by_group(leg["thigh"])))
        calf = Link(f"calf_{tag}", parts=list(cad.by_group(leg["calf"])))
        wheel = Link(f"wheel_{tag}", parts=list(cad.by_group(leg["wheel"])))
        base.children.append(hip)
        hip.children.append(thigh)
        thigh.children.append(calf)
        calf.children.append(wheel)
        for child, parent in ((hip, base), (thigh, hip), (calf, thigh), (wheel, calf)):
            child.parent = parent
        links.update({hip.name: hip, thigh.name: thigh, calf.name: calf, wheel.name: wheel})

        kinds = ("hip_roll", "hip_pitch", "knee", "wheel")
        for kind, child in zip(kinds, (hip, thigh, calf, wheel)):
            jdata = next(j for j in leg["joints"] if j["kind"] == kind)
            axis = cad.R @ np.asarray(jdata["axis"], float)
            point = cad.R @ np.asarray(jdata["point"], float) - cad.shift
            name = {"hip_roll": f"roll_{tag}", "hip_pitch": f"hip_{tag}",
                    "knee": f"knee_{tag}", "wheel": f"wheel_{tag}"}[kind]
            joints[name] = Joint(name=name, kind=kind, axis=axis, point=point,
                                 parent=child.parent.name, child=child.name)

    # ---- bake the joint-zero pose (and its axis sign convention) ----------
    if not cad_pose:
        bake_zero_pose(links, joints)
        orient_axes(links, joints)

    # ---- joint `ref`: the standing pose ----------------------------------
    # MuJoCo applies (qpos - ref) to a hinge, so with the *standing* pose baked
    # into the geometry and ref = the stance angles:
    #     qpos0 = ref      -> the robot loads standing,
    #     qpos = 0         -> the mechanical zero (legs horizontal, straight),
    #     qpos             -> exactly the angle convention below.
    refs = ({tag: {"roll": 0.0, "hip": 0.0, "knee": 0.0} for tag in LEG_ORDER}
            if cad_pose else solve_stance(joints, links, thigh_deg, calf_deg))
    for tag in LEG_ORDER:
        joints[f"roll_{tag}"].ref = refs[tag]["roll"]
        joints[f"hip_{tag}"].ref = refs[tag]["hip"]
        joints[f"knee_{tag}"].ref = refs[tag]["knee"]
        if not cad_pose:
            apply_leg_pose(links, joints, tag, refs[tag]["roll"],
                           refs[tag]["hip"], refs[tag]["knee"])

    # ---- link frames: origin on the joint axis, world-aligned rotation ----
    base.frame = np.eye(4)
    for tag in LEG_ORDER:
        for name in (f"hip_{tag}", f"thigh_{tag}", f"calf_{tag}", f"wheel_{tag}"):
            link = links[name]
            jname = {"hip": "roll", "thigh": "hip", "calf": "knee", "wheel": "wheel"}[
                name.split("_")[0]]
            joint = joints[f"{jname}_{tag}"]
            link.frame = np.eye(4)
            link.frame[:3, 3] = joint.point
            joint.point = np.zeros(3)          # the joint sits at the link origin

    # joint ranges, in the new zero convention
    for tag in LEG_ORDER:
        joints[f"roll_{tag}"].rng = ROLL_RANGE
        joints[f"hip_{tag}"].rng = HIP_RANGE
        joints[f"knee_{tag}"].rng = KNEE_RANGE
        joints[f"wheel_{tag}"].rng = (-1e6, 1e6)
    return base, joints, links


def link_dynamics(link: Link) -> None:
    m = 0.0
    acc = np.zeros(3)
    for part in link.parts:
        m += part.mass
        acc += part.mass * part.com_world
    if m <= 0:
        link.mass, link.com, link.inertia = 1e-4, np.zeros(3), np.eye(3) * 1e-7
        return
    com_w = acc / m
    I = np.zeros((3, 3))
    for part in link.parts:
        d = part.com_world - com_w
        I += part.tf[:3, :3] @ part.inertia @ part.tf[:3, :3].T
        I += part.mass * (np.dot(d, d) * np.eye(3) - np.outer(d, d))
    link.mass = m
    link.com = com_w - link.frame[:3, 3]
    link.inertia = I


# --------------------------------------------------------------------------
# geometry for collisions
# --------------------------------------------------------------------------
def part_vertices_world(link: Link) -> np.ndarray:
    return np.vstack([p.tris.reshape(-1, 3) @ p.tf[:3, :3].T + p.tf[:3, 3] for p in link.parts])


def local_vertices(link: Link) -> np.ndarray:
    v = part_vertices_world(link)
    return v - link.frame[:3, 3]


def wheel_cylinder(link: Link) -> tuple[float, float, float]:
    """(tire radius, half width, lateral centre) of a wheel link.

    The wheel link frame sits on the wheel axle with the model orientation
    (z up, y lateral), so the radius is taken in the x-z plane and the width
    along y.  The wheel part origin is on the tyre's inner face, so the tyres are
    offset laterally in the link frame.
    """
    v = local_vertices(link)
    r = float(np.max(np.linalg.norm(v[:, [0, 2]], axis=1)))
    lo, hi = float(v[:, 1].min()), float(v[:, 1].max())
    half = max(0.5 * (hi - lo), 1e-3)
    return r, half, 0.5 * (lo + hi)


# --------------------------------------------------------------------------
# MJCF writer
# --------------------------------------------------------------------------
class Writer:
    """Writes one MJCF flavour: ``servo`` (position + velocity servos, what the
    viewer and simple controllers use) or ``torque`` (bare torque/current
    actuators for RL / MPC)."""

    def __init__(self, base: Link, joints: dict[str, Joint], links: dict[str, Link],
                 base_height: float, actuator_mode: str = "servo"):
        self.actuator_mode = actuator_mode
        self.base = base
        self.joints = joints
        self.links = links
        self.base_height = base_height
        # hinge order = the order the bodies appear in the MJCF (depth first),
        # which is exactly the order of their qpos entries
        self.hinge_names: list[str] = []

        def walk(link: Link) -> None:
            for child in link.children:
                joint = next(j for j in self.joints.values() if j.child == child.name)
                self.hinge_names.append(joint.name)
                walk(child)

        walk(self.base)
        # `ctrl` on the other hand is ordered by the <actuator> section below
        self.actuator_names = ([f"{kind}_{tag}" for tag in LEG_ORDER
                                for kind in ("roll", "hip", "knee")]
                               + [f"wheel_{tag}" for tag in LEG_ORDER])
        self.lines: list[str] = []

    def add(self, line: str = "") -> None:
        self.lines.append(line)

    # ---------------------------------------------------------------
    def write(self) -> str:
        self.header()
        self.assets()
        self.world()
        self.body_tree()
        self.actuators()
        self.sensors()
        self.keyframe()
        self.add("</mujoco>")
        return "\n".join(self.lines)

    def header(self) -> None:
        a = self.add
        a('<?xml version="1.0" encoding="utf-8"?>')
        a("<!-- w2l_12 wheel-legged robot - generated by src/create_model/build_model.py")
        a("     source: SolidWorks w2l_12.SLDASM -> per-part STL + MJCF (see README)")
        a("     links: base + 4 x (hip bracket, thigh, calf, wheel)")
        a("     joints: 4 x (roll abduction, hip pitch, knee, wheel) = 16 DOF")
        a(f"     actuators: {self.actuator_mode} -->")
        a('<mujoco model="w2l_12">')
        a('  <compiler angle="radian" meshdir="meshes" autolimits="true"/>')
        a('  <option timestep="0.002" integrator="implicitfast" cone="elliptic" impratio="10"'
          ' gravity="0 0 -9.81"/>')
        a('  <size njmax="4000" nconmax="800"/>')
        a('  <statistic center="0 0 0.11" extent="0.28"/>')
        a("  <visual>")
        a('    <global offwidth="1920" offheight="1080"/>')
        a('    <headlight ambient="0.34 0.34 0.37" diffuse="0.62 0.62 0.62" specular="0.12 0.12 0.12"/>')
        a('    <quality shadowsize="4096" offsamples="8"/>')
        a('    <map znear="0.004" zfar="8" shadowclip="1.5"/>')
        a("  </visual>")
        a("  <default>")
        a('    <joint type="hinge" damping="0.06" armature="0.002" frictionloss="0.01"/>')
        a('    <default class="visual">')
        a('      <geom contype="0" conaffinity="0" group="1" density="0" material="mat_printed"/>')
        a("    </default>")
        a('    <default class="collision">')
        a('      <geom contype="1" conaffinity="1" group="3" condim="3" friction="0.7 0.02 0.002"')
        a('            solref="0.006 1" solimp="0.9 0.95 0.001" rgba="0.95 0.45 0.10 0.40"/>')
        a("    </default>")
        a('    <default class="wheel_collision">')
        a('      <geom contype="1" conaffinity="1" group="3" condim="4" friction="1.2 0.02 0.002"')
        a('            solref="0.004 1" solimp="0.92 0.96 0.001" rgba="0.95 0.45 0.10 0.40"/>')
        a("    </default>")
        a('    <default class="servo">')
        a(f'      <position kp="26" kv="1.6" forcerange="-{SERVO_TORQUE} {SERVO_TORQUE}"/>')
        a("    </default>")
        a('    <default class="wheelmotor">')
        a(f'      <velocity kv="0.7" forcerange="-{WHEEL_TORQUE} {WHEEL_TORQUE}"/>')
        a("    </default>")
        if self.actuator_mode == "torque":
            a('    <default class="servo_torque">')
            a(f'      <motor gear="1" forcerange="-{SERVO_TORQUE} {SERVO_TORQUE}"/>')
            a("    </default>")
            a('    <default class="wheel_torque">')
            a(f'      <motor gear="1" forcerange="-{WHEEL_TORQUE} {WHEEL_TORQUE}"/>')
            a("    </default>")
        a("  </default>")

    def assets(self) -> None:
        a = self.add
        a("  <asset>")
        a('    <texture name="tex_grid" type="2d" builtin="checker" rgb1="0.26 0.28 0.30"')
        a('             rgb2="0.33 0.35 0.38" width="512" height="512"/>')
        a('    <material name="mat_floor" texture="tex_grid" texrepeat="8 8" reflectance="0.05"/>')
        for key, (rgba, mat) in COLORS.items():
            extra = ' reflectance="0.2"' if key == "metal" else ""
            a(f'    <material name="{mat}" rgba="{fmt(rgba, 3)}" specular="0.32" shininess="0.3"{extra}/>')
        meshes = sorted({p.mesh[:-4] for link in self.links.values() for p in link.parts})
        for mesh in meshes:
            a(f'    <mesh name="{mesh}" file="{mesh}.stl"/>')
        a("  </asset>")

    def world(self) -> None:
        a = self.add
        a("  <worldbody>")
        a('    <light name="key" pos="0.7 0.5 1.5" dir="-0.4 -0.3 -1" directional="true"')
        a('           diffuse="0.72 0.72 0.72" specular="0.22 0.22 0.22" castshadow="true"/>')
        a('    <light name="fill" pos="-0.9 -0.7 0.9" dir="0.5 0.45 -1" directional="true"')
        a('           diffuse="0.26 0.28 0.32" specular="0.04 0.04 0.04" castshadow="false"/>')
        a('    <geom name="floor" type="plane" size="4 4 0.02" material="mat_floor" group="0"')
        a('          contype="1" conaffinity="1" friction="1.0 0.02 0.002"/>')
        a('    <camera name="overview" pos="0.46 -0.50 0.34" xyaxes="0.74 0.67 0 -0.20 0.22 0.95"/>')
        a('    <camera name="side" pos="0 0.66 0.14" xyaxes="-1 0 0 0 0 1"/>')
        a('    <camera name="front" pos="0.66 0 0.14" xyaxes="0 1 0 0 0 1"/>')
        a("  </worldbody>")

    def body_tree(self) -> None:
        a = self.add
        # the worldbody block is reopened so that base is inside it
        self.lines[-1] = "  </worldbody>"
        a("  <worldbody>")
        a(f'    <body name="base" pos="0 0 {self.base_height:.6f}">')
        a('      <freejoint name="root"/>')
        a('      <site name="imu" pos="0 0 0" size="0.006"/>')
        self.inertial(self.base, 3)
        self.visuals(self.base, 3)
        self.collision_base(3)
        for link in self.base.children:
            self.body(link, 4)
        a("    </body>")
        a("  </worldbody>")

    def body(self, link: Link, indent: int) -> None:
        a = self.add
        pad = "  " * indent
        joint = next(j for j in self.joints.values() if j.child == link.name)
        parent = self.links[link.parent.name]
        pos = link.frame[:3, 3] - parent.frame[:3, 3]
        a(f'{pad}<body name="{link.name}" pos="{fmt(pos)}">')
        self.inertial(link, indent + 1)
        self.visuals(link, indent + 1)
        self.collision_link(link, indent + 1)
        jtype = 'type="hinge"'
        rng = f'range="{joint.rng[0]:.5f} {joint.rng[1]:.5f}"'
        ref = f'ref="{joint.ref:.5f}"'
        if link.name.startswith("wheel"):
            a(f'{pad}  <joint name="{joint.name}" {jtype} axis="{fmt(joint.axis)}"'
              f' damping="0.002" armature="0.0002" frictionloss="0.004"/>')
        else:
            a(f'{pad}  <joint name="{joint.name}" {jtype} axis="{fmt(joint.axis)}" '
              f'{rng} {ref}/>')
        for child in link.children:
            self.body(child, indent + 1)
        a(f"{pad}</body>")

    def inertial(self, link: Link, indent: int) -> None:
        pad = "  " * indent
        I = link.inertia
        self.add(f'{pad}<inertial pos="{fmt(link.com)}" mass="{link.mass:.6f}" '
                 f'fullinertia="{I[0, 0]:.9g} {I[1, 1]:.9g} {I[2, 2]:.9g} '
                 f'{I[0, 1]:.9g} {I[0, 2]:.9g} {I[1, 2]:.9g}"/>')

    def visuals(self, link: Link, indent: int) -> None:
        pad = "  " * indent
        for part in link.parts:
            pos = part.tf[:3, 3] - link.frame[:3, 3]
            quat = mat_to_quat(part.tf[:3, :3])
            self.add(f'{pad}<geom class="visual" type="mesh" mesh="{part.mesh[:-4]}" '
                     f'material="{part.material}" pos="{fmt(pos)}" quat="{fmt(quat)}"/>')

    def collision_base(self, indent: int) -> None:
        pad = "  " * indent
        v = local_vertices(self.base)
        lo, hi = v.min(0), v.max(0)
        # the visual box also holds the protruding servos; shrink a little
        size = (hi - lo) * np.array([0.97, 0.93, 0.92])
        self.add(f'{pad}<geom name="base_collision" class="collision" type="box" '
                 f'size="{fmt(size / 2)}" pos="{fmt((lo + hi) / 2)}"/>')

    def collision_link(self, link: Link, indent: int) -> None:
        pad = "  " * indent
        if link.name.startswith("wheel"):
            r, half, cy = wheel_cylinder(link)
            joint = next(j for j in self.joints.values() if j.child == link.name)
            # MuJoCo puts a cylinder's axis on the local z, the wheel axle is the
            # hinge axis (lateral), so rotate z onto it
            axis = np.asarray(joint.axis, float)
            axis = axis / np.linalg.norm(axis)
            z = np.array([0.0, 0.0, 1.0])
            v = np.cross(z, axis)
            s = float(np.linalg.norm(v))
            c = float(np.dot(z, axis))
            if s < 1e-9:
                quat = np.array([1.0, 0.0, 0.0, 0.0]) if c > 0 else np.array([0.0, 1.0, 0.0, 0.0])
            else:
                half_angle = math.atan2(s, c) / 2.0
                quat = np.concatenate([[math.cos(half_angle)], v / s * math.sin(half_angle)])
            self.add(f'{pad}<geom name="{link.name}_collision" class="wheel_collision" '
                     f'type="cylinder" size="{r:.5f} {half:.5f}" pos="0 {fmt(cy)} 0" '
                     f'quat="{fmt(quat)}"/>')
            return
        if not link.name.startswith("thigh"):
            return                      # only the base, thighs and wheels collide
        # printed brackets: collide the real walls (convex hulls of the parts)
        for part in link.parts:
            if "sts3032" in part.stem:
                continue
            pos = part.tf[:3, 3] - link.frame[:3, 3]
            quat = mat_to_quat(part.tf[:3, :3])
            self.add(f'{pad}<geom class="collision" type="mesh" mesh="{part.mesh[:-4]}" '
                     f'pos="{fmt(pos)}" quat="{fmt(quat)}"/>')

    def actuators(self) -> None:
        a = self.add
        a("  " + ("<!-- torque / current mode: ctrl = joint torque [N*m], "
                  "the drives accept it directly -->"
                  if self.actuator_mode == "torque" else
                  "<!-- position servos (legs) + velocity drives (wheels) -->"))
        a("  <actuator>")
        for tag in LEG_ORDER:
            for name in (f"roll_{tag}", f"hip_{tag}", f"knee_{tag}"):
                if self.actuator_mode == "torque":
                    a(f'    <motor name="{name}" joint="{name}" class="servo_torque"/>')
                else:
                    j = self.joints[name]
                    a(f'    <position name="{name}" joint="{name}" class="servo"'
                      f' ctrlrange="{j.rng[0]:.4f} {j.rng[1]:.4f}"/>')
        for tag in LEG_ORDER:
            if self.actuator_mode == "torque":
                a(f'    <motor name="wheel_{tag}" joint="wheel_{tag}" class="wheel_torque"/>')
            else:
                a(f'    <velocity name="wheel_{tag}" joint="wheel_{tag}" class="wheelmotor"'
                  f' ctrlrange="{-WHEEL_VMAX} {WHEEL_VMAX}"/>')
        a("  </actuator>")

    def sensors(self) -> None:
        a = self.add
        a("  <sensor>")
        a('    <framequat name="base_quat" objtype="body" objname="base"/>')
        a('    <gyro name="base_gyro" site="imu"/>')
        a('    <accelerometer name="base_acc" site="imu"/>')
        a("  </sensor>")

    def keyframe(self) -> None:
        a = self.add
        n = len(LEG_ORDER) * 4
        a("  <keyframe>")
        # "stand": qpos0, i.e. the joint `ref` values = the standing pose
        q = [0.0, 0.0, self.base_height, 1.0, 0.0, 0.0, 0.0]
        q += [self.joints[name].ref for name in self.hinge_names]
        c = ([0.0] * len(self.actuator_names) if self.actuator_mode == "torque"
             else [self.joints[name].ref for name in self.actuator_names])
        a(f'    <key name="stand" qpos="{fmt(q)}" ctrl="{fmt(c)}"/>')
        # "zero": every joint at its mechanical zero (legs horizontal, straight)
        q = [0.0, 0.0, self.base_height, 1.0, 0.0, 0.0, 0.0] + [0.0] * len(self.hinge_names)
        c = [0.0] * len(self.actuator_names)
        a(f'    <key name="zero" qpos="{fmt(q)}" ctrl="{fmt(c)}"/>')
        a("  </keyframe>")


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--actuators", choices=("torque", "servo"), default="torque",
                    help="torque/current mode (default) or position/velocity servos")
    ap.add_argument("--thigh-deg", type=float, default=DEFAULT_THIGH_DEG)
    ap.add_argument("--calf-deg", type=float, default=DEFAULT_CALF_DEG)
    ap.add_argument("--cad-pose", action="store_true", help="keep the pose of the CAD assembly")
    args = ap.parse_args()

    cad = CadModel()
    print(f"CAD parts: {len(cad.parts)} instances, {len({p.mesh for p in cad.parts})} meshes")

    base, joints, links = build_tree(cad, args.thigh_deg, args.calf_deg, args.cad_pose)
    for link in links.values():
        link_dynamics(link)

    # settle the robot on the floor: lowest point of the four tyres -> z = 0
    # (the baked geometry is the standing pose, so this is the stance height)
    base_height = -min(part_vertices_world(links[f"wheel_{tag}"])[:, 2].min()
                       for tag in LEG_ORDER)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(Writer(base, joints, links, base_height, args.actuators).write(),
                        encoding="utf-8")
    print(f"\nwrote {args.out}   (actuators: {args.actuators})")
    print(f"{'link':12s} {'mass[g]':>8s} {'parts':>6s} {'com z[mm]':>10s}")
    for link in links.values():
        print(f"{link.name:12s} {link.mass*1e3:8.2f} {len(link.parts):6d} {link.com[2]*1e3:10.1f}")
    total = sum(l.mass for l in links.values())
    print(f"{'TOTAL':12s} {total*1e3:8.2f}")

    print("\njoint zero conventions (deg):")
    print("  roll : 0 = leg straight down, + = swing OUT (abduction), range -90 ..  90")
    print("  hip  : 0 = thigh straight down, + = swing to -x,        range -180 .. 180")
    print("  knee : 0 = calf in line with the thigh, + = swing to -x, range -135 .. 135")
    print(f"\nstanding pose (qpos0 = ref), base height {base_height*1e3:.2f} mm")
    for tag in LEG_ORDER:
        r = math.degrees(joints[f"roll_{tag}"].ref)
        h = math.degrees(joints[f"hip_{tag}"].ref)
        k = math.degrees(joints[f"knee_{tag}"].ref)
        print(f"  {tag}: roll={r:+7.2f}  hip={h:+7.2f}  knee={k:+7.2f}")

    import mujoco
    model = mujoco.MjModel.from_xml_path(str(args.out))
    print(f"\nmodel ok: nbody={model.nbody} njnt={model.njnt} ngeom={model.ngeom} "
          f"nu={model.nu} nq={model.nq} mass={mujoco.mj_getTotalmass(model):.4f} kg")

    # verify the conventions against the loaded model at qpos0
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    wz = []
    for tag in LEG_ORDER:
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"wheel_{tag}_collision")
        wz.append(float(data.geom_xpos[gid][2]) - float(model.geom_size[gid][0]))
    print(f"tyre contact heights at qpos0: " + "  ".join(f"{z*1e3:+.2f}" for z in wz)
          + f" mm (spread {np.ptp(wz)*1e3:.2f} mm)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
