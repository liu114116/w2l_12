#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Self-check of the built w2l_12 model.

Reports
  * stance: wheel centres / hip positions / symmetry of the four legs
  * contacts at qpos = 0 (should be exactly the four tyres on the floor)
  * actuators: gain, limits
  * a 3 s gravity settle from the standing pose (does it stand or collapse?)

    python src/create_model/check_model.py
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_XML = PROJECT_DIR / "model" / "w2l_12.xml"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

LEGS = ("FL", "FR", "RL", "RR")


def main() -> int:
    import mujoco

    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", type=Path, default=DEFAULT_XML)
    ap.add_argument("--settle", type=float, default=3.0)
    args = ap.parse_args()

    model = mujoco.MjModel.from_xml_path(str(args.xml))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    name = lambda obj, i: mujoco.mj_id2name(model, obj, i) or ""   # noqa: E731

    print(f"model: nbody={model.nbody} njnt={model.njnt} ngeom={model.ngeom} "
          f"nu={model.nu} nq={model.nq} mass={mujoco.mj_getTotalmass(model):.4f} kg\n")

    print("=== stance (qpos0 = the standing pose) ===")
    # the tyre collision cylinder sits exactly at the tyre centre
    wheel_geom = {leg: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM,
                                         f"wheel_{leg}_collision") for leg in LEGS}
    print(f"{'leg':4s} {'roll':>8s} {'hip':>8s} {'knee':>8s}   {'tyre centre [mm]':>26s}")
    for leg in LEGS:
        jv = [math.degrees(data.qpos[int(model.jnt_qposadr[
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{k}_{leg}")])])
            for k in ("roll", "hip", "knee")]
        w = data.geom_xpos[wheel_geom[leg]] * 1e3
        print(f"{leg:4s} {jv[0]:8.2f} {jv[1]:8.2f} {jv[2]:8.2f}   "
              f"({w[0]:+7.1f},{w[1]:+7.1f},{w[2]:+7.1f})")

    tyre = np.array([data.geom_xpos[wheel_geom[leg]] for leg in LEGS])
    wx, wy, wz = tyre[:, 0], tyre[:, 1], tyre[:, 2]
    radius = float(model.geom_size[wheel_geom[LEGS[0]]][0])
    print(f"\ntyre height spread   {np.ptp(wz)*1e3:6.2f} mm "
          f"(ground clearance {wz.min()*1e3 - radius*1e3:+.2f} mm)")
    print(f"wheel base (x)       {np.ptp(wx)*1e3:6.2f} mm, track (y) {np.ptp(wy)*1e3:6.2f} mm")
    print(f"left/right symmetry  x err {abs(wx[0]-wx[1])*1e3:.2f} mm, "
          f"y err {abs(wy[0]+wy[1]+0.0)*1e3:.2f} mm")

    print("\n=== joint zero convention check ===")
    print("  roll 0 = leg straight down, + = swing OUT (abduction), range -90 ..  90 deg")
    print("  hip  0 = thigh straight down, + = swing to -x,         range -180 .. 180 deg")
    print("  knee 0 = calf in line with the thigh, + = swing to -x, range -135 .. 135 deg")
    adr = {name(mujoco.mjtObj.mjOBJ_JOINT, i): int(model.jnt_qposadr[i])
           for i in range(model.njnt)}
    zero = data.qpos.copy()
    mujoco.mj_resetData(model, data)
    for leg in LEGS:
        for k in ("roll", "hip", "knee"):
            data.qpos[adr[f"{k}_{leg}"]] = 0.0
    mujoco.mj_forward(model, data)

    def link_dir(p_from, a_from, p_to):
        v = p_to - p_from
        v = v - np.dot(v, a_from) * a_from
        return v / np.linalg.norm(v)

    def world_axis(body, joint):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint)
        return data.body(body).xmat.reshape(3, 3) @ model.jnt_axis[jid]

    for leg in LEGS:
        p1 = data.body(f"thigh_{leg}").xpos
        a1 = world_axis(f"thigh_{leg}", f"hip_{leg}")
        p2 = data.body(f"calf_{leg}").xpos
        a2 = world_axis(f"calf_{leg}", f"knee_{leg}")
        p3 = data.body(f"wheel_{leg}").xpos
        t = link_dir(p1, a1, p2)
        c = link_dir(p2, a2, p3)
        knee = math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(t, c))))))
        print(f"  {leg}: qpos=0 -> thigh=({t[0]:+.3f},{t[1]:+.3f},{t[2]:+.3f}) "
              f"(down), calf/thigh angle {knee:5.2f} deg (should be 0)")
    data.qpos[:] = zero
    mujoco.mj_forward(model, data)

    print("\n=== contacts at qpos0 ===")
    mujoco.mj_forward(model, data)
    pairs = []
    for i in range(data.ncon):
        c = data.contact[i]
        pairs.append((name(mujoco.mjtObj.mjOBJ_GEOM, c.geom1),
                      name(mujoco.mjtObj.mjOBJ_GEOM, c.geom2), float(c.dist)))
    if not pairs:
        print("  none")
    for g1, g2, d in pairs:
        print(f"  {g1:22s} <-> {g2:22s} dist={d*1e3:+6.2f} mm")
    print(f"  {len(pairs)} contact(s)")

    print("\n=== actuators ===")
    torque_model = all(float(model.actuator_biasprm[i, 1]) == 0.0
                       and float(model.actuator_biastype[i]) == 0 for i in range(model.nu))
    print(f"  mode: {'torque / current (ctrl = N*m)' if torque_model else 'servo (position + velocity)'}")
    for i in range(model.nu):
        nm = name(mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        jid = int(model.actuator_trnid[i, 0])
        jn = name(mujoco.mjtObj.mjOBJ_JOINT, jid)
        # both position and velocity servos are AFFINE; they differ in biasprm[1]
        kq = float(model.actuator_biasprm[i, 1])
        lo, hi = model.actuator_ctrlrange[i]
        fr = model.actuator_forcerange[i]
        if torque_model:
            print(f"  {nm:12s} torque   joint={jn:10s} "
                  f"force=[{fr[0]:+.2f},{fr[1]:+.2f}] N*m")
        else:
            kind = "position" if abs(kq) > 1e-9 else "velocity"
            print(f"  {nm:12s} {kind:8s} joint={jn:10s} gain={model.actuator_gainprm[i,0]:6.2f} "
                  f"kp={kq:+7.2f} kv={model.actuator_biasprm[i,2]:+7.2f} "
                  f"ctrl=[{lo:+.2f},{hi:+.2f}] force=[{fr[0]:+.2f},{fr[1]:+.2f}]")

    print(f"\n=== settle {args.settle:.1f}s from the standing pose ===")
    mujoco.mj_resetDataKeyframe(model, data, 0)
    start = data.qpos[:3].copy()
    start_quat = data.qpos[3:7].copy()
    if torque_model:
        # a bare torque model has nothing holding it up: use a simple PD on the
        # 12 leg joints and a damping term on the wheels
        print("  (torque model: holding with an external PD, kp=30, kd=1.8)")
        target = data.qpos.copy()
        jids = [int(model.actuator_trnid[i, 0]) for i in range(model.nu)]
        for _ in range(int(args.settle / model.opt.timestep)):
            for i, jid in enumerate(jids):
                q = data.qpos[model.jnt_qposadr[jid]]
                qd = data.qvel[model.jnt_dofadr[jid]]
                if name(mujoco.mjtObj.mjOBJ_JOINT, jid).startswith("wheel"):
                    data.ctrl[i] = -0.5 * qd
                else:
                    data.ctrl[i] = 30.0 * (target[model.jnt_qposadr[jid]] - q) - 1.8 * qd
            mujoco.mj_step(model, data)
    else:
        for _ in range(int(args.settle / model.opt.timestep)):
            mujoco.mj_step(model, data)
    drop = (data.qpos[2] - start[2]) * 1e3
    # tilt = angle between the body z axis and world z
    rot = data.qpos[3:7]
    w, x, y, z = rot
    body_z = np.array([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)])
    tilt = np.degrees(np.arccos(np.clip(body_z[2], -1.0, 1.0)))
    speed = float(np.linalg.norm(data.qvel[:3]))
    print(f"  base height {start[2]*1e3:7.2f} -> {data.qpos[2]*1e3:7.2f} mm  (drop {drop:+.2f} mm)")
    print(f"  body tilt from vertical {tilt:5.2f} deg, base speed {speed:.4f} m/s")
    print(f"  contacts now: {data.ncon}")
    maxpen = 0.0
    for i in range(data.ncon):
        maxpen = max(maxpen, -float(data.contact[i].dist))
    print(f"  max penetration {maxpen*1e3:.3f} mm")
    verdict = "stands" if (abs(drop) < 8.0 and tilt < 5.0 and speed < 0.12) else "NOT stable"
    print(f"  verdict: {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
