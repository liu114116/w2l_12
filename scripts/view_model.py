#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""w2l_12 轮腿机器人 · MuJoCo 交互式查看器

用法（conda 环境 ms）:
    python scripts/view_model.py
    python scripts/view_model.py --model model/w2l_12.xml

两个窗口
--------
* MuJoCo 窗口: 左键拖拽旋转视角, 右键拖拽平移, 滚轮缩放;
  右上角可以切换 Camera (overview / side / front)、Joint、Control
* 关节面板: 四条腿各 4 个滑块 —— 侧摆(roll) / 大腿(hip) / 小腿(knee) / 轮子(wheel),
  拖动即可看到机构随动; 勾选 "四腿同步" 可同时调节四条腿

角度基准
--------
0° = 模型的站姿 (qpos = 0), 也就是大腿后倾 30°、小腿前倾 50° 的姿态。
滑块范围和模型里的关节限位一致。

这个查看器是**运动学**展示: 只改 qpos 再做 mj_forward, 不做物理积分,
所以拖关节时机器人不会倒、也不会自己滑走。
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

try:  # Windows 控制台默认 GBK, 直接 print 中文会炸
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # pragma: no cover
    pass

PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = PROJECT_DIR / "model" / "w2l_12.xml"

LEGS = [("FL", "左前"), ("FR", "右前"), ("RL", "左后"), ("RR", "右后")]
JOINTS = [
    ("roll", "侧摆", 90.0),
    ("hip", "大腿", 180.0),
    ("knee", "小腿", 135.0),
    ("wheel", "轮子", 180.0),
]
FONT = ("Microsoft YaHei UI", 9)


class JointPanel:
    """Tkinter slider panel driving the MuJoCo joint angles (polled each frame)."""

    def __init__(self, root, model: mujoco.MjModel, data: mujoco.MjData):
        import tkinter as tk
        from tkinter import ttk

        self.tk = tk
        self.ttk = ttk
        self.root = root
        self.model = model
        self.data = data
        self.viewer = None

        self.qadr: dict[tuple[str, str], int] = {}
        self.limits: dict[tuple[str, str], tuple[float, float]] = {}
        for leg, _ in LEGS:
            for key, _, default in JOINTS:
                jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{key}_{leg}")
                if jid < 0:
                    raise SystemExit(f"模型里找不到关节 {key}_{leg}")
                self.qadr[(key, leg)] = int(model.jnt_qposadr[jid])
                lo, hi = (math.degrees(v) for v in model.jnt_range[jid])
                self.limits[(key, leg)] = (lo, hi) if hi - lo > 1.0 else (-default, default)

        self.qpos0 = model.qpos0.copy()
        # every colliding geom of the robot (the floor plane is excluded): the
        # viewer keeps *all* of them above z = 0, not just the tyres
        self.robot_geoms = [g for g in range(model.ngeom)
                            if model.geom_contype[g] != 0
                            and model.geom_type[g] != mujoco.mjtGeom.mjGEOM_PLANE]
        self.mesh_verts = {}
        for g in self.robot_geoms:
            if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
                adr = int(model.geom_dataid[g])
                a0 = int(model.mesh_vertadr[adr])
                n = int(model.mesh_vertnum[adr])
                self.mesh_verts[g] = np.asarray(model.mesh_vert[a0:a0 + n], dtype=np.float64)
        self.box_corners = np.array([[sx, sy, sz] for sx in (-1.0, 1.0)
                                     for sy in (-1.0, 1.0) for sz in (-1.0, 1.0)])
        self.base_qpos = data.qpos.copy()
        self.values = {(key, leg): float(data.qpos[self.qadr[(key, leg)]])
                       for key, _, _ in JOINTS for leg, _ in LEGS}
        self.shown = dict(self.values)
        self.sliders: dict[tuple[str, str], object] = {}
        self.value_labels: dict[tuple[str, str], object] = {}
        self.sync_var = tk.BooleanVar(value=False)
        self.collision_var = tk.BooleanVar(value=False)
        self.status_var = tk.StringVar(value="")

        root.title("w2l_12 轮腿机器人 · 关节调节")
        root.resizable(False, False)
        root.geometry("+40+40")

        self._build_header()
        grid = ttk.Frame(root, padding=(8, 0, 8, 2))
        grid.pack(fill="both", expand=True)
        for column, (leg, label) in enumerate(LEGS):
            self._build_leg(grid, leg, label, column)
        self._build_footer(root)

    # ------------------------------------------------------------------
    def _build_header(self) -> None:
        ttk = self.ttk
        bar = ttk.Frame(self.root, padding=(10, 8, 10, 0))
        bar.pack(fill="x")
        ttk.Label(bar, text="拖动滑块调节关节角度 (度), 0 = 机械零点, 复位 = 站姿",
                  font=FONT).pack(side="left")
        ttk.Button(bar, text="复位", width=6, command=self.reset).pack(side="right", padx=(6, 0))
        ttk.Button(bar, text="蹲下", width=6,
                   command=lambda: self.apply_preset(roll=0.0, hip=-52.0, knee=118.0)
                   ).pack(side="right", padx=(6, 0))
        ttk.Button(bar, text="伸展", width=6,
                   command=lambda: self.apply_preset(roll=0.0, hip=0.0, knee=0.0)
                   ).pack(side="right", padx=(6, 0))
        ttk.Button(bar, text="外摆", width=6,
                   command=lambda: self.apply_preset(roll=22.0)
                   ).pack(side="right", padx=(6, 0))

    def _build_leg(self, parent, leg: str, label: str, column: int) -> None:
        tk, ttk = self.tk, self.ttk
        frame = ttk.LabelFrame(parent, text=f" {label} {leg} ", padding=(8, 2, 8, 6))
        frame.grid(row=0, column=column, padx=4, pady=4, sticky="nsew")
        frame.columnconfigure(1, weight=1)

        for row, (key, name, _) in enumerate(JOINTS):
            ttk.Label(frame, text=name, width=4, font=FONT).grid(row=row, column=0, sticky="w")
            lo, hi = self.limits[(key, leg)]
            scale = tk.Scale(
                frame, from_=lo, to=hi, resolution=0.5, orient="horizontal",
                length=190, showvalue=False, sliderlength=16, width=11,
            )
            scale.set(math.degrees(self.values[(key, leg)]))
            scale.grid(row=row, column=1, sticky="ew", padx=(4, 6))
            value = ttk.Label(frame, text=f"{math.degrees(self.values[(key, leg)]):+6.1f}°",
                              width=7, anchor="e", font=FONT)
            value.grid(row=row, column=2, sticky="e")
            self.sliders[(key, leg)] = scale
            self.value_labels[(key, leg)] = value

    def _build_footer(self, root) -> None:
        ttk = self.ttk
        bar = ttk.Frame(root, padding=(10, 4, 10, 8))
        bar.pack(fill="x")
        ttk.Checkbutton(bar, text="四腿同步", variable=self.sync_var).pack(side="left")
        ttk.Checkbutton(bar, text="显示碰撞体", variable=self.collision_var,
                        command=self._toggle_collision).pack(side="left", padx=(12, 0))
        ttk.Label(bar, textvariable=self.status_var, font=FONT,
                  foreground="#666").pack(side="right")

    def attach_viewer(self, viewer) -> None:
        self.viewer = viewer
        # hide the collision group unless the checkbox is on
        for group in (3, 4, 5):
            viewer.opt.geomgroup[group] = 0
        self.status_var.set(f"{self.model.nq} qpos · {self.model.nu} 执行器 · "
                            f"{mujoco.mj_getTotalmass(self.model):.2f} kg")

    # ------------------------------------------------------------------
    def _toggle_collision(self) -> None:
        if self.viewer is None:
            return
        on = 1 if self.collision_var.get() else 0
        self.viewer.opt.geomgroup[3] = on
        self.viewer.opt.geomgroup[4] = on

    def poll(self) -> None:
        changed: dict[tuple[str, str], float] = {}
        for key, scale in self.sliders.items():
            value = float(scale.get())
            if abs(value - self.values[key]) > 1e-9:
                self.values[key] = value
                changed[key] = value

        if changed and self.sync_var.get():
            for (key, leg), value in changed.items():
                for other, _ in LEGS:
                    if other != leg:
                        self.values[(key, other)] = value
                        self.sliders[(key, other)].set(value)

        for (key, leg), value in self.values.items():
            self.data.qpos[self.qadr[(key, leg)]] = math.radians(value)
            if abs(value - self.shown[(key, leg)]) > 0.05:
                self.shown[(key, leg)] = value
                self.value_labels[(key, leg)].configure(text=f"{value:+6.1f}°")
        self._ground()

    def _ground(self) -> None:
        """Keep the whole robot on the floor (this viewer is kinematic).

        Every colliding geom of the robot is considered, not just the tyres, so
        crouching far enough makes the chassis rest on the floor instead of
        sinking through it.
        """
        mujoco.mj_kinematics(self.model, self.data)
        low = self._lowest_point()
        if not np.isfinite(low):
            return
        self.data.qpos[2] -= low
        mujoco.mj_kinematics(self.model, self.data)

    def _lowest_point(self) -> float:
        """Lowest point of the robot's collision geometry, in world z."""
        m, d = self.model, self.data
        low = np.inf
        for gid in self.robot_geoms:
            pos = d.geom_xpos[gid]
            mat = d.geom_xmat[gid].reshape(3, 3)
            size = m.geom_size[gid]
            gtype = m.geom_type[gid]
            if gtype == mujoco.mjtGeom.mjGEOM_MESH:
                verts = self.mesh_verts.get(gid)
                if verts is None or len(verts) == 0:
                    continue
                z = float((verts @ mat[2]).min()) + pos[2]
            elif gtype == mujoco.mjtGeom.mjGEOM_BOX:
                z = float((self.box_corners * size @ mat[2]).min()) + pos[2]
            elif gtype == mujoco.mjtGeom.mjGEOM_CYLINDER:
                c = abs(float(mat[2, 2]))          # z component of the local z axis
                z = pos[2] - (size[1] * c + size[0] * math.sqrt(max(0.0, 1.0 - c * c)))
            elif gtype in (mujoco.mjtGeom.mjGEOM_SPHERE,):
                z = pos[2] - size[0]
            else:
                z = pos[2] - float(m.geom_rbound[gid])
            low = min(low, z)
        return low

    # ------------------------------------------------------------------
    def _set_all(self, key: str, value: float) -> None:
        for leg, _ in LEGS:
            self.values[(key, leg)] = value
            self.sliders[(key, leg)].set(value)
            self.shown[(key, leg)] = value
            self.value_labels[(key, leg)].configure(text=f"{value:+6.1f}°")
            self.data.qpos[self.qadr[(key, leg)]] = math.radians(value)

    def apply_preset(self, roll: float | None = None, hip: float | None = None,
                     knee: float | None = None) -> None:
        if roll is not None:
            # the roll axis is mirrored left/right, so the same number means the
            # same pose on both sides
            self._set_all("roll", roll)
        if hip is not None:
            self._set_all("hip", hip)
        if knee is not None:
            self._set_all("knee", knee)

    def reset(self) -> None:
        self.data.qpos[:] = self.base_qpos
        for key, _, _ in JOINTS:
            self._set_all(key, float(math.degrees(self.qpos0[self.qadr[(key, "FL")]])))


def main() -> int:
    ap = argparse.ArgumentParser(description="w2l_12 轮腿机器人 MuJoCo 交互查看器")
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="MJCF 模型路径")
    args = ap.parse_args()

    model_path = args.model if args.model.is_absolute() else (Path.cwd() / args.model)
    if not model_path.exists():
        print(f"[error] 找不到模型文件: {model_path}")
        print("        先运行: python src/create_model/build_model.py")
        return 1

    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    import tkinter as tk
    from tkinter import messagebox

    root = tk.Tk()
    panel = JointPanel(root, model, data)

    try:
        viewer = mujoco.viewer.launch_passive(model, data)
    except Exception as exc:  # pragma: no cover
        messagebox.showerror("MuJoCo 启动失败", f"{exc}\n\n请确认 OpenGL / 显卡驱动正常。")
        return 2

    viewer.cam.lookat[:] = [0.0, 0.0, 0.10]
    viewer.cam.distance = 0.62
    viewer.cam.azimuth = 135
    viewer.cam.elevation = -18
    panel.attach_viewer(viewer)

    # raise the control panel above the freshly created MuJoCo window
    root.update()
    root.attributes("-topmost", True)
    root.lift()
    root.update()
    root.after(1200, lambda: root.attributes("-topmost", False))
    panel.reset()  # Tk may have nudged the scales during layout

    state = {"closing": False}

    def on_close() -> None:
        state["closing"] = True
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)

    print(f"模型: {model_path}")
    print(f"  nbody={model.nbody} njnt={model.njnt} ngeom={model.ngeom} nu={model.nu} "
          f"总质量={mujoco.mj_getTotalmass(model):.3f} kg")
    print("拖动滑块即可转动侧摆/大腿/小腿/轮子; 关闭任一窗口退出。")

    try:
        while viewer.is_running():
            if state["closing"]:
                break
            try:
                root.update()
                if state["closing"]:
                    break
                panel.poll()
                mujoco.mj_forward(model, data)
                viewer.sync()
            except tk.TclError:
                break  # window was destroyed while we were polling it
            time.sleep(1.0 / 60.0)
    finally:
        try:
            viewer.close()
        except Exception:  # noqa: BLE001
            pass
        try:
            if not state["closing"]:
                root.destroy()
        except Exception:  # noqa: BLE001
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
