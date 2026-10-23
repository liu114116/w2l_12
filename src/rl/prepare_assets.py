"""把 model/w2l_12.xml 转成训练用的轻量 MJCF。

原始模型有 149 万个三角形的视觉网格（两个舵机模型就占了 27 万面），
单进程编译/加载要几十 MB，开 16 个并行环境会吃爆内存。
训练只需要碰撞体和惯量，所以这里：

1. 删掉所有 ``class="visual"`` 的 geom（contype=0，不参与物理）；
2. 删掉没有被任何 geom 引用的 mesh 资源；
3. 把 16 个 ``motor`` 换成 MuJoCo 的 position / velocity 伺服
   （kp/kv 直接复用 XML 里已有的 ``servo`` / ``wheelmotor`` default class），
   这样就等价于真机上"舵机力矩环里跑 PD"的那一层，而且隐式积分更稳。

生成物：``src/rl/assets/w2l_12_train.xml``（相对原文件，网格不复制）

用法::

    python src/rl/prepare_assets.py            # 位置/速度伺服版（默认，用来训练）
    python src/rl/prepare_assets.py --torque   # 纯力矩版（保留 motor）
    python src/rl/prepare_assets.py --keep-visual
"""

from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

RL_DIR = Path(__file__).resolve().parent
PROJECT = RL_DIR.parents[1]
SRC_XML = PROJECT / "model" / "w2l_12.xml"
ASSET_DIR = RL_DIR / "assets"

LEG_JOINTS = {f"{k}_{leg}" for leg in ("FL", "FR", "RL", "RR") for k in ("roll", "hip", "knee")}


def build(out_path: Path, torque: bool, keep_visual: bool) -> Path:
    tree = ET.parse(SRC_XML)
    root = tree.getroot()

    # ---------------------------------------------------------------- 1. 删视觉体
    removed_geoms = 0
    removed_assets = 0
    if not keep_visual:
        for parent in root.iter():
            for geom in list(parent.findall("geom")):
                if geom.get("class") == "visual":
                    parent.remove(geom)
                    removed_geoms += 1

    # ---------------------------------------------------------------- 2. 清无用 mesh
    used_meshes: set[str] = set()
    for geom in root.iter("geom"):
        if geom.get("type") == "mesh" or geom.get("mesh"):
            if geom.get("mesh"):
                used_meshes.add(geom.get("mesh"))
    asset = root.find("asset")
    if asset is not None and not keep_visual:
        for mesh in list(asset.findall("mesh")):
            if mesh.get("name") not in used_meshes:
                asset.remove(mesh)
                removed_assets += 1

    # ---------------------------------------------------------------- 3. 换执行器
    actuator = root.find("actuator")
    if actuator is None:
        raise RuntimeError("找不到 <actuator>")
    actuator.clear()
    for name in [f"{k}_{leg}" for leg in ("FL", "FR", "RL", "RR") for k in ("roll", "hip", "knee")]:
        el = ET.SubElement(actuator, "motor" if torque else "position")
        el.set("name", name)
        el.set("joint", name)
        el.set("class", "servo_torque" if torque else "servo")
    for leg in ("FL", "FR", "RL", "RR"):
        name = f"wheel_{leg}"
        el = ET.SubElement(actuator, "motor" if torque else "velocity")
        el.set("name", name)
        el.set("joint", name)
        el.set("class", "wheel_torque" if torque else "wheelmotor")

    # ---------------------------------------------------------------- 4. 资源路径
    compiler = root.find("compiler")
    if compiler is not None:
        # 生成物在 src/rl/assets/，网格在 model/meshes/，用相对路径保证可迁移
        compiler.set("meshdir", "../../../model/meshes")
    root.set("model", "w2l_12_train")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(out_path, encoding="utf-8", xml_declaration=True)

    print(f"[prepare] {SRC_XML}")
    print(f"[prepare] -> {out_path}")
    print(f"[prepare] 删除视觉 geom {removed_geoms} 个, 无用 mesh {removed_assets} 个, "
          f"执行器 {'motor(力矩)' if torque else 'position/velocity(伺服)'}")
    return out_path


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--torque", action="store_true", help="保留纯力矩 motor 执行器")
    p.add_argument("--keep-visual", action="store_true", help="保留视觉网格（慢，只用于渲染）")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    out = args.out
    if out is None:
        suffix = "_torque" if args.torque else ""
        suffix += "_visual" if args.keep_visual else ""
        out = ASSET_DIR / f"w2l_12_train{suffix}.xml"
    build(out.resolve(), args.torque, args.keep_visual)

    # 自检：能不能编译、执行器数量对不对
    import mujoco

    m = mujoco.MjModel.from_xml_path(str(out))
    print(f"[check] nq={m.nq} nv={m.nv} nu={m.nu} nbody={m.nbody} ngeom={m.ngeom}")
    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(m.nu)]
    print(f"[check] actuators: {names}")
    assert m.nu == 16, m.nu
    if not args.keep_visual:
        assert m.ngeom < 40, f"视觉体没删干净: {m.ngeom}"
    return 0


if __name__ == "__main__":
    sys.exit(main())
