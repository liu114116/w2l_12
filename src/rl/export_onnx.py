"""把训练好的 SB3 策略导出成 ONNX，并做数值对齐校验。

    python src/rl/export_onnx.py --tag v1
    python src/rl/export_onnx.py --checkpoint src/rl/checkpoints/v1/model_5000000.zip

产物：
    src/rl/models/policy.onnx      给 onnxruntime / TensorRT / 下位机用
    src/rl/models/policy.onnx.json  输入输出说明 + 权重哈希
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import yaml  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402

from robot_cfg import (  # noqa: E402
    ACTION_SCALE_LEG,
    DEFAULT_LEG_POS,
    JOINT_POS_HIGH,
    JOINT_POS_LOW,
    STAND_REG,
    WHEEL_VEL_SCALE,
)

RL_DIR = _HERE
MODEL_DIR = RL_DIR / "models"


class PolicyNet(torch.nn.Module):
    """只保留 actor 的确定性前向：obs(53) -> 归一化动作(16) ∈ [-1,1]。"""

    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        features = self.policy.extract_features(obs, self.policy.pi_features_extractor)
        latent = self.policy.mlp_extractor.forward_actor(features)
        return self.policy.action_net(latent)


class PolicyTargets(torch.nn.Module):
    """把动作缩放一起烘进网络：obs(53) -> **12 个腿部目标角(rad) + 4 个轮子目标角速度(rad/s)**。

    这就是 RC_WheelLeg / legged_gym 那一层的输出语义：
        q_des = clip(default_leg_pos + a[:12] * action_scale, joint_low, joint_high)
        w_des = a[12:] * wheel_vel_scale
    部署端拿到 16 个数就能直接写进舵机/驱动器，不用再记一遍尺度。
    """

    def __init__(self, policy):
        super().__init__()
        self.actor = PolicyNet(policy)
        self.register_buffer("default_pos", torch.tensor(DEFAULT_LEG_POS, dtype=torch.float32))
        self.register_buffer("scale", torch.tensor(ACTION_SCALE_LEG, dtype=torch.float32))
        self.register_buffer("low", torch.tensor(JOINT_POS_LOW, dtype=torch.float32))
        self.register_buffer("high", torch.tensor(JOINT_POS_HIGH, dtype=torch.float32))
        self.wheel_scale = float(WHEEL_VEL_SCALE)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        a = torch.clamp(self.actor(obs), -1.0, 1.0)
        q = torch.clamp(self.default_pos + a[:, :12] * self.scale, min=self.low, max=self.high)
        w = a[:, 12:] * self.wheel_scale
        return torch.cat([q, w], dim=-1)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()[:16]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--tag", default="v1")
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--opset", type=int, default=17)
    args = p.parse_args()

    ckpt = (args.checkpoint or (RL_DIR / "checkpoints" / args.tag / "model.zip")).resolve()
    if not ckpt.exists():
        print(f"[错误] 找不到 {ckpt}")
        return 1
    out = args.out or (MODEL_DIR / "policy.onnx")
    out.parent.mkdir(parents=True, exist_ok=True)

    model = PPO.load(str(ckpt), device="cpu")
    policy = model.policy
    policy.eval()
    net_norm = PolicyNet(policy).eval()
    net_tgt = PolicyTargets(policy).eval()

    dummy = torch.zeros(1, 53, dtype=torch.float32)
    out_norm = out if out.name != "policy.onnx" else out.with_name("policy_norm.onnx")

    # ---- 1) 归一化动作版（和训练里的 action 一模一样，便于对照开源实现）----
    torch.onnx.export(
        net_norm, dummy, str(out_norm),
        input_names=["obs"], output_names=["action"],
        dynamic_axes={"obs": {0: "batch"}, "action": {0: "batch"}},
        opset_version=args.opset, do_constant_folding=True,
    )
    # ---- 2) 物理目标版：12 个腿部目标角(rad) + 4 个轮子目标角速度(rad/s) ----
    torch.onnx.export(
        net_tgt, dummy, str(out),
        input_names=["obs"], output_names=["target"],
        dynamic_axes={"obs": {0: "batch"}, "target": {0: "batch"}},
        opset_version=args.opset, do_constant_folding=True,
    )
    print(f"[onnx] {out.name}      12 目标角 + 4 目标轮速 (rad, rad/s)  {out.stat().st_size/1024:.1f} KB")
    print(f"[onnx] {out_norm.name} 归一化动作 [-1,1]                     {out_norm.stat().st_size/1024:.1f} KB")

    # ---------------- 数值校验 ----------------
    try:
        import onnxruntime as ort

        sess_t = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
        sess_n = ort.InferenceSession(str(out_norm), providers=["CPUExecutionProvider"])
        rng = np.random.default_rng(0)
        worst_t = worst_n = worst_consistency = 0.0
        for scale in (0.1, 1.0, 5.0):
            x = (rng.standard_normal((64, 53)) * scale).astype(np.float32)
            with torch.no_grad():
                t_ref = net_tgt(torch.from_numpy(x)).numpy()
                n_ref = net_norm(torch.from_numpy(x)).numpy()
            t_out = sess_t.run(["target"], {"obs": x})[0]
            n_out = sess_n.run(["action"], {"obs": x})[0]
            worst_t = max(worst_t, float(np.max(np.abs(t_ref - t_out))))
            worst_n = max(worst_n, float(np.max(np.abs(n_ref - n_out))))
            # 两个模型必须描述同一个策略：target = default + clip(a)*scale
            expect = np.clip(n_out, -1, 1)
            expect[:, :12] = np.clip(
                DEFAULT_LEG_POS + expect[:, :12] * ACTION_SCALE_LEG, JOINT_POS_LOW, JOINT_POS_HIGH
            )
            expect[:, 12:] *= WHEEL_VEL_SCALE
            worst_consistency = max(worst_consistency, float(np.max(np.abs(expect - t_out))))
        print(f"[check] targets vs torch = {worst_t:.2e} | norm vs torch = {worst_n:.2e} | "
              f"两版一致性 = {worst_consistency:.2e}")
        assert max(worst_t, worst_n, worst_consistency) < 1e-4, "ONNX 数值对不上！"
    except ImportError:
        print("[check] 没装 onnxruntime，跳过数值校验（部署前务必在 ms 环境里跑一次）")

    # ---------------- 部署信息 ----------------
    meta = {
        "source": str(ckpt.relative_to(RL_DIR)).replace("\\", "/"),
        "onnx": out.name,
        "sha256_16": sha256(out),
        "input": {"name": "obs", "shape": [1, 53], "dtype": "float32"},
        "output": {
            "name": "target",
            "shape": [1, 16],
            "dtype": "float32",
            "format": "targets",
            "layout": "0:12 = 12 个腿部目标角 (rad, 绝对角)；12:16 = 4 个轮子目标角速度 (rad/s)",
            "note": "已把 action_scale / 默认站姿 / 关节限位烘进网络，输出可直接下发",
        },
        "norm_onnx": out_norm.name,
        "trained_steps": int(model.num_timesteps),
        "net_arch": [256, 128, 64],
    }
    (out.parent / f"{out.name}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # 顺手更新 deploy.yaml 里的 onnx 字段
    dy = RL_DIR / "deploy.yaml"
    if dy.exists():
        contract = yaml.safe_load(dy.read_text(encoding="utf-8"))
        contract.setdefault("policy", {})["onnx"] = f"models/{out.name}"
        contract["policy"]["onnx_norm"] = f"models/{out_norm.name}"
        contract["policy"]["output_format"] = "targets"
        contract["policy"]["sha256_16"] = meta["sha256_16"]
        contract["policy"]["trained_steps"] = meta["trained_steps"]
        contract["stand_regulator"] = dict(STAND_REG)
        dy.write_text(yaml.safe_dump(contract, allow_unicode=True, sort_keys=False), encoding="utf-8")
        print(f"[deploy] 已更新 {dy}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
