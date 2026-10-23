"""把训练好的 SB3 策略导出成 ONNX，并做数值对齐校验。

    python src/rl/export_onnx.py --tag v1
    python src/rl/export_onnx.py --checkpoint src/rl/checkpoints/v1/model_5000000_steps.zip

网络输出**直接就是物理量**：12 个腿部目标角 (rad) + 4 个轮子目标角速度 (rad/s)，
部署端限幅后可以直接写进 ctrl / 下位机，不需要任何换算。

产物：
    src/rl/models/policy.onnx       ONNX 权重
    src/rl/models/policy.onnx.json  输入输出说明 + 哈希
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
    ACTION_HIGH,
    ACTION_LOW,
    ACTION_SCALE,
    DEFAULT_LEG_POS,
    JOINT_POS_HIGH,
    JOINT_POS_LOW,
    WHEEL_VEL_SCALE,
)

RL_DIR = _HERE
MODEL_DIR = RL_DIR / "models"

#: 绝对目标的上下界（腿 clip 到关节软限位，轮 clip 到 ±25 rad/s）
_TARGET_LOW = np.concatenate([JOINT_POS_LOW, -WHEEL_VEL_SCALE * np.ones(4)])
_TARGET_HIGH = np.concatenate([JOINT_POS_HIGH, WHEEL_VEL_SCALE * np.ones(4)])


class PolicyNet(torch.nn.Module):
    """把训练内部的归一化动作**烘成绝对目标**：obs(53) -> 16 个物理量。

    ``out[0:12] = clip(default_leg_pos + a[0:12]*action_scale, 关节软限位)``
    ``out[12:16] = a[12:16] * 25``

    这样部署端拿到的就是 12 个腿部目标角 (rad) + 4 个轮子目标角速度 (rad/s)，
    不需要任何换算。
    """

    def __init__(self, policy):
        super().__init__()
        self.policy = policy
        self.register_buffer("default_pos", torch.tensor(DEFAULT_LEG_POS, dtype=torch.float32))
        self.register_buffer("leg_scale", torch.tensor(ACTION_SCALE[:12], dtype=torch.float32))
        self.register_buffer("low", torch.tensor(_TARGET_LOW, dtype=torch.float32))
        self.register_buffer("high", torch.tensor(_TARGET_HIGH, dtype=torch.float32))
        self.wheel_scale = float(WHEEL_VEL_SCALE)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        a = self.policy.extract_features(obs, self.policy.pi_features_extractor)
        a = self.policy.mlp_extractor.forward_actor(a)
        a = self.policy.action_net(a)
        a = torch.clamp(a, -1.0, 1.0)
        q = self.default_pos + a[:, :12] * self.leg_scale
        w = a[:, 12:] * self.wheel_scale
        return torch.clamp(torch.cat([q, w], dim=-1), self.low, self.high)


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
    net = PolicyNet(policy).eval()

    dummy = torch.zeros(1, 53, dtype=torch.float32)
    torch.onnx.export(
        net, dummy, str(out),
        input_names=["obs"], output_names=["target"],
        dynamic_axes={"obs": {0: "batch"}, "target": {0: "batch"}},
        opset_version=args.opset, do_constant_folding=True,
    )
    print(f"[onnx] {out}  ({out.stat().st_size/1024:.1f} KB)  sha={sha256(out)}")

    # ---------------- 数值校验 + 输出范围自检 ----------------
    try:
        import onnxruntime as ort

        sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
        rng = np.random.default_rng(0)
        worst = 0.0
        for scale in (0.1, 1.0, 5.0):
            x = (rng.standard_normal((64, 53)) * scale).astype(np.float32)
            with torch.no_grad():
                t = net(torch.from_numpy(x)).numpy()
            o = sess.run(["target"], {"obs": x})[0]
            worst = max(worst, float(np.max(np.abs(t - o))))
        print(f"[check] torch vs onnxruntime 最大偏差 = {worst:.2e}")
        assert worst < 1e-4, "ONNX 数值对不上！"

        x = rng.standard_normal((256, 53)).astype(np.float32)
        y = sess.run(["target"], {"obs": x})[0]
        print(f"[check] 输出范围  腿 {y[:, :12].min():+.2f}~{y[:, :12].max():+.2f} rad   "
              f"轮 {y[:, 12:].min():+.1f}~{y[:, 12:].max():+.1f} rad/s")
        assert y[:, :12].min() >= ACTION_LOW[:12].min() and y[:, :12].max() <= ACTION_HIGH[:12].max()
        assert np.abs(y[:, 12:]).max() <= ACTION_HIGH[12:].max() * 1.05
    except ImportError:
        print("[check] 没装 onnxruntime，跳过数值校验")

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
            "format": "absolute",
            "layout": "0:12 = 12 个腿部目标角 (rad, 绝对角)；12:16 = 4 个轮子目标角速度 (rad/s)",
            "limits": {"low": ACTION_LOW.tolist(), "high": ACTION_HIGH.tolist()},
        },
        "trained_steps": int(model.num_timesteps),
        "net_arch": [256, 128, 64],
    }
    (out.parent / f"{out.name}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    dy = RL_DIR / "deploy.yaml"
    if dy.exists():
        contract = yaml.safe_load(dy.read_text(encoding="utf-8"))
        contract.setdefault("policy", {})["onnx"] = f"models/{out.name}"
        contract["policy"]["sha256_16"] = meta["sha256_16"]
        contract["policy"]["trained_steps"] = meta["trained_steps"]
        contract["policy"]["output_format"] = "absolute"
        contract.pop("stand_regulator", None)
        dy.write_text(yaml.safe_dump(contract, allow_unicode=True, sort_keys=False), encoding="utf-8")
        print(f"[deploy] 已更新 {dy}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
