"""w2l_12 轮腿机器人 —— PPO 训练入口。

    python src/rl/train_ppo.py --tag v1 --num-envs 16 --steps 30e6

训练完会在 ``src/rl/`` 下留下：

* ``checkpoints/<tag>/model.zip``        最终策略（SB3 格式）
* ``checkpoints/<tag>/model_<step>.zip`` 过程 checkpoint
* ``logs/<tag>/``                        TensorBoard
* ``deploy.yaml``                        部署契约（关节顺序 / 观测 / 动作 / PD 增益）

Phi 备注：轮腿的"稳妥"配方来自 legged_gym / rsl_rl 那套 PPO + 指令课程 + 域随机化，
本项目把它搬到了 MuJoCo + Gymnasium + stable-baselines3 上，Windows 单卡也能训。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from robot_cfg import deploy_contract  # noqa: E402
from vec_env import build_vec_env  # noqa: E402
from w2l_env import EnvCfg, W2LEnv  # noqa: E402

RL_DIR = _HERE
CKPT_DIR = RL_DIR / "checkpoints"
LOG_DIR = RL_DIR / "logs"

#: 评估用的固定指令（都是"以后 SLAM / 遥控器会发的那种"）
EVAL_COMMANDS = (
    ("stand", np.array([0.0, 0.0, 0.0])),
    ("forward", np.array([0.30, 0.0, 0.0])),
    ("turn", np.array([0.0, 0.0, 0.60])),
    ("combo", np.array([0.25, 0.15, 0.30])),
)


class TrackingEvalCallback(BaseCallback):
    """定期用固定指令跑一遍确定性策略，把"到底会不会走"记到 TensorBoard。"""

    def __init__(self, cfg: EnvCfg, eval_every: int = 500_000, eval_steps: int = 250, verbose: int = 0):
        super().__init__(verbose)
        self.cfg = cfg
        self.eval_every = eval_every
        self.eval_steps = eval_steps
        self.next_eval = eval_every
        self._env: W2LEnv | None = None

    def _get_env(self) -> W2LEnv:
        if self._env is None:
            ecfg = replace(
                self.cfg,
                obs_noise=0.0,
                randomize_friction=False,
                randomize_mass=False,
                push_interval_steps=(10**9, 10**9),
                latency_min=0,
                latency_max=0,
                cmd_resample_steps=10**9,
                zero_cmd_prob=0.0,
            )
            self._env = W2LEnv(ecfg)
        return self._env

    def _on_step(self) -> bool:
        if self.num_timesteps < self.next_eval:
            return True
        self.next_eval += self.eval_every
        env = self._get_env()
        surv_all = []
        msg = []
        for name, cmd in EVAL_COMMANDS:
            env.forced_cmd = cmd
            obs, _ = env.reset(seed=1234)
            vx_err = vy_err = wz_err = 0.0
            n = 0
            for _ in range(self.eval_steps):
                action, _ = self.model.predict(obs, deterministic=True)
                obs, _, term, trunc, _ = env.step(action)
                v = env._policy_frame_vel()
                vx_err += abs(v[0] - cmd[0])
                vy_err += abs(v[1] - cmd[1])
                wz_err += abs(env.data.qvel[5] - cmd[2])
                n += 1
                if term or trunc:
                    break
            vx_err /= n
            vy_err /= n
            wz_err /= n
            surv = n / self.eval_steps
            surv_all.append(surv)
            self.logger.record(f"eval/{name}_vx_err", vx_err)
            self.logger.record(f"eval/{name}_vy_err", vy_err)
            self.logger.record(f"eval/{name}_wz_err", wz_err)
            msg.append(f"{name}: |vx|={vx_err:.3f} |wz|={wz_err:.3f} s={surv*100:.0f}%")
        surv_mean = float(np.mean(surv_all))
        self.logger.record("eval/survival", surv_mean)
        if self.verbose:
            print(f"[eval @ {self.num_timesteps:>10,}] " + " | ".join(msg), flush=True)
        return True


def build_cfg(args: argparse.Namespace) -> EnvCfg:
    cfg = EnvCfg()
    if args.torque:
        from w2l_env import ASSET_TORQUE

        cfg.xml = str(ASSET_TORQUE)
        cfg.action_mode = "torque"
    cfg.cmd_lin_x = args.cmd_vx
    cfg.cmd_lin_y = args.cmd_vy
    cfg.cmd_ang_z = args.cmd_wz
    cfg.seed = args.seed
    cfg.zero_cmd_prob = args.zero_cmd_prob
    if args.slope_deg > 0:
        cfg.slope_max_deg = args.slope_deg
    if args.rew:
        for item in args.rew.split(","):
            k, v = item.split("=")
            cfg.rewards[k.strip()] = float(v)
        print(f"[cfg] 奖励覆盖: {args.rew}")
    return cfg


def main() -> int:
    p = argparse.ArgumentParser(description="w2l_12 轮腿 PPO 训练")
    p.add_argument("--tag", default="v1", help="实验名，决定 checkpoints/logs 子目录")
    p.add_argument("--num-envs", type=int, default=16)
    p.add_argument("--steps", type=float, default=30e6, help="总环境步数")
    p.add_argument("--n-steps", type=int, default=256, help="每个环境每次 rollout 的步数")
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--ent-coef", type=float, default=0.003)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu", help="PPO 网络跑 cpu 通常比 cuda 快（批量太小）")
    p.add_argument("--cmd-vx", type=float, default=0.6)
    p.add_argument("--cmd-vy", type=float, default=0.4)
    p.add_argument("--cmd-wz", type=float, default=1.0)
    p.add_argument("--slope-deg", type=float, default=0.0)
    p.add_argument("--rew", default=None, help="临时覆盖奖励权重，如 'track_ang_vel=3.0,alive=0.5'")
    p.add_argument("--zero-cmd-prob", type=float, default=0.15, help="抽到\"站住\"指令的概率")
    p.add_argument("--torque", action="store_true", help="直接用纯力矩模式（策略输出力矩）")
    p.add_argument("--resume", default=None, help="从某个 model.zip 继续训练")
    p.add_argument("--eval-every", type=int, default=500_000)
    p.add_argument("--ckpt-every", type=int, default=1_000_000)
    p.add_argument("--num-threads", type=int, default=1, help="每个 torch 进程线程数")
    args = p.parse_args()

    torch.set_num_threads(args.num_threads)
    cfg = build_cfg(args)
    run_dir = CKPT_DIR / args.tag
    run_dir.mkdir(parents=True, exist_ok=True)

    total = int(args.steps)
    print("=" * 78)
    print(f"  w2l_12 PPO  tag={args.tag}  envs={args.num_envs}  steps={total:,}  "
          f"control={1/cfg.control_dt:.0f}Hz  obs={53} act={16}")
    print(f"  xml={cfg.xml}")
    print("=" * 78, flush=True)

    venv = build_vec_env(args.num_envs, cfg, seed=args.seed)
    venv.seed(args.seed)

    t0 = time.time()
    if args.resume:
        resume = Path(args.resume)
        if not resume.is_absolute():
            resume = (RL_DIR / resume).resolve() if not resume.exists() else resume.resolve()
        model = PPO.load(str(resume), env=venv, device=args.device, print_system_info=True)
        print(f"[resume] {resume}  ({model.num_timesteps:,} steps)")
    else:
        model = PPO(
            "MlpPolicy",
            venv,
            learning_rate=args.lr,
            n_steps=args.n_steps,
            batch_size=args.batch,
            n_epochs=args.epochs,
            gamma=args.gamma,
            gae_lambda=0.95,
            clip_range=0.2,
            clip_range_vf=None,
            ent_coef=args.ent_coef,
            vf_coef=0.5,
            max_grad_norm=1.0,
            use_sde=False,
            target_kl=0.03,
            policy_kwargs=dict(
                net_arch=dict(pi=[256, 128, 64], vf=[256, 128, 64]),
                activation_fn=torch.nn.ELU,
                log_std_init=-0.5,
                ortho_init=True,
            ),
            tensorboard_log=str(LOG_DIR / args.tag),
            device=args.device,
            verbose=1,
            seed=args.seed,
        )

    callbacks = [
        CheckpointCallback(
            save_freq=max(1, args.ckpt_every // args.num_envs),
            save_path=str(run_dir),
            name_prefix="model",
            save_vecnormalize=False,
            verbose=0,
        ),
        TrackingEvalCallback(cfg, eval_every=args.eval_every, verbose=1),
    ]

    try:
        model.learn(total_timesteps=total, callback=callbacks, progress_bar=False,
                    reset_num_timesteps=not bool(args.resume))
    except KeyboardInterrupt:
        print("\n[interrupt] 保存当前策略 ...")

    model.save(str(run_dir / "model"))

    # ---------------- 部署契约 ---------------- #
    contract = deploy_contract(control_freq_hz=1.0 / cfg.control_dt, filter_hz=cfg.action_filter_hz)
    contract["trained_steps"] = int(model.num_timesteps)
    contract["tag"] = args.tag
    contract["action_mode"] = cfg.action_mode
    contract["policy"] = {"sb3_zip": f"checkpoints/{args.tag}/model.zip", "onnx": "models/policy.onnx"}
    contract["env_cfg"] = json.loads(json.dumps(cfg.to_dict(), default=str))
    with open(RL_DIR / "deploy.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(contract, f, allow_unicode=True, sort_keys=False)

    venv.close()
    print(f"\n[done] {(time.time()-t0)/60:.1f} min, model -> {run_dir / 'model.zip'}")
    print(f"[done] contract -> {RL_DIR / 'deploy.yaml'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
