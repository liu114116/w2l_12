"""并行环境工厂（单独一个模块，**不 import torch**）。

SB3 在 Windows 上用 spawn 起子进程，子进程会重新 import 定义 ``make_env`` 的模块。
把工厂放在这里，子进程只会加载 mujoco / numpy / gymnasium，
不会每个进程再吃一份 torch（16 进程能省 5 GB 内存）。
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from typing import Any

import gymnasium as gym
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor

from w2l_env import EnvCfg, W2LEnv


def make_env(rank: int, cfg_dict: dict[str, Any], seed: int = 0) -> gym.Env:
    """返回一个 thunk，供 SubprocVecEnv / DummyVecEnv 调用。

    注意：这里**不**包 gymnasium 的 Monitor —— 外层用 SB3 的 ``VecMonitor``
    统一统计 episode（否则两层 Monitor 会互相覆盖，还会多一层 Python 开销）。
    """

    def _thunk() -> gym.Env:
        cfg = EnvCfg(**cfg_dict)
        env = W2LEnv(cfg)
        env.reset(seed=seed + rank)
        return env

    _thunk.rank = rank  # type: ignore[attr-defined]
    return _thunk


def build_vec_env(
    num_envs: int,
    cfg: EnvCfg,
    seed: int = 0,
    use_subproc: bool = True,
) -> VecMonitor:
    cfg_dict = cfg.to_dict()
    thunks = [make_env(i, cfg_dict, seed) for i in range(num_envs)]
    venv = SubprocVecEnv(thunks) if (use_subproc and num_envs > 1) else DummyVecEnv(thunks)
    keywords = tuple(f"rew_{k}" for k in cfg.rewards)
    try:
        return VecMonitor(venv, info_keywords=keywords)
    except TypeError:            # 老版本 SB3 没有 info_keywords
        return VecMonitor(venv)
