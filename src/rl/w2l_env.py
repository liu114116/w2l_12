"""w2l_12 轮腿机器人 —— 强化学习训练环境（MuJoCo + Gymnasium）。

设计要点（对齐 legged_gym / RC_WheelLeg 的成熟做法）：

* **动作 16 维**：12 个腿部目标角 + 4 个轮子目标角速度。
  腿部在仿真里是 position 伺服（kp/kv），真机上就是"力矩环里跑 PD"，
  数学上等价：``tau = kp*(q_des - q) - kv*qd``，限幅 ±0.98 N·m。
  轮子是 velocity 伺服，限幅 ±0.29 N·m。
  想要策略直接出力矩，把 ``cfg.action_mode`` 设成 ``"torque"``，配合
  ``prepare_assets.py --torque`` 生成的模型即可。
* **观测 53 维**：与部署端共享同一份代码（robot_cfg / PolicyRuntime），
  训练-部署不会对不上。
* **速度指令跟踪**：``vx, vy, wz`` 三个量就是以后 SLAM / 遥控器的接口。
* **域随机化**：摩擦、质量、质心、伺服增益、动作延迟、观测噪声、外力推扰。
* **课程**：指令范围随训练步数放大，先学会站住/慢走，再学快走和转向。

坐标约定：模型里机器人**前方是 -x**（FL/FR 在 -x 侧），
所以 "前向速度" = ``-v_x_body``，观测和奖励都在"前向=+x"的指令系里做。
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:            # 让 scripts/ 里的脚本也能 import robot_cfg
    sys.path.insert(0, str(_HERE))

from robot_cfg import (  # noqa: E402
    ACTION_SCALE_LEG,
    ACT_DIM,
    build_obs,
    DEFAULT_LEG_POS,
    JOINT_POS_HIGH,
    JOINT_POS_LOW,
    LEG_JOINTS,
    OBS_CLIP,
    OBS_DIM,
    STAND_HEIGHT,
    WHEEL_JOINTS,
    WHEEL_VEL_SCALE,
)

ASSET = Path(__file__).resolve().parent / "assets" / "w2l_12_train.xml"
ASSET_TORQUE = Path(__file__).resolve().parent / "assets" / "w2l_12_train_torque.xml"

#: 奖励权重（名字 -> 权重），改这里就能调；训练日志里会打印
DEFAULT_REWARDS = {
    "track_lin_vel": 2.0,
    "track_ang_vel": 2.8,
    "lin_vel_z": -1.0,
    "ang_vel_xy": -0.05,
    "orientation": -2.5,
    "base_height": -6.0,
    "joint_acc": -2.5e-7,
    "action_rate": -0.02,
    "torques": -0.02,
    "wheel_acc": -2.5e-7,
    "collision": -1.5,
    "dof_limits": -2.0,
    "stand_still": -0.6,
    "stand_vel": -3.0,
    "alive": 0.6,
}


@dataclass
class EnvCfg:
    """环境配置。所有可调参数都在这里，训练脚本通过 --set 覆盖。"""

    xml: str = str(ASSET)
    action_mode: str = "servo"        # "servo" | "torque"
    sim_dt: float = 0.002
    decimation: int = 10              # 控制频率 = 1/(dt*decimation) = 50 Hz
    episode_steps: int = 500          # 10 s

    # --- 指令课程 ---
    cmd_lin_x: float = 0.6            # 最大前向速度 m/s
    cmd_lin_y: float = 0.4            # 最大横移 m/s
    cmd_ang_z: float = 1.0            # 最大偏航角速度 rad/s
    curriculum_start: float = 0.35    # 训练开始时指令范围的比例
    curriculum_steps: float = 30e6    # 用多少环境步线性放大到 100%
    cmd_resample_steps: int = 250     # 5 s 重采样一次指令
    zero_cmd_prob: float = 0.15       # 15% 概率抽到"站住"指令

    # --- 动作 / 观测 ---
    action_filter_hz: float = 10.0
    latency_min: int = 0
    latency_max: int = 2
    obs_noise: float = 1.0            # 0 关掉，1 用下面这组标准差
    obs_noise_dof_pos: float = 0.01
    obs_noise_dof_vel: float = 0.5
    obs_noise_gyro: float = 0.05
    obs_noise_grav: float = 0.02
    obs_noise_wheel: float = 0.5

    # --- 域随机化 ---
    randomize_friction: bool = True
    friction_range: tuple = (0.6, 1.4)
    randomize_mass: bool = True
    base_mass_range: tuple = (0.9, 1.15)      # 相对倍数
    com_offset: float = 0.008                 # m
    kp_scale_range: tuple = (0.85, 1.15)
    kd_scale_range: tuple = (0.85, 1.15)
    push_interval_steps: int = (150, 350)
    push_vel: float = 0.5                     # m/s 等效冲量
    slope_max_deg: float = 0.0                # 地面倾角随机化，0 = 纯平地

    # --- 终止 ---
    terminate_height: float = 0.055
    terminate_tilt: float = 0.65              # rad
    terminate_base_contact: bool = True

    # --- 奖励 ---
    rewards: dict = field(default_factory=lambda: dict(DEFAULT_REWARDS))
    target_height: float = STAND_HEIGHT

    seed: int = 0

    @property
    def control_dt(self) -> float:
        return self.sim_dt * self.decimation

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["rewards"] = dict(self.rewards)
        return d


class W2LEnv(gym.Env):
    """单机器人环境（并行由 SB3 的 SubprocVecEnv 负责）。"""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, cfg: EnvCfg | None = None, render_mode: str | None = None):
        super().__init__()
        self.cfg = cfg or EnvCfg()
        self.render_mode = render_mode

        self.model = mujoco.MjModel.from_xml_path(str(self.cfg.xml))
        self.model.opt.timestep = self.cfg.sim_dt
        self.data = mujoco.MjData(self.model)

        # ---------------- 索引 ---------------- #
        m = self.model
        self.base_body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base")
        self.floor_geom = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        self.base_col_geom = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "base_collision")
        self.leg_qpos_adr = np.array(
            [m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in LEG_JOINTS]
        )
        self.leg_dof_adr = np.array(
            [m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in LEG_JOINTS]
        )
        self.wheel_qpos_adr = np.array(
            [m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in WHEEL_JOINTS]
        )
        self.wheel_dof_adr = np.array(
            [m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in WHEEL_JOINTS]
        )
        self.act_leg = np.array(
            [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in LEG_JOINTS]
        )
        self.act_wheel = np.array(
            [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in WHEEL_JOINTS]
        )
        self.wheel_geoms = [
            mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"wheel_{leg}_collision")
            for leg in ("FL", "FR", "RL", "RR")
        ]
        self.wheel_geoms = [g for g in self.wheel_geoms if g >= 0]

        # 原始参数（域随机化要按比例还原）
        self._nominal = {
            "base_mass": float(m.body_mass[self.base_body]),
            "base_ipos": m.body_ipos[self.base_body].copy(),
            "floor_friction": m.geom_friction[self.floor_geom].copy(),
            "wheel_friction": m.geom_friction[self.wheel_geoms].copy() if self.wheel_geoms else None,
            "act_gain": m.actuator_gainprm.copy(),
            "act_bias": m.actuator_biasprm.copy(),
        }

        # ---------------- 空间 ---------------- #
        self.action_space = spaces.Box(-1.0, 1.0, (ACT_DIM,), np.float32)
        self.observation_space = spaces.Box(-OBS_CLIP, OBS_CLIP, (OBS_DIM,), np.float32)

        # ---------------- 状态缓冲 ---------------- #
        self.cmd = np.zeros(3)
        self.last_action = np.zeros(ACT_DIM)
        self.filtered_action = np.zeros(ACT_DIM)
        self.prev_dof_vel = np.zeros(ACT_DIM)
        self.prev_wheel_vel = np.zeros(4)
        self.obs = np.zeros(OBS_DIM, np.float32)
        self.ctrl = np.zeros(ACT_DIM)
        self.steps = 0
        self.total_steps = 0          # 用于课程
        self._latency = 0
        self._action_queue: list[np.ndarray] = []
        self._push_countdown = 10**9
        self._cmd_countdown = 0
        self._alpha_filter = 1.0
        self.forced_cmd: np.ndarray | None = None   # 评估时固定指令用

        self._setup_episode_constants()
        self._renderer = None

    # ------------------------------------------------------------------ 工具
    def _setup_episode_constants(self) -> None:
        self._alpha_filter = 1.0 - math.exp(-2.0 * math.pi * self.cfg.action_filter_hz * self.cfg.control_dt)
        self._cmd_scale = min(1.0, self.cfg.curriculum_start + (1.0 - self.cfg.curriculum_start) * self.total_steps / max(1.0, self.cfg.curriculum_steps))

    def _gravity_body(self) -> np.ndarray:
        quat = self.data.qpos[3:7]
        rot = np.zeros(9)
        mujoco.mju_quat2Mat(rot, quat)
        return rot.reshape(3, 3).T @ np.array([0.0, 0.0, -1.0])

    def _base_lin_vel_body(self) -> np.ndarray:
        quat = self.data.qpos[3:7]
        rot = np.zeros(9)
        mujoco.mju_quat2Mat(rot, quat)
        return rot.reshape(3, 3).T @ self.data.qvel[0:3]

    def _policy_frame_vel(self) -> np.ndarray:
        """把机身速度转到"前方=+x"的指令系（模型前方是 -x）。"""
        v = self._base_lin_vel_body()
        return np.array([-v[0], -v[1], v[2]])

    # ------------------------------------------------------------------ 域随机化
    def _randomize(self, rng: np.random.Generator) -> None:
        m, cfg = self.model, self.cfg

        # 质量 / 质心
        if cfg.randomize_mass:
            m.body_mass[self.base_body] = self._nominal["base_mass"] * rng.uniform(*cfg.base_mass_range)
            off = rng.uniform(-cfg.com_offset, cfg.com_offset, 3)
            m.body_ipos[self.base_body] = self._nominal["base_ipos"] + off

        # 摩擦
        if cfg.randomize_friction:
            f = rng.uniform(*cfg.friction_range)
            m.geom_friction[self.floor_geom, 0] = self._nominal["floor_friction"][0] * f
            if self._nominal["wheel_friction"] is not None:
                m.geom_friction[self.wheel_geoms, 0] = self._nominal["wheel_friction"][:, 0] * f

        # 伺服增益（等价于真机上 Kp/Kd 有误差、电池电压变化）
        gs = rng.uniform(*cfg.kp_scale_range)
        ds = rng.uniform(*cfg.kd_scale_range)
        if cfg.action_mode == "servo":
            m.actuator_gainprm[self.act_leg, 0] = self._nominal["act_gain"][self.act_leg, 0] * gs
            m.actuator_biasprm[self.act_leg, 1] = self._nominal["act_bias"][self.act_leg, 1] * gs
            m.actuator_biasprm[self.act_leg, 2] = self._nominal["act_bias"][self.act_leg, 2] * ds
            m.actuator_gainprm[self.act_wheel, 0] = self._nominal["act_gain"][self.act_wheel, 0] * ds
            m.actuator_biasprm[self.act_wheel, 2] = self._nominal["act_bias"][self.act_wheel, 2] * ds

        # 动作延迟
        self._latency = int(rng.integers(self.cfg.latency_min, self.cfg.latency_max + 1))
        self._action_queue = [np.zeros(ACT_DIM) for _ in range(self._latency)]

        # 外力推扰
        lo, hi = self.cfg.push_interval_steps
        self._push_countdown = int(rng.integers(lo, hi)) if hi > lo else 10**9

    def _push(self, rng: np.random.Generator) -> None:
        direction = rng.normal(size=2)
        direction /= max(1e-6, np.linalg.norm(direction))
        impulse = direction * self.cfg.push_vel * self.model.body_mass[self.base_body]
        self.data.qvel[0:2] += impulse / self.model.body_mass[self.base_body]

    # ------------------------------------------------------------------ 指令
    def _sample_command(self, rng: np.random.Generator) -> np.ndarray:
        if self.forced_cmd is not None:
            return np.asarray(self.forced_cmd, dtype=np.float64).copy()
        cfg = self.cfg
        if rng.random() < cfg.zero_cmd_prob:
            return np.zeros(3)
        s = self._cmd_scale
        vx = rng.uniform(-cfg.cmd_lin_x, cfg.cmd_lin_x) * s
        vy = rng.uniform(-cfg.cmd_lin_y, cfg.cmd_lin_y) * s
        wz = rng.uniform(-cfg.cmd_ang_z, cfg.cmd_ang_z) * s
        # 抽到的速度太小时容易退化成"原地不动"，这里给个下限
        if abs(vx) < 0.1 and abs(vy) < 0.05 and abs(wz) < 0.2:
            vx = math.copysign(0.15 * s, vx if vx != 0 else 1.0)
        return np.array([vx, vy, wz])

    # ------------------------------------------------------------------ Gym API
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        rng = self.np_random

        self._randomize(rng)
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)  # 站姿

        d = self.data
        # 关节角小扰动
        d.qpos[self.leg_qpos_adr] = DEFAULT_LEG_POS + rng.uniform(-0.08, 0.08, 12)
        d.qpos[self.wheel_qpos_adr] = 0.0
        # 机身位姿
        d.qpos[0:2] = rng.uniform(-0.02, 0.02, 2)
        d.qpos[2] = STAND_HEIGHT + rng.uniform(-0.005, 0.02)
        roll, pitch = rng.uniform(-0.05, 0.05, 2)
        d.qpos[3:7] = self._quat_from_rpy(roll, pitch, rng.uniform(-math.pi, math.pi))
        d.qvel[:] = 0.0
        d.qvel[0:3] = rng.uniform(-0.05, 0.05, 3)
        d.qvel[3:6] = rng.uniform(-0.05, 0.05, 3)

        if self.cfg.slope_max_deg > 0:
            ang = math.radians(self.cfg.slope_max_deg)
            q = np.zeros(4)
            axis = np.array([rng.uniform(-1, 1), rng.uniform(-1, 1), 0.0])
            axis /= max(1e-6, np.linalg.norm(axis))
            mujoco.mju_axisAngle2Quat(q, axis, rng.uniform(-ang, ang))
            self.model.geom_quat[self.floor_geom] = q
        mujoco.mj_forward(self.model, d)

        self.cmd = self._sample_command(rng)
        self.last_action[:] = 0.0
        self.filtered_action[:] = 0.0
        self.prev_dof_vel[:] = 0.0
        self.prev_wheel_vel[:] = 0.0
        self.steps = 0
        self._cmd_countdown = self.cfg.cmd_resample_steps
        self._setup_episode_constants()
        # 落地稳定：先把站姿摆稳再开始计时（不算奖励）
        self.ctrl[:] = self._action_to_ctrl(np.zeros(ACT_DIM))
        self._apply_ctrl(self.ctrl)
        mujoco.mj_step(self.model, d, nstep=40)
        self.ctrl[:] = self._action_to_ctrl(np.zeros(ACT_DIM))
        self._apply_ctrl(self.ctrl)

        self._build_obs()
        return self.obs.copy(), {}

    def step(self, action: np.ndarray):
        self.steps += 1
        self.total_steps += 1

        # 动作低通（真机上舵机目标也要这么做），再走延迟队列
        a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        self.filtered_action += self._alpha_filter * (a - self.filtered_action)
        if self._latency > 0:
            self._action_queue.append(self.filtered_action.copy())
            a_exec = self._action_queue.pop(0)
        else:
            a_exec = self.filtered_action

        self.ctrl[:] = self._action_to_ctrl(a_exec)
        self._apply_ctrl(self.ctrl)
        mujoco.mj_step(self.model, self.data, nstep=self.cfg.decimation)

        d = self.data
        terminated = self._check_termination()
        reward, terms = self._compute_reward(a, terminated)
        truncated = self.steps >= self.cfg.episode_steps

        # 指令重采样
        self._cmd_countdown -= 1
        if self._cmd_countdown <= 0:
            self.cmd = self._sample_command(self.np_random)
            self._cmd_countdown = self.cfg.cmd_resample_steps
            self._cmd_scale = min(1.0, self.cfg.curriculum_start + (1.0 - self.cfg.curriculum_start) * self.total_steps / max(1.0, self.cfg.curriculum_steps))

        # 推扰
        self._push_countdown -= 1
        if self._push_countdown <= 0:
            self._push(self.np_random)
            lo, hi = self.cfg.push_interval_steps
            self._push_countdown = int(self.np_random.integers(lo, hi)) if hi > lo else 10**9

        self.prev_dof_vel[:12] = d.qvel[self.leg_dof_adr]
        self.prev_dof_vel[12:] = d.qvel[self.wheel_dof_adr]
        self.prev_wheel_vel[:] = d.qvel[self.wheel_dof_adr]
        self.last_action[:] = a

        self._build_obs()
        info = {"command": self.cmd.copy(), **{f"rew_{k}": v for k, v in terms.items()}}
        return self.obs.copy(), float(reward), bool(terminated), bool(truncated), info

    # ------------------------------------------------------------------ 内部
    def _action_to_ctrl(self, action: np.ndarray) -> np.ndarray:
        if self.cfg.action_mode == "torque":
            ctrl = np.empty(ACT_DIM)
            ctrl[:12] = action[:12] * 0.98
            ctrl[12:] = action[12:] * 0.29
            return ctrl
        q_target = DEFAULT_LEG_POS + action[:12] * ACTION_SCALE_LEG
        q_target = np.clip(q_target, JOINT_POS_LOW, JOINT_POS_HIGH)
        w_target = action[12:] * WHEEL_VEL_SCALE
        return np.concatenate([q_target, w_target])

    def _apply_ctrl(self, ctrl: np.ndarray) -> None:
        self.data.ctrl[self.act_leg] = ctrl[:12]
        self.data.ctrl[self.act_wheel] = ctrl[12:]

    @staticmethod
    def _quat_from_rpy(roll: float, pitch: float, yaw: float) -> np.ndarray:
        q = np.zeros(4)
        mujoco.mju_euler2Quat(q, np.array([roll, pitch, yaw]), "xyz")
        return q

    def _build_obs(self) -> None:
        """加传感器噪声后交给 robot_cfg.build_obs 统一编码（与部署端共用）。"""
        cfg, d = self.cfg, self.data
        gyro = d.qvel[3:6]
        grav = self._gravity_body()
        q_leg = d.qpos[self.leg_qpos_adr]
        qd_leg = d.qvel[self.leg_dof_adr]
        wd = d.qvel[self.wheel_dof_adr]

        n = cfg.obs_noise
        rng = self.np_random
        if n > 0:
            gyro = gyro + rng.normal(0, cfg.obs_noise_gyro, 3) * n
            grav = grav + rng.normal(0, cfg.obs_noise_grav, 3) * n
            q_leg = q_leg + rng.normal(0, cfg.obs_noise_dof_pos, 12) * n
            qd_leg = qd_leg + rng.normal(0, cfg.obs_noise_dof_vel, 12) * n
            wd = wd + rng.normal(0, cfg.obs_noise_wheel, 4) * n

        self.obs[:] = build_obs(gyro, grav, self.cmd, q_leg, qd_leg, wd, self.last_action)

    def _check_termination(self) -> bool:
        d, cfg = self.data, self.cfg
        if d.qpos[2] < cfg.terminate_height:
            return True
        grav = self._gravity_body()
        if math.hypot(grav[0], grav[1]) > math.sin(cfg.terminate_tilt):
            return True
        if cfg.terminate_base_contact:
            for i in range(d.ncon):
                c = d.contact[i]
                g1, g2 = c.geom1, c.geom2
                if (g1 == self.base_col_geom and g2 == self.floor_geom) or (
                    g2 == self.base_col_geom and g1 == self.floor_geom
                ):
                    return True
        return False

    def _compute_reward(self, action: np.ndarray, terminated: bool) -> tuple[float, dict]:
        d, cfg = self.data, self.cfg
        rw = cfg.rewards
        terms: dict[str, float] = {}

        v = self._policy_frame_vel()
        wz = d.qvel[5]
        cmd = self.cmd

        # 约定：terms 里放"量"（惩罚项是正值大小），符号由 rewards 权重给
        # ---- 跟踪 ----
        terms["track_lin_vel"] = math.exp(-float(np.sum((v[:2] - cmd[:2]) ** 2)) / 0.16)
        terms["track_ang_vel"] = math.exp(-((wz - cmd[2]) ** 2) / 0.16)

        # ---- 平滑 / 姿态 ----
        terms["lin_vel_z"] = float(v[2] ** 2)
        terms["ang_vel_xy"] = float(np.sum(d.qvel[3:5] ** 2))
        grav = self._gravity_body()
        terms["orientation"] = float(np.sum(grav[:2] ** 2))
        terms["base_height"] = float((d.qpos[2] - cfg.target_height) ** 2)

        qd = d.qvel[self.leg_dof_adr]
        wd = d.qvel[self.wheel_dof_adr]
        qd_all = np.concatenate([qd, wd])
        terms["joint_acc"] = float(np.sum(((qd_all - self.prev_dof_vel) / cfg.control_dt) ** 2))
        terms["action_rate"] = float(np.mean((action - self.last_action) ** 2))
        tau = d.actuator_force
        terms["torques"] = float(np.sum(tau**2))
        terms["wheel_acc"] = float(np.sum(((wd - self.prev_wheel_vel) / cfg.control_dt) ** 2))

        # ---- 碰撞 / 限位 ----
        base_contact = 0.0
        for i in range(d.ncon):
            c = d.contact[i]
            if self.base_col_geom in (c.geom1, c.geom2):
                base_contact += 1.0
        terms["collision"] = base_contact

        q_leg = d.qpos[self.leg_qpos_adr]
        lo = np.minimum(q_leg - JOINT_POS_LOW, 0.0)
        hi = np.maximum(q_leg - JOINT_POS_HIGH, 0.0)
        terms["dof_limits"] = float(np.sum(lo**2 + hi**2))

        # ---- 静止时保持站姿 / 不许自己溜 ----
        # exp() 核在 v≈0 附近是平的，光靠 track_lin_vel 压不住 5 cm/s 的残余漂移，
        # 所以指令为零时额外用**线性**速度惩罚，把"站着别动"这件事直接约束死。
        if float(np.abs(cmd).sum()) < 0.05:
            terms["stand_still"] = float(np.sum(np.abs(q_leg - DEFAULT_LEG_POS)))
            terms["stand_vel"] = float(np.linalg.norm(v[:2]) + abs(wz))
        else:
            terms["stand_still"] = 0.0
            terms["stand_vel"] = 0.0

        terms["alive"] = 1.0 if not terminated else 0.0

        reward = sum(rw.get(k, 0.0) * val for k, val in terms.items())
        return reward, terms

    # ------------------------------------------------------------------ 渲染
    def render(self):
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, 480, 640)
            self._cam = mujoco.MjvCamera()
            self._cam.lookat[:] = [0, 0, 0.12]
            self._cam.distance = 0.65
            self._cam.azimuth = 130
            self._cam.elevation = -20
        self._renderer.update_scene(self.data, self._cam)
        return self._renderer.render()

    def close(self):
        self._renderer = None
