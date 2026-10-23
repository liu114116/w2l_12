"""w2l_12 轮腿机器人 —— 训练 / 部署共用的唯一契约。

这里定义的关节顺序、默认站姿、观测和动作尺度、PD 增益，
训练环境（src/rl/w2l_env.py）和遥控接口（scripts/remote_control.py）都从这里取，
所以**永远不会出现训练和部署对不上**的问题。

约定
----
* 动作 16 维 = 12 个腿部关节目标角 (rad) + 4 个轮子目标角速度 (rad/s)。
* 观测 53 维，与 RC_WheelLeg / legged_gym 的轮腿 contract 一致：
  角速度 3 + 投影重力 3 + 指令 3 + 关节角 12 + 关节角速度 12 + 轮速 4 + 上一步动作 16。
* 腿部执行器在仿真里用 MuJoCo 的 position 伺服（kp/kv），
  真机上用舵机的**力矩环**：tau = kp*(q_des - q) - kv*qd，再按电流下发。
  两者数学上完全等价，所以策略可以直搬。
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------- 关节顺序
LEGS = ("FL", "FR", "RL", "RR")
LEG_KINDS = ("roll", "hip", "knee")

#: 12 个腿部关节，观测 / 动作索引顺序
LEG_JOINTS = tuple(f"{k}_{leg}" for leg in LEGS for k in LEG_KINDS)
#: 4 个轮子关节
WHEEL_JOINTS = tuple(f"wheel_{leg}" for leg in LEGS)

#: 全部 16 个执行器（= 动作顺序）
ACTUATOR_ORDER = LEG_JOINTS + WHEEL_JOINTS

# ---------------------------------------------------------------- 站姿 / 尺度
#: 站姿关节角 (rad)，即 MJCF 里的 joint ref：
#: roll 0° / hip -30° / knee +80°（大腿后倾 30°，小腿前倾 50°）
DEFAULT_LEG_POS = np.array([0.0, -0.5236, 1.3963] * 4, dtype=np.float64)

#: 动作缩放：侧摆 0.125 rad，髋/膝 0.25 rad（与 RC_WheelLeg 一致）
ACTION_SCALE_LEG = np.array([0.125, 0.25, 0.25] * 4, dtype=np.float64)

#: 轮子动作 = 归一化动作 * 该值 (rad/s)
#: 25 rad/s * 0.03 m = 0.75 m/s 轮速上限
WHEEL_VEL_SCALE = 25.0

#: 关节软限位（用于 clip 目标角，比 XML 的硬限位留 5° 余量）
JOINT_POS_LOW = np.array([-1.4835, -3.0543, -2.2689] * 4, dtype=np.float64)
JOINT_POS_HIGH = np.array([1.4835, 3.0543, 2.2689] * 4, dtype=np.float64)

# ---------------------------------------------------------------- 伺服增益
#: MJCF 里 servo / wheelmotor 两个 default class 用的就是这两组值
KP_LEG = 26.0     # N·m/rad
KD_LEG = 1.6      # N·m/(rad/s)
KV_WHEEL = 0.7    # N·m/(rad/s)
TAU_LEG_LIMIT = 0.98   # N·m
TAU_WHEEL_LIMIT = 0.29 # N·m

# ---------------------------------------------------------------- 观测尺度
OBS_DIM = 53
ACT_DIM = 16

ANG_VEL_SCALE = 0.25
GRAVITY_SCALE = 1.0
CMD_SCALE = np.array([1.0, 1.0, 1.0])
DOF_POS_SCALE = 1.0
DOF_VEL_SCALE = 0.05
WHEEL_VEL_OBS_SCALE = 0.05
LAST_ACTION_SCALE = 1.0
OBS_CLIP = 100.0

#: 指令范围：训练课程的上限是 (0.6, 0.4, 1.0)，
#: 但**实测可用**的范围受轮胎横向摩擦限制（4 个平行轮原地转向要横刮），
#: 偏航最高只有 ~0.45 rad/s，横移（vy）几乎做不到，所以部署端按这个给。
CMD_LIMITS = np.array([[0.55, 0.15, 0.45]])  # vx, vy, wz

#: 机身站姿高度（base 原点离地）
STAND_HEIGHT = 0.122456


def split_action(action: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """16 维动作 -> (12 腿目标角, 4 轮目标角速度)。"""
    a = np.asarray(action, dtype=np.float64)
    return a[:12], a[12:]


def action_to_ctrl(action: np.ndarray) -> np.ndarray:
    """16 维动作 -> MuJoCo ctrl（position 伺服给目标角，velocity 伺服给目标角速度）。"""
    a_leg, a_wheel = split_action(action)
    q_target = DEFAULT_LEG_POS + a_leg * ACTION_SCALE_LEG
    q_target = np.clip(q_target, JOINT_POS_LOW, JOINT_POS_HIGH)
    w_target = a_wheel * WHEEL_VEL_SCALE
    return np.concatenate([q_target, w_target])


def action_to_joint_target(action: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """真机用：返回 (12 个目标角, 4 个目标轮速)，与 action_to_ctrl 完全一致。"""
    ctrl = action_to_ctrl(action)
    return ctrl[:12], ctrl[12:]


def targets_to_action(target: np.ndarray) -> np.ndarray:
    """``models/policy.onnx`` 输出的是物理目标，这里反解回归一化动作。

    反解出来的就是策略原本的动作（关节限位饱和时会有极小误差），
    这样观测里的 ``last_action`` 才能和训练时保持一致。
    """
    t = np.asarray(target, dtype=np.float64).reshape(-1)
    a = np.empty(16)
    a[:12] = (t[:12] - DEFAULT_LEG_POS) / ACTION_SCALE_LEG
    a[12:] = t[12:] / WHEEL_VEL_SCALE
    return np.clip(a, -1.0, 1.0)


def build_obs(
    gyro: np.ndarray,
    gravity_body: np.ndarray,
    command: np.ndarray,
    q_leg: np.ndarray,
    qd_leg: np.ndarray,
    wd_wheel: np.ndarray,
    last_action: np.ndarray,
) -> np.ndarray:
    """53 维观测。训练环境和部署端都调这一个函数，保证一字不差。

    ====== ==========================
    索引    含义
    ====== ==========================
    0:3    机身角速度 (body) * 0.25
    3:6    投影重力 (body)
    6:9    速度指令 (vx, vy, wz)
    9:21   12 个腿部关节角 - 默认站姿
    21:33  12 个腿部关节角速度 * 0.05
    33:37  4 个轮子角速度 * 0.05
    37:53  上一步动作
    ====== ==========================
    """
    obs = np.empty(OBS_DIM, dtype=np.float32)
    obs[0:3] = np.asarray(gyro, dtype=np.float64) * ANG_VEL_SCALE
    obs[3:6] = np.asarray(gravity_body, dtype=np.float64) * GRAVITY_SCALE
    obs[6:9] = np.asarray(command, dtype=np.float64) * CMD_SCALE
    obs[9:21] = (np.asarray(q_leg, dtype=np.float64) - DEFAULT_LEG_POS) * DOF_POS_SCALE
    obs[21:33] = np.asarray(qd_leg, dtype=np.float64) * DOF_VEL_SCALE
    obs[33:37] = np.asarray(wd_wheel, dtype=np.float64) * WHEEL_VEL_OBS_SCALE
    obs[37:53] = np.asarray(last_action, dtype=np.float64) * LAST_ACTION_SCALE
    np.clip(obs, -OBS_CLIP, OBS_CLIP, out=obs)
    return obs


#: 静止调节器：**不改策略**，只在指令≈0 时用轮速闭环压掉残余漂移。
#: 纯 PPO 的 exp 跟踪奖励在 v≈0 附近太平，压不住 4~6 cm/s 的溜车，
#: 这一层是把"站着别动"变成硬约束的最小改动（固件里同样加这段即可）。
STAND_REG = {
    "v_eps": 0.03,       # |cmd| 范数小于它才启用
    "kp_v": 30.0,        # rad/s per (m/s)
    "ki_v": 10.0,
    "kp_w": 0.0,         # 偏航通道实测是负收益（见 README），默认关掉
    "ki_w": 0.0,
    "db_v": 0.01,        # 速度死区，避免零点附近来回追
    "db_w": 0.0,
    "max_delta": 10.0,   # 单个轮子的修正上限 rad/s
    "ramp_s": 0.3,       # 启用/退出的渐变时间
    "int_limit": 0.5,
    "lpf_hz": 8.0,       # 测速低通，压掉轮子打滑的高频噪声
}


class StandRegulator:
    """静止时把机身速度闭环回 0 的轮速外环。

    符号来自实测（很重要，别凭直觉写）：

    * 四个轮速同时 ``+Δ`` → 机身朝 **+x** 走（模型前方是 −x，所以指令系 vx 变小）；
    * **左**侧轮 ``+Δ``、**右**侧轮 ``−Δ`` → 偏航角速度变 **负**。

    所以两路修正都写成 ``+k * 测量值`` 的形式：
    ``u_v = kp_v*vx + ki_v*∫vx`` 直接加到四个轮子上；
    ``u_w = kp_w*wz + ki_w*∫wz`` 是左 +、右 − 的差动量。
    """

    def __init__(self, dt: float, cfg: dict | None = None):
        self.dt = dt
        self.cfg = dict(STAND_REG if cfg is None else cfg)
        self.reset()

    def reset(self) -> None:
        self.ivx = 0.0
        self.iwz = 0.0
        self.blend = 0.0
        self.vx_f = 0.0
        self.wz_f = 0.0

    def update(self, ctrl: np.ndarray, vx: float, wz: float, cmd: np.ndarray) -> np.ndarray:
        c = self.cfg
        a = 1.0 - np.exp(-2.0 * np.pi * c.get("lpf_hz", 4.0) * self.dt)
        self.vx_f += a * (vx - self.vx_f)
        self.wz_f += a * (wz - self.wz_f)

        want = float(np.abs(cmd).sum()) < c["v_eps"]
        rate = self.dt / max(1e-6, c["ramp_s"])
        self.blend += float(np.clip((1.0 if want else 0.0) - self.blend, -rate, rate))
        if self.blend <= 1e-3:
            self.reset()
            return ctrl

        self.ivx = float(np.clip(self.ivx + self.vx_f * self.dt, -c["int_limit"], c["int_limit"]))
        self.iwz = float(np.clip(self.iwz + self.wz_f * self.dt, -c["int_limit"], c["int_limit"]))
        # 死区：接近 0 时不再输出，避免在零点附近反复"追"
        evx = 0.0 if abs(self.vx_f) < c.get("db_v", 0.0) else self.vx_f
        ewz = 0.0 if abs(self.wz_f) < c.get("db_w", 0.0) else self.wz_f
        u_v = float(np.clip(c["kp_v"] * evx + c["ki_v"] * self.ivx, -c["max_delta"], c["max_delta"]))
        u_w = float(np.clip(c["kp_w"] * ewz + c["ki_w"] * self.iwz, -c["max_delta"], c["max_delta"]))

        out = np.array(ctrl, dtype=np.float64, copy=True)
        delta = self.blend * np.array([u_v + u_w, u_v - u_w, u_v + u_w, u_v - u_w])
        out[12:16] = np.clip(out[12:16] + delta, -3.0 * WHEEL_VEL_SCALE, 3.0 * WHEEL_VEL_SCALE)
        return out


def deploy_contract(control_freq_hz: float = 50.0, filter_hz: float = 10.0) -> dict:
    """写进 deploy.yaml 的部署契约。真机固件/ROS 节点照着这份表实现即可。"""
    return {
        "robot": "w2l_12",
        "contract_version": 1,
        "control_freq_hz": control_freq_hz,
        "action_filter_hz": filter_hz,
        "obs_dim": OBS_DIM,
        "act_dim": ACT_DIM,
        "obs_layout": [
            {"name": "base_ang_vel", "size": 3, "scale": ANG_VEL_SCALE, "frame": "body", "unit": "rad/s"},
            {"name": "projected_gravity", "size": 3, "scale": GRAVITY_SCALE, "frame": "body"},
            {"name": "command", "size": 3, "scale": list(map(float, CMD_SCALE)), "unit": "[m/s, m/s, rad/s]"},
            {"name": "leg_joint_pos_rel", "size": 12, "scale": DOF_POS_SCALE, "unit": "rad"},
            {"name": "leg_joint_vel", "size": 12, "scale": DOF_VEL_SCALE, "unit": "rad/s"},
            {"name": "wheel_vel", "size": 4, "scale": WHEEL_VEL_OBS_SCALE, "unit": "rad/s"},
            {"name": "last_action", "size": 16, "scale": LAST_ACTION_SCALE},
        ],
        "leg_joint_order": list(LEG_JOINTS),
        "wheel_joint_order": list(WHEEL_JOINTS),
        "default_leg_pos": DEFAULT_LEG_POS.tolist(),
        "action": {
            "leg_pos_scale": ACTION_SCALE_LEG.tolist(),
            "wheel_vel_scale": WHEEL_VEL_SCALE,
            "leg_target_limit": [JOINT_POS_LOW.tolist(), JOINT_POS_HIGH.tolist()],
            "note": "action[0:12] -> q_des = default + a*scale; action[12:16] -> w_des = a*wheel_vel_scale",
        },
        "servo": {
            "kp_leg": KP_LEG,
            "kd_leg": KD_LEG,
            "kv_wheel": KV_WHEEL,
            "tau_leg_limit": TAU_LEG_LIMIT,
            "tau_wheel_limit": TAU_WHEEL_LIMIT,
            "law": "tau = kp*(q_des - q) - kd*qd  (per leg joint, clipped)",
        },
        "command_limits": {"vx": float(CMD_LIMITS[0, 0]), "vy": float(CMD_LIMITS[0, 1]), "wz": float(CMD_LIMITS[0, 2])},
        "stand_height_m": STAND_HEIGHT,
        "forward_axis_note": "模型前方是 -x；指令 vx>0 表示机器人朝自己的前方走",
    }
