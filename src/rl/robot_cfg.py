"""w2l_12 轮腿机器人 —— 训练 / 部署共用的唯一契约。

训练环境（``w2l_env.py``）和遥控接口（``scripts/remote_control.py``）都从这里取，
所以不会出现"训的时候一个尺度、部署时另一个尺度"。

动作 / 观测
-----------
* **动作 16 维 = 12 个腿部目标角 (rad，绝对角) + 4 个轮子目标角速度 (rad/s)**。
  策略网络直接输出物理量，没有额外的 action_scale / default_pose 换算，
  下位机拿到 16 个数就能下发。
* 观测 53 维，与 RC_WheelLeg / legged_gym 的轮腿 contract 一致：
  角速度 3 + 投影重力 3 + 指令 3 + 关节角 12 + 关节角速度 12 + 轮速 4 + 上一步动作 16。

力矩环
------
腿部在仿真里用 MuJoCo 的 position 伺服（kp/kv），真机上用舵机的**力矩环**：
``tau = kp*(q_des - q) - kv*qd`` 再按电流下发。两者数学上完全等价。
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

# ---------------------------------------------------------------- 站姿 / 限位
#: 站姿关节角 (rad)，即 MJCF 里的 joint ref：
#: roll 0° / hip -30° / knee +80°（大腿后倾 30°，小腿前倾 50°）
DEFAULT_LEG_POS = np.array([0.0, -0.5236, 1.3963] * 4, dtype=np.float64)

#: 关节软限位（比 XML 的硬限位留 5° 余量）
JOINT_POS_LOW = np.array([-1.4835, -3.0543, -2.2689] * 4, dtype=np.float64)
JOINT_POS_HIGH = np.array([1.4835, 3.0543, 2.2689] * 4, dtype=np.float64)

#: 轮子目标角速度上限 (rad/s)。25 rad/s × 0.03 m ≈ 0.75 m/s
WHEEL_SPEED_LIMIT = 25.0

#: 动作空间上下界（16 维，直接就是物理量）。网络输出经 ACTION_SCALE 映射后落在这里
ACTION_LOW = np.concatenate([JOINT_POS_LOW, -WHEEL_SPEED_LIMIT * np.ones(4)])
ACTION_HIGH = np.concatenate([JOINT_POS_HIGH, WHEEL_SPEED_LIMIT * np.ones(4)])

#: 中立动作（绝对量）：站姿 + 轮子不转。策略初始输出、复位、动作延迟队列都从这里开始
NEUTRAL_ACTION = np.concatenate([DEFAULT_LEG_POS, np.zeros(4)])

#: **训练内部**的归一化尺度（和 RC_WheelLeg / legged_gym 的 action_scale 一致）。
#: 只有 w2l_env.py 用得到；导出的 ONNX 已经把它烘进输出层，部署端看不到。
ACTION_SCALE_LEG = np.array([0.125, 0.25, 0.25] * 4, dtype=np.float64)
WHEEL_VEL_SCALE = 25.0
ACTION_SCALE = np.concatenate([ACTION_SCALE_LEG, np.full(4, WHEEL_VEL_SCALE)])

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
#: 上一步动作的归一化：把绝对目标减掉中立值再除以动作尺度，等价于**训练时的归一化动作**。
#: 部署端不用自己算 —— 直接把 ctrl（绝对目标）传进 ``build_obs`` 就行。
LAST_ACTION_SCALE = 1.0
OBS_CLIP = 100.0

#: 部署端指令上限。横移（vy）物理上做不到（四个平行轮），原地转向受轮胎横向摩擦
#: 限制在 ~0.45 rad/s，所以这里按真实可达范围给，训练课程会略大一点。
CMD_LIMITS = np.array([[0.55, 0.12, 0.45]])  # vx, vy, wz

#: 机身站姿高度（base 原点离地）
STAND_HEIGHT = 0.122456


def split_action(action: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """16 维动作 -> (12 腿目标角, 4 轮目标角速度)。"""
    a = np.asarray(action, dtype=np.float64)
    return a[:12], a[12:]


def action_to_ctrl(action: np.ndarray) -> np.ndarray:
    """**训练内部**：归一化动作 ([-1,1]) -> 绝对目标（12 角度 rad + 4 轮速 rad/s）。

    导出的 ONNX 已经把这一步烘进网络了，部署端不需要调用它。
    """
    a = np.asarray(action, dtype=np.float64)
    q = np.clip(DEFAULT_LEG_POS + a[:12] * ACTION_SCALE_LEG, JOINT_POS_LOW, JOINT_POS_HIGH)
    w = a[12:] * WHEEL_VEL_SCALE
    return np.concatenate([q, w])


def clip_target(target: np.ndarray) -> np.ndarray:
    """**部署端**：ONNX 输出的绝对目标 -> 限幅后的 ctrl。"""
    return np.clip(np.asarray(target, dtype=np.float64), ACTION_LOW, ACTION_HIGH)


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

    ``last_action`` 传的**是上一步下发的绝对目标**（12 角度 rad + 4 轮速 rad/s），
    函数内部会减掉中立值、除以动作尺度，等价于训练时的归一化动作。
    部署端直接把 ``ctrl`` 传进来即可，不需要自己换算。

    ====== ==========================
    索引    含义
    ====== ==========================
    0:3    机身角速度 (body) * 0.25
    3:6    投影重力 (body)
    6:9    速度指令 (vx, vy, wz)
    9:21   12 个腿部关节角 - 默认站姿
    21:33  12 个腿部关节角速度 * 0.05
    33:37  4 个轮子角速度 * 0.05
    37:53  上一步动作（归一化到 ~[-1,1]）
    ====== ==========================
    """
    obs = np.empty(OBS_DIM, dtype=np.float32)
    obs[0:3] = np.asarray(gyro, dtype=np.float64) * ANG_VEL_SCALE
    obs[3:6] = np.asarray(gravity_body, dtype=np.float64) * GRAVITY_SCALE
    obs[6:9] = np.asarray(command, dtype=np.float64) * CMD_SCALE
    obs[9:21] = (np.asarray(q_leg, dtype=np.float64) - DEFAULT_LEG_POS) * DOF_POS_SCALE
    obs[21:33] = np.asarray(qd_leg, dtype=np.float64) * DOF_VEL_SCALE
    obs[33:37] = np.asarray(wd_wheel, dtype=np.float64) * WHEEL_VEL_OBS_SCALE
    prev = (np.asarray(last_action, dtype=np.float64) - NEUTRAL_ACTION) / ACTION_SCALE
    obs[37:53] = prev * LAST_ACTION_SCALE
    np.clip(obs, -OBS_CLIP, OBS_CLIP, out=obs)
    return obs


def deploy_contract(control_freq_hz: float = 50.0, filter_hz: float = 10.0) -> dict:
    """写进 deploy.yaml 的部署契约。真机固件 / ROS 节点照着这份表实现即可。"""
    return {
        "robot": "w2l_12",
        "contract_version": 2,
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
            {"name": "last_action", "size": 16, "scale": "((target - neutral) / action_scale)", "unit": "归一化动作"},
        ],
        "leg_joint_order": list(LEG_JOINTS),
        "wheel_joint_order": list(WHEEL_JOINTS),
        "default_leg_pos": DEFAULT_LEG_POS.tolist(),
        "action": {
            "format": "absolute",
            "layout": "0:12 = 12 个腿部目标角 (rad, 绝对角)；12:16 = 4 个轮子目标角速度 (rad/s)",
            "low": ACTION_LOW.tolist(),
            "high": ACTION_HIGH.tolist(),
            "note": "策略输出即目标值，限幅后直接写进 ctrl / 下位机",
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
