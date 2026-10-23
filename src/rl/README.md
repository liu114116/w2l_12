# w2l_12 强化学习训练与部署

四条轮腿、16 个执行器（12 个舵机关节 + 4 个轮子），**全力矩环**，
用 MuJoCo + Gymnasium + stable-baselines3 的 PPO 训练，
训练好的策略导出 ONNX，在 `scripts/remote_control.py` 里用键盘 / 手柄遥控。

```
SolidWorks ─► model/w2l_12.xml ─► prepare_assets.py ─► assets/w2l_12_train.xml
                                              │
                                              ▼
                          w2l_env.py (Gymnasium) ──► train_ppo.py (PPO)
                                              │              │
                                              │              ▼
                                              │        checkpoints/v1/model.zip
                                              │              │
                                              │              ▼
                                              │        models/policy.onnx
                                              │              │
                                              └──────────────▼
                                       scripts/remote_control.py  ◄── 键盘 / 手柄 / SLAM
```

## 快速开始

```bat
conda activate wheel_leg
cd /d D:\project\robot_leg\mujoco\w2l_12

python src\rl\prepare_assets.py                          :: 1. 生成训练用轻量 MJCF
python src\rl\train_ppo.py --tag v1 --num-envs 16 --steps 20e6   :: 2. 训练（约 70 min）
python src\rl\export_onnx.py --tag v1                    :: 3. 导出 ONNX

conda activate ms
python scripts\remote_control.py                         :: 4. 遥控
```

`conda activate ms` 只需要 `mujoco + numpy + onnxruntime (+pygame/pynput)`；
训练环境 `wheel_leg` 需要 `torch + stable-baselines3 + gymnasium + onnx`。

## 目录

| 文件 | 作用 |
|------|------|
| `robot_cfg.py` | **唯一契约**：关节顺序、站姿、动作/观测尺度、PD 增益、`build_obs`、`deploy_contract` |
| `prepare_assets.py` | 从 `model/w2l_12.xml` 生成训练用 MJCF（删 149 万面视觉网格、换伺服类型） |
| `w2l_env.py` | Gymnasium 环境：53 维观测 / 16 维动作、奖励、终止、域随机化、指令课程 |
| `vec_env.py` | 并行环境工厂（子进程只 import mujoco，不再吃一份 torch） |
| `train_ppo.py` | PPO 训练 + 定期固定指令评估 + checkpoint + 写 `deploy.yaml` |
| `export_onnx.py` | 导出 actor 到 ONNX（**12 目标角 + 4 目标轮速**），并做数值对齐校验 |
| `deploy.yaml` | 部署契约（训练时自动生成，`remote_control.py` 直接读） |
| `assets/` | 生成的训练 MJCF |
| `checkpoints/<tag>/` | SB3 权重（`model.zip` 是最终的） |
| `models/policy.onnx` | 部署用：输出 12 个目标角 (rad) + 4 个目标轮速 (rad/s) |
| `models/policy_norm.onnx` | 对照用：输出归一化动作 `[-1,1]`（和训练逐位一致） |
| `logs/<tag>/` | TensorBoard |

## 遥控窗口（自建 GLFW，不用 mujoco.viewer）

`mujoco.viewer` 把键盘绑死在自己的快捷键上（空格暂停、退格重置、`[` `]` 切相机……），
WASD 会被吃掉，相机也只能在它的小 UI 里拖。所以 `scripts/remote_control.py` 自带一个
GLFW + `mjv_updateScene` / `mjr_render` 的窗口：

| 操作 | 作用 |
|---|---|
| 鼠标左键拖拽 | 旋转视角 |
| 鼠标右键 / 中键拖拽 | 平移 |
| 滚轮 | 缩放（0.12 m ~ 4 m） |
| `F` | 跟随 / 自由相机（跟随时仍可自由环绕缩放） |
| `V` | 视角复位到默认角度 |
| `H` | 显示/隐藏帮助 |
| `WASD` / `QE` / `SPACE` / `R` / `[` `]` / `G` | 控制指令（`R` 真正复位到站姿） |

窗口左上角实时显示 `cmd`（指令）、`meas`（实测速度）、机身高度、输入设备、速度倍率和
相机模式（`FOLLOW` / `FREE` / `FIXED`）。`--snap foo.png` 可以截一张窗口图（调试相机时方便）。

> 默认是**自由相机 + 跟随**，鼠标可以直接转。只有显式写 `--camera overview` 才会切到
> 模型里的固定相机——固定相机是不能用鼠标转的，HUD 上会显示 `FIXED`。
>
> 另外 `--cmd 0.3 0 0` 这种写法会**绕过键盘**（给脚本做回归测试用），此时按键不起作用。

## 为什么用"位置伺服 / 速度伺服"，而不是直接输出力矩

真机上舵机是**力矩环**：固件每个控制周期算

```c
tau = Kp * (q_des - q) - Kd * qd;     // 限幅 ±0.98 N·m
set_current(tau / Kt);
```

也就是说**策略给目标角，舵机的力矩环负责跟**。MuJoCo 的
`<position kp=26 kv=1.6 forcerange="-0.98 0.98"/>` 执行器在数学上和它是同一个东西
（隐式积分，所以 500 Hz 也稳），所以训练时用伺服、部署时用 `tau = kp*(q_des-q) - kd*qd`，
两边严格等价。

轮子同理：`<velocity kv=0.7 forcerange="-0.29 0.29"/>` ↔ 真机速度环。

如果你确实想让策略直接输出力矩：

```bat
python src\rl\prepare_assets.py --torque
python src\rl\train_ppo.py --torque --tag v1_torque
```

动作就变成 `tau = action * [0.98 ×12, 0.29 ×4]`。这条路线对力矩标定、摩擦建模要求高得多，
第一版先用伺服路线更稳妥。

## 观测 / 动作契约（53 → 16）

| 观测 | 维度 | 说明 |
|------|-----:|------|
| 角速度 | 3 | body 系，rad/s，×0.25 |
| 投影重力 | 3 | body 系，IMU 直接给 |
| 速度指令 | 3 | `(vx, vy, wz)`，**以后 SLAM / 遥控器就接这里** |
| 腿部关节角 | 12 | 相对站姿，rad |
| 腿部关节速度 | 12 | rad/s，×0.05 |
| 轮子角速度 | 4 | rad/s，×0.05 |
| 上一步动作 | 16 | — |

动作 16 维（**导出的 `models/policy.onnx` 直接输出下面这 16 个数**）：

```
q_des[12] = 站姿[0, -30°, +80°] + action[0:12] * [0.125, 0.25, 0.25] rad   (侧摆/髋/膝)
w_des[4]  = action[12:16] * 25 rad/s
```

即 ONNX 的输出就是 **12 个腿部目标角 (rad, 绝对角) + 4 个轮子目标角速度 (rad/s)**，
`action_scale`、默认站姿、关节限位都已经烘进网络，下位机拿到 16 个数就能直接下发。
另有一份 `models/policy_norm.onnx` 输出归一化动作 `[-1,1]`，
和训练时逐位一致，方便和 RC_WheelLeg / legged_gym 的实现对照。

关节顺序固定为 `FL, FR, RL, RR`，每条腿 `roll, hip, knee`；轮子 `FL, FR, RL, RR`。
**永远从 `robot_cfg.py` 取，别写死下标。**

## 静止不溜车：部署层速度调节器（不改策略）

纯 PPO 的 `exp(-|Δv|²/0.16)` 跟踪奖励在 `v≈0` 附近非常平，
压不住 `4~6 cm/s` 的残余漂移（实测 20 s 走掉 1.3 m、偏航 68°）。
这不需要重训 —— 在部署层加一个**轮速外环**就够了：指令为零时把机身速度闭环回 0。

`robot_cfg.StandRegulator` 就是这个外环，符号是实测出来的（别凭直觉写）：

* 四个轮同时 `+Δ` → 机身朝 **+x** 走（模型前方是 −x，所以指令系 `vx` 变小）；
* **左**轮 `+Δ`、**右**轮 `−Δ` → 偏航角速度变 **负**。

所以修正量写成 `u = kp·v + ki·∫v`，直接加到 4 个轮速目标上，
指令非零时按 0.3 s 斜坡退出。

| 指标（20 s 站立） | 无调节器 | 有调节器 |
|---|---:|---:|
| 位移 | 1.28 m（还在走） | **0.15 m（2 s 后完全停住）** |
| 末速度 | 0.05 m/s | ~0.02 m/s |
| 偏航 | 68° | **7°** |

默认增益 `kp_v=30, ki_v=10`，测速 8 Hz 低通，`db_v=0.01` 死区。
**偏航通道 `kp_w/ki_w` 实测是负收益**（任何差动都会诱发轮胎横刮，反而让偏航更糟），
默认关掉，只靠速度通道就能把偏航从 68° 压到 7°。

对比实验：`python scripts\remote_control.py --no-reg`。


## 奖励

| 项 | 权重 | 作用 |
|----|-----:|------|
| `track_lin_vel` | +2.0 | `exp(-|v_xy - cmd|²/0.16)` |
| `track_ang_vel` | +1.5 | `exp(-(wz - cmd)²/0.16)` |
| `alive` | +0.6 | 活着就给，避免"早点摔倒早结束" |
| `orientation` | -2.5 | 机身水平 |
| `base_height` | -6.0 | 站姿高度 122.5 mm |
| `lin_vel_z` / `ang_vel_xy` | -1.0 / -0.05 | 抑制上下颠、横滚俯仰 |
| `torques` | -0.02 | 省电、降温 |
| `action_rate` / `joint_acc` / `wheel_acc` | 小 | 动作平滑，sim2real 关键 |
| `collision` / `dof_limits` | -1.5 / -2.0 | 机身撞地、超限位 |
| `stand_still` | -0.6 | 指令为 0 时保持站姿 |

改权重直接改 `w2l_env.py` 里的 `DEFAULT_REWARDS`，然后重训。

## 域随机化

| 项 | 范围 |
|----|------|
| 地面/轮胎摩擦 | 0.6 ~ 1.4 × |
| 机身质量 | 0.9 ~ 1.15 × |
| 质心偏移 | ±8 mm（三轴） |
| Kp / Kd | 0.85 ~ 1.15 × |
| 动作延迟 | 0 ~ 2 个控制步 |
| 观测噪声 | 关节角 ±0.01 rad、角速度 ±0.5、陀螺 ±0.05 |
| 外力推扰 | 每 3~7 s 一次，0.5 m/s 等效冲量 |

真机上最先暴露的通常是**延迟**和**舵机死区/回差**，这两项在 `EnvCfg` 里调。

## 接到 SLAM / 遥控器

策略的输入只有 3 个速度指令，所以：

```python
cmd = np.array([vx, vy, wz])          # m/s, m/s, rad/s
obs = build_obs(gyro, gravity, cmd, q_leg, qd_leg, wd_wheel, last_action)
action = policy(obs)                  # ONNX / TensorRT
ctrl = action_to_ctrl(action)         # 12 个目标角 + 4 个轮速
```

以前向为例（模型**前方是 -x**，所以指令 `vx > 0` 对应机身 -x 方向）：

| 模块 | 给什么 |
|------|--------|
| 遥控器 | 摇杆 → `vx, vy, wz` |
| SLAM / Nav2 | `/cmd_vel` → `vx, vy, wz`（差分驱动时 `vy=0`，用 `wz` 转向） |
| 视觉跟随 | 目标框偏差 → `wz`、距离 → `vx` |

`scripts/remote_control.py` 里 `CommandSource.update()` 就是这层适配器，
换成 ROS 订阅只需替换这一个方法。

## 实测能力（12M 步训出来的 v1 策略）

在 `scripts/remote_control.py` 里跑（含 10 Hz 动作滤波、域随机化关掉），
稳定值取最后 0.5 s：

| 指令 | 实测 | 说明 |
|------|------|------|
| `(0, 0, 0)` | `vx ≈ 0`，机身 121 mm | 站得住；20 s 只走 0.15 m（调节器生效后位置完全冻结） |
| `vx = 0.3 m/s` | `0.30 ~ 0.32 m/s` | 跟得很准 |
| `vx = 0.5 m/s` | `0.37 ~ 0.41 m/s` | 轮速/摩擦到顶，实际最快约 0.45 m/s |
| `wz = 0.4 rad/s` | `0.39 ~ 0.40 rad/s` | 原地转向 OK |
| `vx 0.3 + wz 0.3` | `vx 0.27`、`wz 0.13~0.24` | 边走边转会掉一点偏航 |
| `vy = 0.25 m/s` | ≈ 0 | **做不到**，见下 |

### 为什么横移做不到、偏航有上限

四条腿的轮轴**互相平行**，轮子只有滚动自由度：

* 横移需要轮子横向滑动，没有驱动源 → `vy` 基本为零。接 SLAM/Nav2 时按
  **差速车**用（`vy = 0`，靠 `wz` 转向）即可。
* 原地转向要让四个轮子同时横向刮地。直接给差速轮速（左右 ±w）测过：

  | 轮速差 | 偏航角速度 |
  |-------:|-----------:|
  | ±3 rad/s | 0.05 rad/s |
  | ±10 rad/s | 0.16 rad/s |
  | ±15 rad/s | 0.22 rad/s |

  也就是**纯轮差速最多 0.2 rad/s 左右**，再大就只是打滑和车身俯仰。
  策略能到 0.4 rad/s，是因为它同时用了腿部姿态（重心转移 + 侧摆）。
  想更快只有三条路：换麦轮/全向轮、抬腿迈步转向、或者给轮子加转向舵机。

所以 `deploy.yaml` 里给的部署指令上限是 `(0.55, 0.15, 0.45)`，
训练课程上限是 `(0.6, 0.4, 1.0)` —— 后者是为了让策略见过大指令，
前者才是这套机械真正能跟上的范围。

## 性能参考

| 项 | 数值（RTX 4060 Laptop / i7-14700HX） |
|----|--------------------------------------|
| MuJoCo 单步（轻量模型） | 23.5 µs |
| 单环境控制步（50 Hz，10 子步） | 0.30 ms |
| 16 进程并行 | ~10.4k env-step/s |
| PPO 端到端（16 envs, n_steps 512） | ~4.5k step/s → 12M 步约 45 min |
| 策略推理（ONNX，CPU） | 0.013 ms |

模型只有 1.25 kg、18 个 body，**GPU 加速仿真在这里没有意义**，
瓶颈是 Python/IPC 而不是物理；真要再快就上 `mjlab`（MuJoCo Warp）或 Isaac Lab。

## 参考

* [legged_gym](https://github.com/leggedrobotics/legged_gym) / [rsl_rl](https://github.com/leggedrobotics/rsl_rl) —— PPO + 指令课程 + 域随机化这套配方的出处
* [Wheel-Legged-Gym](https://github.com/clearlab-sustech/Wheel-Legged-Gym) —— 轮腿 legged_gym 变体
* [RC_WheelLeg](https://github.com/zeitvex/RC_WheelLeg) —— 16DOF 轮腿，mjlab + MuJoCo + ROS2/Nav2 sim2real 全链路
* [mjlab](https://github.com/mujocolab/mjlab) —— MuJoCo Warp + Isaac Lab 风格 API
* [Wheel-Legged-Lab](https://github.com/zyicome/Wheel-Legged-Lab) —— Isaac Lab 轮腿平衡/行走/跳跃
* Lee et al., *Learning Robust Autonomous Navigation and Locomotion for Wheeled-Legged Robots*, Science Robotics 9(89), 2024 —— 轮腿 + 导航的标杆工作
