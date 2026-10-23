"""w2l_12 轮腿机器人 —— 遥控接口（唯一入口）。

在 MuJoCo 里跑训练好的策略，用**键盘**或**手柄**实时遥控；
以后接 SLAM 时，只要把 ``CommandSource`` 换成 ROS / 导航模块发布的
``(vx, vy, wz)`` 就行 —— 策略本来就是吃速度指令的。

    conda activate ms
    cd /d D:\\project\\robot_leg\\mujoco\\w2l_12
    python scripts\\remote_control.py                  # 键盘 + 手柄，可视化
    python scripts\\remote_control.py --fast           # 轻量碰撞模型，帧率更高
    python scripts\\remote_control.py --serial COM3    # 同时把目标发给下位机
    python scripts\\remote_control.py --headless --seconds 8 --cmd 0.3 0 0

按键（自建窗口，窗口获得焦点时生效）
------------------------------------
    W / S      前进 / 后退          A / D      左移 / 右移
    Q / E      左转 / 右转          SPACE      急停（指令清零）
    R          复位到站姿           [ / ]      速度倍率 0.2x ~ 2.0x
    G          切换"手柄 / 键盘"     F          跟随 / 自由相机
    V          视角复位               H          显示/隐藏帮助
    ESC        退出

鼠标
----
    左键拖拽   旋转视角            右键/中键拖拽  平移
    滚轮       缩放

> 查看器是**自己写的 GLFW 窗口**，不是 ``mujoco.viewer``：
> 后者把键盘绑死在自己的快捷键上（空格暂停、退格重置、`[` `]` 切相机……），
> 会把 WASD 吃掉，相机也只能在它的小 UI 里拖。这里按键全部归我们，鼠标直接控相机。

手柄（Xbox 布局）
----------------
    左摇杆 Y  前进/后退            左摇杆 X   横移
    右摇杆 X  偏航                  Start/Back 复位/急停

**力矩环**在这里是显式算的：脚本每 2 ms 用
``tau = kp*(q_des - q) - kd*qd`` 算一次力矩再写进 ``ctrl``，
和真机上舵机电流环里跑的那一层完全一样。用 ``--fast`` 模型时，
XML 里已经带 position / velocity 伺服，这时直接把目标写进 ``ctrl``。

串口协议（``--serial``，50 Hz，小端）
-------------------------------------
    b'\\xAA\\x55' + 12*float32(腿部目标角 rad) + 4*float32(轮子目标角速度 rad/s) + uint8 校验和
    校验和 = 所有字节之和 & 0xFF。改自己固件的格式只要动 ``SerialBridge.frame``。
"""

from __future__ import annotations

import argparse
import math
import struct
import sys
import time
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
RL_DIR = PROJECT / "src" / "rl"
sys.path.insert(0, str(RL_DIR))

import mujoco  # noqa: E402

from robot_cfg import (  # noqa: E402
    ACT_DIM,
    KD_LEG,
    KP_LEG,
    KV_WHEEL,
    LEG_JOINTS,
    STAND_REG,
    StandRegulator,
    TAU_LEG_LIMIT,
    TAU_WHEEL_LIMIT,
    WHEEL_JOINTS,
    action_to_ctrl,
    build_obs,
    targets_to_action,
)

DEFAULT_POLICY = RL_DIR / "models" / "policy.onnx"
VISUAL_XML = PROJECT / "model" / "w2l_12.xml"
FAST_XML = RL_DIR / "assets" / "w2l_12_train.xml"


# ====================================================================== 策略
class Policy:
    """ONNX 策略封装（onnxruntime，不需要 torch）。"""

    def __init__(self, path: Path):
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.log_severity_level = 3
        self.sess = ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])
        self.input_name = self.sess.get_inputs()[0].name
        self.output_name = self.sess.get_outputs()[0].name
        self.obs_dim = 53
        print(f"[policy] {path.name}  ({path.stat().st_size/1024:.1f} KB)")

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        out = self.sess.run([self.output_name], {self.input_name: obs[None, :].astype(np.float32)})[0]
        return np.asarray(out, dtype=np.float64).reshape(-1)[:ACT_DIM]


# ====================================================================== 键盘
class KeyboardState:
    """窗口内键盘状态。

    由**自建 GLFW 窗口**的 key callback 喂，不用 pynput、也不经过 MuJoCo 自带查看器，
    所以 MuJoCo 那些内置快捷键（空格暂停、退格重置、`[` `]` 切相机、H 帮助……）
    完全不会参与，WASD/QE 也不会被它们吃掉。
    """

    def __init__(self):
        self.pressed: set[str] = set()

    def press(self, name: str) -> None:
        self.pressed.add(name)

    def release(self, name: str) -> None:
        self.pressed.discard(name)

    def __contains__(self, key: str) -> bool:
        return key in self.pressed

    def stop(self) -> None:
        pass


# ====================================================================== 查看器
class GLViewer:
    """自己写的 GLFW + MuJoCo 渲染窗口。

    为什么要自己写：``mujoco.viewer`` 把键盘绑死在自己的快捷键上，
    而且相机只能在它的小 UI 里拖，很容易被按键抢掉。这里：

    * 按键全部由我们处理（WASD/QE/SPACE/R/G/F/H/ESC），没有别的消费者；
    * 鼠标左键旋转、右键（或中键）平移、滚轮缩放；
    * ``F`` 切换"跟随机身"，跟随状态下依然可以自由环绕、缩放。
    """

    #: 只注册我们关心的键；其余一律忽略。
    #: 注意 python-glfw 的常量名是 ``KEY_SPACE`` / ``KEY_LEFT_BRACKET``，
    #: 不是 ``SPACE`` / ``LEFT_BRACKET`` —— 写错 ``getattr`` 会静默丢掉这个键。
    KEY_NAMES = {
        "KEY_W": "W", "KEY_S": "S", "KEY_A": "A", "KEY_D": "D",
        "KEY_Q": "Q", "KEY_E": "E", "KEY_R": "R", "KEY_G": "G",
        "KEY_F": "F", "KEY_H": "H", "KEY_V": "V",
        "KEY_SPACE": "SPACE",
        "KEY_LEFT_BRACKET": "[", "KEY_RIGHT_BRACKET": "]",
    }

    def __init__(self, model, data, keys: KeyboardState, width: int = 1280, height: int = 800,
                 title: str = "w2l_12 - remote control", camera_id: int = -1):
        import glfw

        self.glfw = glfw
        if not glfw.init():
            raise RuntimeError("glfw.init() 失败")
        glfw.window_hint(glfw.SAMPLES, 4)
        glfw.window_hint(glfw.RESIZABLE, glfw.TRUE)
        self.window = glfw.create_window(width, height, title, None, None)
        if not self.window:
            glfw.terminate()
            raise RuntimeError("创建 GLFW 窗口失败（没有可用的显示设备？）")
        glfw.make_context_current(self.window)
        glfw.swap_interval(1)
        glfw.focus_window(self.window)           # 从控制台启动时尽量把焦点抢过来

        glfw.set_key_callback(self.window, self._on_key)
        glfw.set_cursor_pos_callback(self.window, self._on_cursor)
        glfw.set_mouse_button_callback(self.window, self._on_button)
        glfw.set_scroll_callback(self.window, self._on_scroll)

        self.model, self.data, self.keys = model, data, keys
        self.cam = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(self.cam)
        self.cam.distance = 0.62
        self.cam.azimuth = 135.0
        self.cam.elevation = -16.0
        self.cam.lookat[:] = [0.0, 0.0, 0.12]
        if camera_id >= 0:                       # --camera 指定 XML 里的固定相机
            self.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            self.cam.fixedcamid = camera_id

        self.opt = mujoco.MjvOption()
        mujoco.mjv_defaultOption(self.opt)
        self.opt.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0     # 小模型阴影噪点多，关掉更干净
        self.opt.flags[mujoco.mjtRndFlag.mjRND_SKYBOX] = 1
        self.scene = mujoco.MjvScene(model, 20000)
        self.ctx = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150.value)
        self.bg = (0.10, 0.11, 0.13, 1.0)

        self.follow = True
        self.focus_body = -1
        self.show_help = True
        self._cursor = None
        self._orbit = False
        self._pan = False
        self._home = (self.cam.azimuth, self.cam.elevation, self.cam.distance, self.cam.lookat.copy())
        self._keymap = {getattr(glfw, k): v for k, v in self.KEY_NAMES.items() if hasattr(glfw, k)}

    # ---------------- 键鼠 ----------------
    def _on_key(self, window, key, scancode, action, mods) -> None:
        glfw = self.glfw
        if key == glfw.KEY_ESCAPE and action == glfw.PRESS:
            glfw.set_window_should_close(window, True)
            return
        name = self._keymap.get(key)
        if name is None:
            return
        if action == glfw.PRESS:
            self.keys.press(name)
            if name == "F":
                self.follow = not self.follow
            elif name == "H":
                self.show_help = not self.show_help
            elif name == "V":                    # 视角复位
                az, el, dist, look = self._home
                if self.cam.type == mujoco.mjtCamera.mjCAMERA_FREE:
                    self.cam.azimuth, self.cam.elevation, self.cam.distance = az, el, dist
                    self.cam.lookat[:] = look
                    self.follow = True
        elif action == glfw.RELEASE:
            self.keys.release(name)

    def _on_button(self, window, button, action, mods) -> None:
        glfw = self.glfw
        down = action == glfw.PRESS
        if button == glfw.MOUSE_BUTTON_LEFT:
            self._orbit = down
        elif button in (glfw.MOUSE_BUTTON_RIGHT, glfw.MOUSE_BUTTON_MIDDLE):
            self._pan = down

    def _on_cursor(self, window, x, y) -> None:
        if self._cursor is None:
            self._cursor = (x, y)
            return
        dx, dy = x - self._cursor[0], y - self._cursor[1]
        self._cursor = (x, y)
        if self._orbit:
            self.cam.azimuth -= dx * 0.4
            self.cam.elevation = float(np.clip(self.cam.elevation - dy * 0.4, -89.0, 89.0))
        if self._pan:
            az, el = math.radians(self.cam.azimuth), math.radians(self.cam.elevation)
            right = np.array([math.cos(az), -math.sin(az), 0.0])
            up = np.array([-math.sin(az) * math.sin(el), -math.cos(az) * math.sin(el), math.cos(el)])
            scale = self.cam.distance * 0.0012
            self.cam.lookat[:] = self.cam.lookat - right * dx * scale + up * dy * scale

    def _on_scroll(self, window, dx, dy) -> None:
        self.cam.distance = float(np.clip(self.cam.distance * (0.9 ** dy), 0.12, 4.0))

    # ---------------- 渲染 ----------------
    def camera_mode(self) -> str:
        if self.cam.type == mujoco.mjtCamera.mjCAMERA_FIXED:
            return "FIXED "
        return "FOLLOW" if self.follow else "FREE  "

    def is_running(self) -> bool:
        return not self.glfw.window_should_close(self.window)

    def sync(self, text_left: str = "", text_right: str = "",
             help_text: str = "", help_right: str = "") -> None:
        glfw = self.glfw
        if self.follow and self.focus_body >= 0:
            self.cam.lookat[:] = self.data.xpos[self.focus_body]
        w, h = glfw.get_framebuffer_size(self.window)
        viewport = mujoco.MjrRect(0, 0, max(1, w), max(1, h))
        try:
            from OpenGL import GL

            GL.glClearColor(*self.bg)
            GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        except Exception:                            # noqa: BLE001
            pass
        mujoco.mjv_updateScene(
            self.model, self.data, self.opt, None, self.cam,
            mujoco.mjtCatBit.mjCAT_ALL.value, self.scene,
        )
        mujoco.mjr_render(viewport, self.scene, self.ctx)
        if text_left or text_right:
            mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                               viewport, text_left, text_right, self.ctx)
        if help_text and self.show_help:
            mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,
                               viewport, help_text, help_right, self.ctx)
        glfw.swap_buffers(self.window)
        glfw.poll_events()

    def close(self) -> None:
        try:
            self.glfw.terminate()
        except Exception:                            # noqa: BLE001
            pass

    def snapshot(self, path) -> None:
        """把当前窗口画面存成 PNG（调试相机/做图用）。"""
        from OpenGL import GL

        import imageio.v3 as iio

        w, h = self.glfw.get_framebuffer_size(self.window)
        # 双缓冲下 glReadPixels 默认读 GL_BACK，swap 之后那里是上一帧，必须显式读前缓冲
        GL.glReadBuffer(GL.GL_FRONT)
        GL.glPixelStorei(GL.GL_PACK_ALIGNMENT, 1)
        buf = GL.glReadPixels(0, 0, w, h, GL.GL_RGB, GL.GL_UNSIGNED_BYTE)
        img = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 3)[::-1]
        iio.imwrite(str(path), img)
        print(f"[snap] {path}")


# ====================================================================== 指令
class CommandSource:
    """键盘 / 手柄 / 外部（SLAM、脚本）统一的指令源，输出 3 维速度指令。"""

    def __init__(self, limits: np.ndarray, dt: float, accel: float = 1.6):
        self.limits = np.asarray(limits, dtype=np.float64)
        self.cmd = np.zeros(3)
        self.scale = 1.0
        self.accel = accel
        self.dt = dt
        # 先默认键盘；检测到手柄再切过去（否则没手柄时键盘永远读不到）
        self.use_gamepad = False
        self.keys = KeyboardState()
        self.on_reset = None                 # main() 里挂上真正的复位函数
        self.pad = None
        self._init_pad()

    def _init_pad(self) -> None:
        try:
            import pygame

            pygame.init()
            pygame.joystick.init()
            if pygame.joystick.get_count() > 0:
                self.pad = pygame.joystick.Joystick(0)
                self.pad.init()
                self.use_gamepad = True
                print(f"[pad] {self.pad.get_name()}（G 键可切回键盘）")
            else:
                print("[pad] 没检测到手柄 -> 用键盘")
        except Exception as exc:                     # noqa: BLE001
            print(f"[pad] pygame 不可用 ({exc}) -> 用键盘")

    def update(self) -> np.ndarray:
        k = self.keys
        t = np.zeros(3)

        if "R" in k:
            self.cmd[:] = 0.0
            k.pressed.discard("R")           # 一次按键只复位一次
            if self.on_reset is not None:
                self.on_reset()
                print("[reset] 复位到站姿")
        if "G" in k:
            self.use_gamepad = (not self.use_gamepad) and (self.pad is not None)
            k.pressed.discard("G")
            print(f"[input] {'手柄' if self.use_gamepad else '键盘'}")
        if "]" in k:
            self.scale = min(2.0, self.scale + 0.02)
        if "[" in k:
            self.scale = max(0.2, self.scale - 0.02)

        if not self.use_gamepad:
            t[0] = float("W" in k) - float("S" in k)
            t[1] = float("A" in k) - float("D" in k)
            t[2] = float("Q" in k) - float("E" in k)

        if self.pad is not None and self.use_gamepad:
            try:
                import pygame

                pygame.event.pump()
                ax_lx, ax_ly = self.pad.get_axis(0), self.pad.get_axis(1)
                ax_rx = self.pad.get_axis(3) if self.pad.get_numaxes() > 3 else self.pad.get_axis(2)
                t[0] += -ax_ly if abs(ax_ly) > 0.12 else 0.0
                t[1] += -ax_lx if abs(ax_lx) > 0.12 else 0.0
                t[2] += -ax_rx if abs(ax_rx) > 0.12 else 0.0
                if self.pad.get_button(7) or self.pad.get_button(6):
                    self.cmd[:] = 0.0
            except Exception:                       # noqa: BLE001
                self.pad = None

        if "SPACE" in k:
            t[:] = 0.0
            self.cmd[:] = 0.0

        t = np.clip(t, -1.0, 1.0) * self.limits * self.scale
        step = self.accel * self.dt
        self.cmd += np.clip(t - self.cmd, -step, step)
        return self.cmd.copy()

    def close(self):
        self.keys.stop()


# ====================================================================== 串口
class SerialBridge:
    """把 16 维目标通过串口发给下位机（STM32）。协议见模块 docstring。"""

    def __init__(self, port: str, baud: int = 921600):
        import serial  # pyserial

        self.ser = serial.Serial(port, baud, timeout=0)
        print(f"[serial] {port} @ {baud}")

    @staticmethod
    def frame(ctrl: np.ndarray) -> bytes:
        body = struct.pack("<16f", *np.asarray(ctrl, dtype=np.float32).tolist())
        frame = b"\xAA\x55" + body
        return frame + bytes([sum(frame) & 0xFF])

    def send(self, ctrl: np.ndarray) -> None:
        self.ser.write(self.frame(ctrl))

    def close(self) -> None:
        self.ser.close()


# ====================================================================== 主程序
def main() -> int:
    p = argparse.ArgumentParser(description="w2l_12 遥控接口")
    p.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    p.add_argument("--xml", type=Path, default=None)
    p.add_argument("--fast", action="store_true", help="用去掉视觉网格的轻量模型")
    p.add_argument("--deploy", type=Path, default=RL_DIR / "deploy.yaml")
    p.add_argument("--headless", action="store_true")
    p.add_argument("--seconds", type=float, default=0.0)
    p.add_argument("--cmd", type=float, nargs=3, default=None, metavar=("VX", "VY", "WZ"))
    p.add_argument("--serial", default=None)
    p.add_argument("--baud", type=int, default=921600)
    p.add_argument("--record", type=Path, default=None, help="录 mp4")
    p.add_argument("--camera", default=None,
                   help="用 XML 里的固定相机（overview/side/front）；默认自由相机 + 鼠标控制")
    p.add_argument("--no-reg", action="store_true", help="关掉静止调节器（对比用）")
    p.add_argument("--snap", type=Path, default=None, help="运行中截一张窗口图到 PNG")
    args = p.parse_args()

    # ---------------- 契约 ----------------
    limits = np.array([0.6, 0.4, 1.0])
    filter_hz = 10.0
    if args.deploy.exists():
        try:
            import yaml

            contract = yaml.safe_load(args.deploy.read_text(encoding="utf-8"))
            cl = contract.get("command_limits", {})
            limits = np.array([cl.get("vx", 0.6), cl.get("vy", 0.4), cl.get("wz", 1.0)], dtype=np.float64)
            filter_hz = float(contract.get("action_filter_hz", 10.0))
            print(f"[deploy] limits={limits} filter={filter_hz}Hz")
        except Exception as exc:                     # noqa: BLE001
            print(f"[deploy] 读契约失败（用默认值）: {exc}")

    if not args.policy.exists():
        print(f"[错误] 找不到策略 {args.policy}\n"
              f"       训练：python src/rl/train_ppo.py --tag v1\n"
              f"       导出：python src/rl/export_onnx.py --tag v1")
        return 1

    # ---------------- 模型 ----------------
    xml = args.xml or (FAST_XML if args.fast else VISUAL_XML)
    model = mujoco.MjModel.from_xml_path(str(xml))
    data = mujoco.MjData(model)
    print(f"[sim] {xml.name}  ngeom={model.ngeom}")

    jid = lambda n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)  # noqa: E731
    aid = lambda n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)  # noqa: E731
    leg_qadr = np.array([model.jnt_qposadr[jid(n)] for n in LEG_JOINTS])
    leg_dadr = np.array([model.jnt_dofadr[jid(n)] for n in LEG_JOINTS])
    wheel_dadr = np.array([model.jnt_dofadr[jid(n)] for n in WHEEL_JOINTS])
    act_leg = np.array([aid(n) for n in LEG_JOINTS])
    act_wheel = np.array([aid(n) for n in WHEEL_JOINTS])
    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
    cam_id = -1
    if args.camera:
        cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, args.camera)
        if cam_id < 0:
            print(f"[view] 模型里没有相机 {args.camera!r}，改用自由相机")

    # 执行器类型：biastype>0 说明 XML 里已经是 position/velocity 伺服
    servo_in_xml = bool(model.actuator_biastype[act_leg[0]] != mujoco.mjtBias.mjBIAS_NONE)
    print(f"[sim] 执行器 = {'XML 内置位置/速度伺服' if servo_in_xml else '力矩环（脚本里算 PD）'}")

    policy = Policy(args.policy)
    # 输出格式：policy.onnx 是"12 目标角 + 4 目标轮速"，policy_norm.onnx 是归一化动作
    out_fmt = "norm"
    side = Path(str(args.policy) + ".json")
    if side.exists():
        try:
            import json

            out_fmt = json.loads(side.read_text(encoding="utf-8")).get("output", {}).get("format", "norm")
        except Exception:                           # noqa: BLE001
            pass
    elif args.policy.name == "policy.onnx":
        out_fmt = "targets"
    print(f"[policy] 输出格式 = {out_fmt} "
          f"({'12 个腿部目标角(rad) + 4 个轮速(rad/s)' if out_fmt == 'targets' else '归一化动作 [-1,1]'})")

    dt = 0.02
    substeps = 10
    cmd_src = CommandSource(limits, dt)
    if args.cmd is not None:
        cmd_src.limits[:] = np.abs(args.cmd) + 1e-6
    bridge = SerialBridge(args.serial, args.baud) if args.serial else None

    alpha = 1.0 - math.exp(-2.0 * math.pi * filter_hz * dt)
    filtered_action = np.zeros(ACT_DIM)
    last_action = np.zeros(ACT_DIM)
    regulator = StandRegulator(dt)          # 静止时压残余漂移（不改策略）
    use_reg = not args.no_reg

    def reset() -> None:
        mujoco.mj_resetDataKeyframe(model, data, 0)
        mujoco.mj_forward(model, data)
        filtered_action[:] = 0.0
        last_action[:] = 0.0
        cmd_src.cmd[:] = 0.0
        regulator.reset()
        for i in range(4):
            data.xfrc_applied[base_id, :] = 0.0

    reset()
    cmd_src.on_reset = reset          # R 键真正复位

    def torque_ctrl(q_des: np.ndarray, w_des: np.ndarray) -> None:
        """力矩环：tau = kp*(q_des-q) - kd*qd，逐子步更新（与真机电流环一致）。"""
        q = data.qpos[leg_qadr]
        qd = data.qvel[leg_dadr]
        wd = data.qvel[wheel_dadr]
        tau_leg = np.clip(KP_LEG * (q_des - q) - KD_LEG * qd, -TAU_LEG_LIMIT, TAU_LEG_LIMIT)
        tau_wheel = np.clip(KV_WHEEL * (w_des - wd), -TAU_WHEEL_LIMIT, TAU_WHEEL_LIMIT)
        data.ctrl[act_leg] = tau_leg
        data.ctrl[act_wheel] = tau_wheel

    viewer = None
    if not args.headless:
        try:
            viewer = GLViewer(model, data, cmd_src.keys, camera_id=cam_id)
            viewer.focus_body = base_id
            print("[view] 左键旋转 | 右键/中键平移 | 滚轮缩放 | F 跟随 | H 帮助 | ESC 退出")
        except Exception as exc:                     # noqa: BLE001
            print(f"[view] 自建窗口失败：{exc}\n"
                  f"       可以用 --headless 无界面跑，或者检查显卡/显示环境")
            return 1

    renderer = writer = None
    rec_cam = None
    if args.record is not None:
        import imageio

        renderer = mujoco.Renderer(model, 480, 720)
        renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False   # 小模型阴影噪点多，关掉
        rec_cam = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(rec_cam)
        rec_cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        rec_cam.trackbodyid = base_id
        rec_cam.distance = 0.62
        rec_cam.elevation = -16.0
        rec_cam.azimuth = 135.0
        writer = imageio.get_writer(str(args.record), fps=25)

    # ---------------- 主循环 ----------------
    print("\n  控制频率 50 Hz。WASD 走，QE 转，SPACE 停，R 复位，ESC 退出\n", flush=True)
    t_next = time.perf_counter()
    step = 0
    try:
        while True:
            if viewer is not None and not viewer.is_running():
                break
            now = time.perf_counter()
            if now < t_next:
                time.sleep(min(0.002, t_next - now))
                continue
            t_next = max(t_next + dt, now)

            cmd = np.asarray(args.cmd, dtype=np.float64) if args.cmd is not None else cmd_src.update()

            quat = data.qpos[3:7]
            rot = np.zeros(9)
            mujoco.mju_quat2Mat(rot, quat)
            rot = rot.reshape(3, 3)
            grav = rot.T @ np.array([0.0, 0.0, -1.0])
            obs = build_obs(data.qvel[3:6], grav, cmd, data.qpos[leg_qadr],
                            data.qvel[leg_dadr], data.qvel[wheel_dadr], last_action)

            out = policy(obs)
            action = targets_to_action(out) if out_fmt == "targets" else np.clip(out, -1.0, 1.0)
            filtered_action += alpha * (action - filtered_action)
            last_action = action.copy()

            ctrl = action_to_ctrl(filtered_action)
            v_body = rot.T @ data.qvel[0:3]
            if use_reg:
                ctrl = regulator.update(ctrl, -v_body[0], data.qvel[5], cmd)
            if servo_in_xml:
                data.ctrl[act_leg] = ctrl[:12]
                data.ctrl[act_wheel] = ctrl[12:]
                mujoco.mj_step(model, data, nstep=substeps)
            else:
                for _ in range(substeps):
                    torque_ctrl(ctrl[:12], ctrl[12:])
                    mujoco.mj_step(model, data)

            if bridge is not None:
                bridge.send(ctrl)
            step += 1

            if writer is not None and step % 2 == 0:
                renderer.update_scene(data, camera=rec_cam)
                writer.append_data(renderer.render())

            if viewer is not None:
                v = rot.T @ data.qvel[0:3]
                hud = (f"cmd  vx {cmd[0]:+.2f}  vy {cmd[1]:+.2f}  wz {cmd[2]:+.2f}   "
                       f"meas vx {-v[0]:+.2f}  wz {data.qvel[5]:+.2f}   h {data.qpos[2]*1000:.0f}mm")
                info = (f"{'PAD' if cmd_src.use_gamepad else 'KEY'}  x{cmd_src.scale:.1f}  "
                        f"{'REG' if regulator.blend > 0.5 else '   '}  {viewer.camera_mode()}")
                help_text = ("WASD move | QE turn | SPACE stop | R reset | [ ] speed | "
                             "G pad | F follow | V view | H help | ESC quit")
                help_right = ""
                viewer.sync(hud, info, help_text, help_right)

            if args.snap is not None and step == 75:
                try:
                    viewer.snapshot(args.snap)
                except Exception as exc:             # noqa: BLE001
                    print(f"[snap] 失败: {exc}")

            if step % 25 == 0:
                v = rot.T @ data.qvel[0:3]
                print(f"t={step*dt:6.2f}s cmd=[{cmd[0]:+.2f} {cmd[1]:+.2f} {cmd[2]:+.2f}] "
                      f"v=[{-v[0]:+.2f} {-v[1]:+.2f} {data.qvel[5]:+.2f}] "
                      f"h={data.qpos[2]*1000:5.1f}mm pos=({data.qpos[0]:+.2f},{data.qpos[1]:+.2f}) "
                      f"{'REG' if regulator.blend > 0.5 else '---'}", flush=True)

            if args.seconds > 0 and step * dt >= args.seconds:
                break
    except KeyboardInterrupt:
        print("\n[exit] 用户退出")
    finally:
        cmd_src.close()
        if writer is not None:
            writer.close()
        if renderer is not None:
            renderer.close()
        if bridge is not None:
            bridge.close()
        if viewer is not None:
            viewer.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
