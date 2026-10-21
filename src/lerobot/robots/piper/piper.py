"""
Piper 单臂机器人接口，适配 lerobot 的 Robot 基类。

硬件依赖:
    pip install piper_sdk pyrealsense2

用法:
    与 lerobot-record 配合使用:
        lerobot-record --robot.type=piper --robot.can_port=can0 ...
"""

from __future__ import annotations

import logging
import time
from functools import cached_property
from typing import Any

import numpy as np

from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..robot import Robot
from .configuration_piper import PiperConfig

logger = logging.getLogger(__name__)

JOINT_NAMES = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "gripper"]

# piper_sdk 单位换算常量
JOINT_RAD_FACTOR = 0.001 / 180 * 3.1415926  # sdk 原始值 → 弧度
GRIPPER_FACTOR = 0.001 / 70  # sdk 原始值 → [0, 1]
JOINT_TO_SDK = 1.0 / JOINT_RAD_FACTOR  # 弧度 → sdk 原始值
GRIPPER_TO_SDK = 70 * 1000  # [0, 1] → sdk 原始值


class Piper(Robot):
    """Piper 6-DoF 单臂 + 双 RealSense 相机。"""

    config_class = PiperConfig
    name = "piper"

    def __init__(self, config: PiperConfig):
        super().__init__(config)
        self.config = config
        self._sdk = None
        self._pipelines: dict[str, Any] = {}
        self._connected = False
        # 框架用 len(robot.cameras) 计算 image_writer 线程数
        self.cameras = {name: None for name in config.camera_serials}

    # ------------------------------------------------------------------
    # Features
    # ------------------------------------------------------------------

    @cached_property
    def observation_features(self) -> dict:
        h, w = self.config.image_height, self.config.image_width
        features = {f"{name}.pos": float for name in JOINT_NAMES}
        for cam_name in self.config.camera_serials:
            features[cam_name] = (h, w, 3)
        return features

    @cached_property
    def action_features(self) -> dict:
        return {f"{name}.pos": float for name in JOINT_NAMES}

    # ------------------------------------------------------------------
    # Connection state
    # ------------------------------------------------------------------

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def is_calibrated(self) -> bool:
        return True  # Piper 无需 lerobot 侧校准

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def connect(self, calibrate: bool = True) -> None:
        if self._connected:
            raise DeviceAlreadyConnectedError(f"{self} is already connected.")

        if set(self.config.camera_serials) != {"cam_high", "cam_wrist"}:
            raise ValueError("Configure cam_high and cam_wrist camera serials")
        if len(set(self.config.camera_serials.values())) != 2 or not all(self.config.camera_serials.values()):
            raise ValueError("Camera serials must be non-empty and distinct")

        try:
            # 连接机械臂
            try:
                from piper_sdk import C_PiperInterface_V2
            except ImportError as e:
                raise ImportError("请安装 piper_sdk: pip install piper_sdk") from e

            logger.info(f"Connecting to Piper arm on {self.config.can_port} ...")
            sdk = C_PiperInterface_V2(self.config.can_port)
            self._sdk = sdk
            sdk.ConnectPort()
            sdk.EnableArm(7)
            self._enable_arm(sdk)
            sdk.GripperCtrl(0, 1000, 0x01, 0)
            self._sdk = sdk
            logger.info("Piper arm connected.")

            # 连接相机
            try:
                import pyrealsense2 as rs
            except ImportError as e:
                raise ImportError("请安装 pyrealsense2: pip install pyrealsense2") from e

            for cam_name, serial in self.config.camera_serials.items():
                pipeline = rs.pipeline()
                cfg = rs.config()
                cfg.enable_device(serial)
                cfg.enable_stream(
                    rs.stream.color, self.config.image_width, self.config.image_height, rs.format.bgr8, 30
                )
                pipeline.start(cfg)
                self._pipelines[cam_name] = pipeline
                logger.info(f"Camera {cam_name} (SN:{serial}) connected.")

            self._connected = True
        except BaseException:
            self.disconnect()
            raise

    def calibrate(self) -> None:
        pass  # Piper 不需要 lerobot 侧校准

    def configure(self) -> None:
        pass

    def disconnect(self) -> None:
        """Release complete or partial connections; always stop both cameras."""
        if self._sdk is not None:
            for operation in (
                lambda: self._sdk.DisableArm(7),
                lambda: self._sdk.GripperCtrl(0, 1000, 0x02, 0),
            ):
                try:
                    operation()
                except Exception:
                    logger.exception("Failed to disable a PiPer actuator")
            try:
                self._sdk.DisconnectPort()
            except Exception:
                logger.exception("Failed to close the CAN interface")
            self._sdk = None
        for pipeline in self._pipelines.values():
            try:
                pipeline.stop()
            except Exception:
                logger.exception("Failed to stop a camera")
        self._pipelines.clear()
        self._connected = False

    # ------------------------------------------------------------------
    # Observation & Action
    # ------------------------------------------------------------------

    def get_observation(self) -> dict[str, Any]:
        if not self._connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        obs = {}

        # 关节状态 (弧度) + 夹爪 ([0,1])
        joint_msg = self._sdk.GetArmJointMsgs()
        js = joint_msg.joint_state
        joints = (
            np.array(
                [
                    js.joint_1,
                    js.joint_2,
                    js.joint_3,
                    js.joint_4,
                    js.joint_5,
                    js.joint_6,
                ]
            )
            * JOINT_RAD_FACTOR
        )
        gripper = self._sdk.GetArmGripperMsgs().gripper_state.grippers_angle * GRIPPER_FACTOR

        for i, name in enumerate(JOINT_NAMES[:-1]):
            obs[f"{name}.pos"] = float(joints[i])
        obs["gripper.pos"] = float(gripper)

        # 相机图像 (RGB, HWC)
        for cam_name, pipeline in self._pipelines.items():
            frames = pipeline.wait_for_frames(timeout_ms=5000)
            color_frame = frames.get_color_frame()
            bgr = np.asanyarray(color_frame.get_data())
            rgb = bgr[:, :, ::-1].copy()
            obs[cam_name] = rgb

        return obs

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        if not self._connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        joints = np.array([action[f"{name}.pos"] for name in JOINT_NAMES[:-1]], dtype=np.float64)
        gripper = float(action["gripper.pos"])

        if not np.isfinite(joints).all() or not np.isfinite(gripper):
            raise ValueError("Refusing non-finite robot actions")
        # Preserve the joint limits used by the supplied deployment script.
        limits = np.deg2rad([[-150, 150], [0, 180], [-170, 0], [-100, 100], [-70, 70], [-120, 120]])
        joints = np.clip(joints, limits[:, 0], limits[:, 1])
        gripper = float(np.clip(gripper, 0, 1))
        action = {
            **{f"{name}.pos": float(joints[i]) for i, name in enumerate(JOINT_NAMES[:-1])},
            "gripper.pos": gripper,
        }

        # 发送关节指令
        factor = 57295.7795  # 1000 * 180 / pi
        j = [round(joints[i] * factor) for i in range(6)]
        self._sdk.MotionCtrl_2(0x01, 0x01, 100, 0x00)
        self._sdk.JointCtrl(*j)

        # 发送夹爪指令
        self._sdk.GripperCtrl(int(gripper * GRIPPER_TO_SDK), 1000, 0x01, 0)

        return action

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _enable_arm(self, sdk) -> None:
        timeout = 5.0
        start = time.time()
        while True:
            info = sdk.GetArmLowSpdInfoMsgs()
            enabled = all(getattr(info, f"motor_{i + 1}").foc_status.driver_enable_status for i in range(6))
            if enabled:
                break
            sdk.EnableArm(7)
            sdk.GripperCtrl(0, 1000, 0x01, 0)
            if time.time() - start > timeout:
                raise RuntimeError("Piper arm enable timeout. Check CAN connection.")
            time.sleep(1.0)
