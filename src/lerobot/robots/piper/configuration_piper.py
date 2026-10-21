from dataclasses import dataclass, field

from ..config import RobotConfig


@RobotConfig.register_subclass("piper")
@dataclass
class PiperConfig(RobotConfig):
    # CAN 接口名称，对应 Piper 机械臂的 USB-CAN 适配器
    can_port: str = "can0"

    # 两个 RealSense 相机的序列号
    camera_serials: dict[str, str] = field(default_factory=dict)

    # 图像分辨率
    image_width: int = 640
    image_height: int = 480

    # 采集频率（Hz）
    fps: int = 10
