import os
import sys
from os.path import join as pjoin
BASEPATH = os.path.dirname(__file__)
sys.path.insert(0, BASEPATH)
sys.path.insert(0, pjoin(BASEPATH, '..'))
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple, List
import numpy as np

from dynamixel_driver import DynamixelDriver
from feetech_driver import FeetechDriver

class FeetechRobot():

    def __init__(
        self,
        joint_ids: Sequence[int],
        joint_offsets: Optional[Sequence[float]] = None,
        joint_signs: Optional[Sequence[int]] = None,
        port: str = "/dev/ttyUSB0",
        baudrate: int = 1000000,
        start_joints: Optional[np.ndarray] = None,
        enable_torque: bool = False,
        models: Optional[Sequence[str]] = None,  
    ):
        self.kP = 600
        self.kI = 0
        self.kD = 200
        self.curr_lim = 550
        self._joint_ids = joint_ids

        self._models = list(models)
        self._resolution = []
        for model in self._models:
            if model == "scs0009":
                self._resolution.append(1228.8)
            elif model == "sts3215":
                self._resolution.append(4096)
            else:
                raise ValueError(f"Invalid model: {model}")

        if joint_offsets is None:
            self._joint_offsets = np.zeros(len(joint_ids))
        else:
            self._joint_offsets = np.array(joint_offsets)
        if joint_signs is None:
            self._joint_signs = np.ones(len(joint_ids))
        else:
            self._joint_signs = np.array(joint_signs)
        assert len(self._joint_ids) == len(self._joint_offsets)
        assert len(self._joint_ids) == len(self._joint_signs)
        assert len(self._joint_ids) == len(self._models)
        assert len(self._joint_ids) == len(self._resolution)
        assert np.all(np.abs(self._joint_signs) == 1)
        self._driver = FeetechDriver(joint_ids, port=port, baudrate=baudrate, models=self._models)
        self._driver.connect()
        self._driver.sync_write(joint_ids, np.ones(len(joint_ids)) * 5, 11, 1)
        self._driver.set_torque_enabled(joint_ids, enable_torque)
        self._driver.sync_write(joint_ids, np.ones(len(joint_ids)) * self.kP, 84, 2)
        self._driver.sync_write(joint_ids, np.ones(len(joint_ids)) * self.kI, 82, 2)
        self._driver.sync_write(joint_ids, np.ones(len(joint_ids)) * self.kD, 80, 2)
        self._driver.sync_write(joint_ids, np.ones(len(joint_ids)) * self.curr_lim, 102, 2)
        self._torque_on = True
        self._last_pos = None
        self._alpha = 0.99
        if start_joints is not None:
            new_joint_offsets = []
            current_joints = self.get_joint_pos()
            assert current_joints.shape == start_joints.shape
            for idx, (c_joint, s_joint, joint_offset) in enumerate(
                zip(current_joints, start_joints, self._joint_offsets)
            ):
                new_joint_offsets.append(
                    np.pi * 2 * np.round((-s_joint + c_joint) / (2 * np.pi)) * self._joint_signs[idx] + joint_offset
                )
            self._joint_offsets = np.array(new_joint_offsets)

    def num_dofs(self) -> int:
        return len(self._joint_ids)

    def get_motor_resolution(self, motor_index: int) -> int:
        """Get resolution for a specific motor by index"""
        if motor_index < 0 or motor_index >= len(self._resolution):
            raise IndexError(f"Motor index {motor_index} out of range. Valid range: 0-{len(self._resolution)-1}")
        return self._resolution[motor_index]

    def get_all_resolutions(self) -> List[int]:
        """Get list of all motor resolutions"""
        return self._resolution.copy()

    def _steps_to_rad(self, steps):
        # Handle different resolutions for each motor
        # Array of steps - handle per-motor resolution
        steps_array = np.array(steps)
        result = []
        for i, step in enumerate(steps_array):
            resolution = self._resolution[i]
            rad = (step - resolution / 2) * (2 * np.pi / resolution)
            result.append(rad)
        return np.array(result)

    def _rad_to_steps(self, rad):
        # Handle different resolutions for each motor
        # Array of radians - handle per-motor resolution
        rad_array = np.array(rad)
        result = []
        for i, r in enumerate(rad_array):
            resolution = self._resolution[i]
            step = np.round((r * (resolution / (2 * np.pi))) + resolution / 2).astype(int)
            result.append(step)
        return np.array(result)

    def read_pos(self) -> np.ndarray:
        # 读取原始步进值，转为弧度
        steps = self._driver.read_pos()
        return self._steps_to_rad(steps)

    def read_vel(self) -> np.ndarray:
        # 读取原始速度（如有需要可补充转换）
        return self._driver.read_vel()

    def write_desired_pos(self, joint_ids, pos_rad):
        # 弧度转步进后写入
        steps = self._rad_to_steps(pos_rad)
        self._driver.write_desired_pos(joint_ids, steps)

    def get_joint_pos(self) -> np.ndarray:
        pos = (self.read_pos() - self._joint_offsets) * self._joint_signs
        assert len(pos) == self.num_dofs()
        if self._last_pos is None:
            self._last_pos = pos
        else:
            pos = self._last_pos * (1 - self._alpha) + pos * self._alpha
            self._last_pos = pos
        return pos

    def get_joint_vel(self) -> np.ndarray:
        return self.read_vel() * self._joint_signs

    def command_joint_state(self, joint_state: np.ndarray) -> None:
        self.write_desired_pos(self._joint_ids, joint_state + self._joint_offsets)

    def set_torque_mode(self, mode: bool):
        if mode == self._torque_on:
            return
        self._driver.set_torque_enabled(self._joint_ids, mode)
        self._torque_on = mode

    def get_observations(self) -> Dict[str, np.ndarray]:
        return {"joint_pos": self.get_joint_pos(), "joint_vel": self.get_joint_vel()}

@dataclass
class FeetechRobotConfig:
    joint_ids: Sequence[int]
    joint_offsets: Sequence[float]
    joint_signs: Sequence[int]

    models: Sequence[str]
    baudrate: int = 1000000

    def __post_init__(self):
        assert len(self.joint_ids) == len(self.joint_offsets), (
            f"joint_ids: {len(self.joint_ids)}, joint_offsets: {len(self.joint_offsets)}"
        )
        assert len(self.joint_ids) == len(self.joint_signs), (
            f"joint_ids: {len(self.joint_ids)}, joint_signs: {len(self.joint_signs)}"
        )
        # Optional: enforce that signs are ±1
        assert all(abs(s) == 1 for s in self.joint_signs), f"joint_signs: {self.joint_signs}"

        # Validate motor models
        valid_models = {"scs0009", "sts3215"}
        assert all(model in valid_models for model in self.models), (
            f"Invalid models found. Valid: {valid_models}, Got: {self.models}"
        )

    def make_robot(
        self,
        port: str = "/dev/ttyUSB0",
        start_joints: Optional[np.ndarray] = None,
        enable_torque: bool = False,
    ) -> FeetechRobot:
        return FeetechRobot(
            joint_ids=self.joint_ids,
            joint_offsets=list(self.joint_offsets),
            joint_signs=list(self.joint_signs),
            port=port,
            baudrate=self.baudrate,
            start_joints=start_joints,
            enable_torque=enable_torque,
            models=list(self.models),
        )

class DynamixelRobot():

    def __init__(
        self,
        joint_ids: Sequence[int],
        joint_offsets: Optional[Sequence[float]] = None,
        joint_signs: Optional[Sequence[int]] = None,
        port: str = "/dev/ttyUSB0",
        baudrate: int = 4000000,
        start_joints: Optional[np.ndarray] = None,
        enable_torque: bool = False,
    ):
        self.kP = 600   #600
        self.kI = 0   #0
        self.kD = 200   #200
        self.curr_lim = 550
        self._joint_ids = joint_ids

        if joint_offsets is None:
            self._joint_offsets = np.zeros(len(joint_ids))
        else:
            self._joint_offsets = np.array(joint_offsets)

        if joint_signs is None:
            self._joint_signs = np.ones(len(joint_ids))
        else:
            self._joint_signs = np.array(joint_signs)

        assert len(self._joint_ids) == len(self._joint_offsets), (
            f"joint_ids: {len(self._joint_ids)}, "
            f"joint_offsets: {len(self._joint_offsets)}"
        )
        assert len(self._joint_ids) == len(self._joint_signs), (
            f"joint_ids: {len(self._joint_ids)}, "
            f"joint_signs: {len(self._joint_signs)}"
        )
        assert np.all(
            np.abs(self._joint_signs) == 1
        ), f"joint_signs: {self._joint_signs}"

        self._driver = DynamixelDriver(joint_ids, port=port, baudrate=baudrate)
        self._driver.connect()
        self._driver.sync_write(joint_ids, np.ones(len(joint_ids)) * 5, 11, 1)
        self._driver.set_torque_enabled(joint_ids, enable_torque)
        # self._driver.set_torque_enabled(joint_ids, False)
        self._driver.sync_write(joint_ids, np.ones(len(joint_ids)) * self.kP, 84, 2)
        self._driver.sync_write(joint_ids, np.ones(len(joint_ids)) * self.kI, 82, 2)
        self._driver.sync_write(joint_ids, np.ones(len(joint_ids)) * self.kD, 80, 2)
        self._driver.sync_write(joint_ids, np.ones(len(joint_ids)) * self.curr_lim, 102, 2)
        self._torque_on = True
        self._last_pos = None
        self._alpha = 0.99

        if start_joints is not None:
            # loop through all joints and add +- 2pi to the joint offsets to get the closest to start joints
            new_joint_offsets = []
            current_joints = self.get_joint_pos()
            assert current_joints.shape == start_joints.shape
            for idx, (c_joint, s_joint, joint_offset) in enumerate(
                zip(current_joints, start_joints, self._joint_offsets)
            ):
                new_joint_offsets.append(
                    np.pi
                    * 2
                    * np.round((-s_joint + c_joint) / (2 * np.pi))
                    * self._joint_signs[idx]
                    + joint_offset
                )
            self._joint_offsets = np.array(new_joint_offsets)

    def num_dofs(self) -> int:
        return len(self._joint_ids)

    def get_joint_pos(self) -> np.ndarray:
        pos = (self._driver.read_pos() - self._joint_offsets) * self._joint_signs
        assert len(pos) == self.num_dofs()

        if self._last_pos is None:
            self._last_pos = pos
        else:
            # exponential smoothing
            pos = self._last_pos * (1 - self._alpha) + pos * self._alpha
            self._last_pos = pos

        return pos
    
    def get_joint_vel(self) -> np.ndarray:
        return self._driver.read_vel() * self._joint_signs

    def command_joint_state(self, joint_state: np.ndarray) -> None:
        self._driver.write_desired_pos(self._joint_ids, joint_state + self._joint_offsets)

    def set_torque_mode(self, mode: bool):
        if mode == self._torque_on:
            return
        self._driver.set_torque_enabled(self._joint_ids, mode)
        self._torque_on = mode

    def get_observations(self) -> Dict[str, np.ndarray]:
        return {"joint_pos": self.get_joint_pos(), "joint_vel": self.get_joint_vel()}
    

@dataclass
class DynamixelRobotConfig:
    joint_ids: Sequence[int]
    """The joint ids of GELLO (not including the gripper). Usually (1, 2, 3 ...)."""

    joint_offsets: Sequence[float]
    """The joint offsets of GELLO. There needs to be a joint offset for each joint_id and should be a multiple of pi/2."""

    joint_signs: Sequence[int]
    """The joint signs of GELLO. There needs to be a joint sign for each joint_id and should be either 1 or -1.

    This will be different for each arm design. Refernce the examples below for the correct signs for your robot.
    """

    def __post_init__(self):
        assert len(self.joint_ids) == len(self.joint_offsets)
        assert len(self.joint_ids) == len(self.joint_signs)

    def make_robot(
        self, port: str = "/dev/ttyUSB0", start_joints: Optional[np.ndarray] = None, enable_torque: bool = False
    ) -> DynamixelRobot:
        return DynamixelRobot(
            joint_ids=self.joint_ids,
            joint_offsets=list(self.joint_offsets),
            joint_signs=list(self.joint_signs),
            port=port,
            start_joints=start_joints,
            enable_torque=enable_torque
        )
    
class Rate:
    def __init__(self, rate: float):
        self.last = time.time()
        self.rate = rate

    def sleep(self) -> None:
        while self.last + 1.0 / self.rate > time.time():
            time.sleep(0.0001)
        self.last = time.time()

class SnakeAgent():
    def __init__(
            self,
            port: str,
            dynamixel_config: Optional[DynamixelRobotConfig] = None,
            enable_torque: bool = False,
            start_joints: Optional[np.ndarray] = None,
            control_rate_hz: float = 100.0,
            camera_dict: Optional[Dict[str, Any]] = None,
            camera_server: Optional[Any] = None,
    ) -> None:
        self._robot = dynamixel_config.make_robot(
            port=port, start_joints=start_joints, enable_torque=enable_torque
        )
        self._rate = Rate(control_rate_hz)
        self._camera_dict = {} if camera_dict is None else camera_dict
        self._camera_server = camera_server

    def get_obs(self) -> Dict[str, Any]:
        observations = {}
        for name, camera in self._camera_dict.items(): 
            image, depth = camera.read()
            observations[f"{name}_rgb"] = image
            if camera.enable_depth:
                observations[f"{name}_depth"] = depth
        if self._camera_server is not None:
            imgs, top_img, side_img = self._camera_server.get_data()
            observations["usb_cam"] = imgs
            if top_img is not None:
                observations["top_cam"] = top_img
            if side_img is not None:
                observations["side_cam"] = side_img

        robot_obs = self._robot.get_observations()
        observations["joint_pos"] = robot_obs["joint_pos"]
        observations["joint_vel"] = robot_obs["joint_vel"]
        return observations

    def get_act(self) -> np.ndarray:
        return self._robot.get_joint_pos()
    
    def set_act(self, joints: np.ndarray) -> Dict[str, Any]:
        self._robot.command_joint_state(joints)
        self._rate.sleep()
        return self.get_obs()

