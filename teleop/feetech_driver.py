# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import enum
import logging
import math
import time
import traceback
from copy import deepcopy

import numpy as np
import tqdm

import abc
from dataclasses import dataclass

import draccus


@dataclass
class MotorsBusConfig(draccus.ChoiceRegistry, abc.ABC):
    @property
    def type(self) -> str:
        return self.get_choice_name(self.__class__)


@MotorsBusConfig.register_subclass("dynamixel")
@dataclass
class DynamixelMotorsBusConfig(MotorsBusConfig):
    port: str
    motors: dict[str, tuple[int, str]]
    mock: bool = False


@MotorsBusConfig.register_subclass("feetech")
@dataclass
class FeetechMotorsBusConfig(MotorsBusConfig):
    port: str
    motors: dict[str, tuple[int, str]]
    mock: bool = False

class RobotDeviceNotConnectedError(Exception):
    """Exception raised when the robot device is not connected."""

    def __init__(
        self, message="This robot device is not connected. Try calling `robot_device.connect()` first."
    ):
        self.message = message
        super().__init__(self.message)


class RobotDeviceAlreadyConnectedError(Exception):
    """Exception raised when the robot device is already connected."""

    def __init__(
        self,
        message="This robot device is already connected. Try not calling `robot_device.connect()` twice.",
    ):
        self.message = message
        super().__init__(self.message)


# STS:0  SCS:1
PROTOCOL_VERSION = {
    "scs0009": 1,
    "sts3215": 0,
}
BAUDRATE = 1_000_000
TIMEOUT_MS = 1000

MAX_ID_RANGE = 252

# The following bounds define the lower and upper joints range (after calibration).
# For joints in degree (i.e. revolute joints), their nominal range is [-180, 180] degrees
# which corresponds to a half rotation on the left and half rotation on the right.
# Some joints might require higher range, so we allow up to [-270, 270] degrees until
# an error is raised.
# LOWER_BOUND_DEGREE = -270
# UPPER_BOUND_DEGREE = 270
LOWER_BOUND_DEGREE = -8
UPPER_BOUND_DEGREE = 55
# For joints in percentage (i.e. joints that move linearly like the prismatic joint of a gripper),
# their nominal range is [0, 100] %. For instance, for Aloha gripper, 0% is fully
# closed, and 100% is fully open. To account for slight calibration issue, we allow up to
# [-10, 110] until an error is raised.
LOWER_BOUND_LINEAR = -10
UPPER_BOUND_LINEAR = 110

HALF_TURN_DEGREE = 180


# See this link for STS3215 Memory Table:
# https://docs.google.com/spreadsheets/d/1GVs7W1VS1PqdhA1nW-abeyAHhTUxKUdR/edit?usp=sharing&ouid=116566590112741600240&rtpof=true&sd=true
# data_name: (address, size_byte)
SCS_SERIES_CONTROL_TABLE = {
    "Model": (3, 2),
    "ID": (5, 1),
    "Baud_Rate": (6, 1),
    "Return_Delay": (7, 1),
    "Response_Status_Level": (8, 1),
    "Min_Angle_Limit": (9, 2),
    "Max_Angle_Limit": (11, 2),
    "Max_Temperature_Limit": (13, 1),
    "Max_Voltage_Limit": (14, 1),
    "Min_Voltage_Limit": (15, 1),
    "Max_Torque_Limit": (16, 2),
    "Phase": (18, 1),
    "Unloading_Condition": (19, 1),
    "LED_Alarm_Condition": (20, 1),
    "P_Coefficient": (21, 1),
    "D_Coefficient": (22, 1),
    "I_Coefficient": (23, 1),
    "Minimum_Startup_Force": (24, 2),
    "CW_Dead_Zone": (26, 1),
    "CCW_Dead_Zone": (27, 1),
    "Protection_Current": (28, 2),
    "Angular_Resolution": (30, 1),
    "Offset": (31, 2),
    "Mode": (33, 1),
    "Protective_Torque": (34, 1),
    "Protection_Time": (35, 1),
    "Overload_Torque": (36, 1),
    "Speed_closed_loop_P_proportional_coefficient": (37, 1),
    "Over_Current_Protection_Time": (38, 1),
    "Velocity_closed_loop_I_integral_coefficient": (39, 1),
    "Torque_Enable": (40, 1),
    "Acceleration": (41, 1),
    "Goal_Position": (42, 2),
    "Goal_Time": (44, 2),
    "Goal_Speed": (46, 2),
    "Torque_Limit": (48, 2),
    "Lock": (55, 1),
    "Present_Position": (56, 2),
    "Present_Speed": (58, 2),
    "Present_Load": (60, 2),
    "Present_Voltage": (62, 1),
    "Present_Temperature": (63, 1),
    "Status": (65, 1),
    "Moving": (66, 1),
    "Present_Current": (69, 2),
    # Not in the Memory Table
    "Maximum_Acceleration": (85, 2),
}

SCS_SERIES_BAUDRATE_TABLE = {
    0: 1_000_000,
    1: 500_000,
    2: 250_000,
    3: 128_000,
    4: 115_200,
    5: 57_600,
    6: 38_400,
    7: 19_200,
}

CALIBRATION_REQUIRED = ["Goal_Position", "Present_Position"]
CONVERT_UINT32_TO_INT32_REQUIRED = ["Goal_Position", "Present_Position"]


MODEL_CONTROL_TABLE = {
    "scs0009": SCS_SERIES_CONTROL_TABLE,
    "scs_series": SCS_SERIES_CONTROL_TABLE,
    "sts3215": SCS_SERIES_CONTROL_TABLE,
}

MODEL_RESOLUTION = {
    "scs0009": 1228.8,
    "scs_series": 4096,
    "sts3215": 4096,
}

MODEL_BAUDRATE_TABLE = {
    "scs0009": SCS_SERIES_BAUDRATE_TABLE,
    "scs_series": SCS_SERIES_BAUDRATE_TABLE,
    "sts3215": SCS_SERIES_BAUDRATE_TABLE,
}

# High number of retries is needed for feetech compared to dynamixel motors.
NUM_READ_RETRY = 200
NUM_WRITE_RETRY = 20


def convert_degrees_to_steps(degrees: float | np.ndarray, models: str | list[str]) -> np.ndarray:
    """This function converts the degree range to the step range for indicating motors rotation.
    It assumes a motor achieves a full rotation by going from -180 degree position to +180.
    The motor resolution (e.g. 4096) corresponds to the number of steps needed to achieve a full rotation.
    """
    resolutions = [MODEL_RESOLUTION[model] for model in models]
    steps = degrees / 180 * np.array(resolutions) / 2
    steps = steps.astype(int)
    return steps


def convert_to_bytes(value, bytes):
    import scservo_sdk as scs

    # Note: No need to convert back into unsigned int, since this byte preprocessing
    # already handles it for us.
    if bytes == 1:
        data = [
            scs.SCS_LOBYTE(scs.SCS_LOWORD(value)),
        ]
    elif bytes == 2:
        data = [
            scs.SCS_LOBYTE(scs.SCS_LOWORD(value)),
            scs.SCS_HIBYTE(scs.SCS_LOWORD(value)),
        ]
    elif bytes == 4:
        data = [
            scs.SCS_LOBYTE(scs.SCS_LOWORD(value)),
            scs.SCS_HIBYTE(scs.SCS_LOWORD(value)),
            scs.SCS_LOBYTE(scs.SCS_HIWORD(value)),
            scs.SCS_HIBYTE(scs.SCS_HIWORD(value)),
        ]
    else:
        raise NotImplementedError(
            f"Value of the number of bytes to be sent is expected to be in [1, 2, 4], but "
            f"{bytes} is provided instead."
        )
    return data


def get_group_sync_key(data_name, motor_names):
    group_key = f"{data_name}_" + "_".join(motor_names)
    return group_key


def get_result_name(fn_name, data_name, motor_names):
    group_key = get_group_sync_key(data_name, motor_names)
    rslt_name = f"{fn_name}_{group_key}"
    return rslt_name


def get_queue_name(fn_name, data_name, motor_names):
    group_key = get_group_sync_key(data_name, motor_names)
    queue_name = f"{fn_name}_{group_key}"
    return queue_name


def get_log_name(var_name, fn_name, data_name, motor_names):
    group_key = get_group_sync_key(data_name, motor_names)
    log_name = f"{var_name}_{fn_name}_{group_key}"
    return log_name


def assert_same_address(model_ctrl_table, motor_models, data_name):
    all_addr = []
    all_bytes = []
    for model in motor_models:
        print("data_name:",data_name)
        addr, bytes = model_ctrl_table[model][data_name]
        all_addr.append(addr)
        all_bytes.append(bytes)

    if len(set(all_addr)) != 1:
        raise NotImplementedError(
            f"At least two motor models use a different address for `data_name`='{data_name}' ({list(zip(motor_models, all_addr, strict=False))}). Contact a LeRobot maintainer."
        )

    if len(set(all_bytes)) != 1:
        raise NotImplementedError(
            f"At least two motor models use a different bytes representation for `data_name`='{data_name}' ({list(zip(motor_models, all_bytes, strict=False))}). Contact a LeRobot maintainer."
        )


class TorqueMode(enum.Enum):
    ENABLED = 1
    DISABLED = 0


class DriveMode(enum.Enum):
    NON_INVERTED = 0
    INVERTED = 1


class CalibrationMode(enum.Enum):
    # Joints with rotational motions are expressed in degrees in nominal range of [-180, 180]
    DEGREE = 0
    # Joints with linear motions (like gripper of Aloha) are expressed in nominal range of [0, 100]
    LINEAR = 1


class JointOutOfRangeError(Exception):
    def __init__(self, message="Joint is out of range"):
        self.message = message
        super().__init__(self.message)


class FeetechMotorsBus:
    """
    The FeetechMotorsBus class allows to efficiently read and write to the attached motors. It relies on
    the python feetech sdk to communicate with the motors. For more info, see the [feetech SDK Documentation](https://emanual.robotis.com/docs/en/software/feetech/feetech_sdk/sample_code/python_read_write_protocol_2_0/#python-read-write-protocol-20).

    A FeetechMotorsBus instance requires a port (e.g. `FeetechMotorsBus(port="/dev/tty.usbmodem575E0031751"`)).
    To find the port, you can run our utility script:
    ```bash
    python lerobot/scripts/find_motors_bus_port.py
    >>> Finding all available ports for the MotorsBus.
    >>> ['/dev/tty.usbmodem575E0032081', '/dev/tty.usbmodem575E0031751']
    >>> Remove the usb cable from your FeetechMotorsBus and press Enter when done.
    >>> The port of this FeetechMotorsBus is /dev/tty.usbmodem575E0031751.
    >>> Reconnect the usb cable.
    ```

    Example of usage for 1 motor connected to the bus:
    ```python
    motor_name = "gripper"
    motor_index = 6
    motor_model = "sts3215"

    config = FeetechMotorsBusConfig(
        port="/dev/tty.usbmodem575E0031751",
        motors={motor_name: (motor_index, motor_model)},
    )
    motors_bus = FeetechMotorsBus(config)
    motors_bus.connect()

    position = motors_bus.read("Present_Position")

    # move from a few motor steps as an example
    few_steps = 30
    motors_bus.write("Goal_Position", position + few_steps)

    # when done, consider disconnecting
    motors_bus.disconnect()
    ```
    """

    def __init__(
        self,
        config: FeetechMotorsBusConfig,
    ):
        self.port = config.port
        self.motors = config.motors

        self.model_ctrl_table = deepcopy(MODEL_CONTROL_TABLE)
        self.model_resolution = deepcopy(MODEL_RESOLUTION)

        self.port_handler = None
        self.packet_handler = None
        self.calibration = None
        self.is_connected = False
        self.group_readers = {}
        self.group_writers = {}
        self.logs = {}

        self.track_positions = {}

    def connect(self):
        if self.is_connected:
            raise RobotDeviceAlreadyConnectedError(
                f"FeetechMotorsBus({self.port}) is already connected. Do not call `motors_bus.connect()` twice."
            )

        import scservo_sdk as scs

        self.port_handler = scs.PortHandler(self.port)
        self.packet_handler = scs.PacketHandler(PROTOCOL_VERSION["sts3215"])

        try:
            if not self.port_handler.openPort():
                raise OSError(f"Failed to open port '{self.port}'.")
        except Exception:
            traceback.print_exc()
            print(
                "\nTry running `python lerobot/scripts/find_motors_bus_port.py` to make sure you are using the correct port.\n"
            )
            raise

        # Allow to read and write
        self.is_connected = True

        self.port_handler.setPacketTimeoutMillis(TIMEOUT_MS)

    def reconnect(self):
        import scservo_sdk as scs

        self.port_handler = scs.PortHandler(self.port)
        self.packet_handler = scs.PacketHandler(PROTOCOL_VERSION["sts3215"])

        if not self.port_handler.openPort():
            raise OSError(f"Failed to open port '{self.port}'.")

        self.is_connected = True

    def are_motors_configured(self):
        # Only check the motor indices and not baudrate, since if the motor baudrates are incorrect,
        # a ConnectionError will be raised anyway.
        try:
            return (self.motor_indices == self.read("ID")).all()
        except ConnectionError as e:
            print(e)
            return False

    def find_motor_indices(self, possible_ids=None, num_retry=2):
        if possible_ids is None:
            possible_ids = range(MAX_ID_RANGE)

        indices = []
        for idx in tqdm.tqdm(possible_ids):
            try:
                present_idx = self.read_with_motor_ids(self.motor_models, [idx], "ID", num_retry=num_retry)[0]
            except ConnectionError:
                continue

            if idx != present_idx:
                # sanity check
                raise OSError(
                    "Motor index used to communicate through the bus is not the same as the one present in the motor memory. The motor memory might be damaged."
                )
            indices.append(idx)

        return indices

    def set_bus_baudrate(self, baudrate):
        present_bus_baudrate = self.port_handler.getBaudRate()
        if present_bus_baudrate != baudrate:
            print(f"Setting bus baud rate to {baudrate}. Previously {present_bus_baudrate}.")
            self.port_handler.setBaudRate(baudrate)

            if self.port_handler.getBaudRate() != baudrate:
                raise OSError("Failed to write bus baud rate.")

    @property
    def motor_names(self) -> list[str]:
        return list(self.motors.keys())

    @property
    def motor_models(self) -> list[str]:
        return [model for _, model in self.motors.values()]

    @property
    def motor_indices(self) -> list[int]:
        return [idx for idx, _ in self.motors.values()]

    def read_with_motor_ids(self, motor_models, motor_ids, data_name, num_retry=NUM_READ_RETRY):
        import scservo_sdk as scs

        if not isinstance(motor_ids, list):
            motor_ids = [motor_ids]

        self.port_handler.ser.reset_output_buffer()
        self.port_handler.ser.reset_input_buffer()
        assert_same_address(self.model_ctrl_table, self.motor_models, data_name)
        values = []
        print("self.motor_models:",self.motor_models)
        print("motor_ids:",motor_ids)
        print("data_name:",data_name)
        for idx in motor_ids:
            motor_id=idx+1
            addr, bytes = self.model_ctrl_table[self.motor_models[idx]][data_name]
            print("addr:", addr, "bytes:", bytes)
            self.packet_handler = scs.PacketHandler(PROTOCOL_VERSION[self.motor_models[idx]])
            if bytes == 1:
                value, comm, error = self.packet_handler.read1ByteTxRx(self.port_handler, motor_id, addr)
            elif bytes == 2:
                value, comm, error = self.packet_handler.read2ByteTxRx(self.port_handler, motor_id, addr)
            if comm != scs.COMM_SUCCESS or error != 0:
                raise ConnectionError(
                    f"Read failed on port {self.port} for idx {motor_id}: comm={comm} ({self.packet_handler.getTxRxResult(comm)}), "
                    f"error={error} ({self.packet_handler.getRxPacketError(error)})"
                )
            values.append(value)
            print("value:", value)

        values = np.array(values)

        if data_name in CONVERT_UINT32_TO_INT32_REQUIRED:
            values = values.astype(np.int32)

        return values
    

    # def read(self, data_name, motor_names: str | list[str] | None = None):
    #     import scservo_sdk as scs

    #     if not self.is_connected:
    #         raise RobotDeviceNotConnectedError(
    #             f"FeetechMotorsBus({self.port}) is not connected. You need to run `motors_bus.connect()`."
    #         )

    #     start_time = time.perf_counter()

    #     if motor_names is None:
    #         motor_names = self.motor_names

    #     if isinstance(motor_names, str):
    #         motor_names = [motor_names]

    #     motor_ids = []
    #     models = []
    #     sts_motor_names = []
    #     sts_motor_ids = []
    #     motor_id2model = {}
    #     for name in motor_names:
    #         motor_idx, model = self.motors[name]
    #         motor_ids.append(motor_idx)
    #         models.append(model)
    #         motor_id2model[motor_idx]=model
    #         if model == 'sts3215':
    #             sts_motor_names.append(name)
    #             sts_motor_ids.append(motor_idx)

    #     assert_same_address(self.model_ctrl_table, models, data_name)
    #     addr, bytes = self.model_ctrl_table[model][data_name]
    #     group_key = get_group_sync_key(data_name, sts_motor_names)
    #     # print("data_name:",data_name)
    #     if data_name not in self.group_readers:
    #         # Very Important to flush the buffer!
    #         self.port_handler.ser.reset_output_buffer()
    #         self.port_handler.ser.reset_input_buffer()

    #         # create new group reader
    #         self.group_readers[group_key] = scs.GroupSyncRead(
    #             self.port_handler, self.packet_handler, addr, bytes
    #         )
    #         for idx in sts_motor_ids:
    #             self.group_readers[group_key].addParam(idx)
    #     if 'sts3215' in self.motor_models:
    #         for _ in range(NUM_READ_RETRY):
    #             comm = self.group_readers[group_key].txRxPacket()
    #             if comm == scs.COMM_SUCCESS:
    #                 break

    #         if comm != scs.COMM_SUCCESS:
    #             raise ConnectionError(
    #                 f"Read failed due to communication error on port {self.port} for group_key {group_key}: "
    #                 f"{self.packet_handler.getTxRxResult(comm)}"
    #             )

    #     values = []
    #     for idx in motor_ids:
    #         self.packet_handler = scs.PacketHandler(PROTOCOL_VERSION[motor_id2model[idx]])
    #         if motor_id2model[idx] == 'scs0009':
    #             if bytes == 1:
    #                 value, comm, error = self.packet_handler.read1ByteTxRx(self.port_handler, idx, addr)
    #             elif bytes == 2:
    #                 value, comm, error = self.packet_handler.read2ByteTxRx(self.port_handler, idx, addr)
    #             if comm != scs.COMM_SUCCESS or error != 0:
    #                 raise ConnectionError(
    #                     f"Write failed on port {self.port} for idx {idx}: comm={comm} ({self.packet_handler.getTxRxResult(comm)}), "
    #                     f"error={error} ({self.packet_handler.getRxPacketError(error)})"
    #                 )
    #         else:
    #             value = self.group_readers[group_key].getData(idx, addr, bytes)
    #         values.append(value)

    #     values = np.array(values)

    #     # Convert to signed int to use range [-2048, 2048] for our motor positions.
    #     # print("before cali:", values)
    #     # time.sleep(0.5)  # wait for the motors to update their values
    #     if data_name in CONVERT_UINT32_TO_INT32_REQUIRED:
    #         values = values.astype(np.int32)

    #     if data_name in CALIBRATION_REQUIRED:
    #         values = self.avoid_rotation_reset(values, motor_names, data_name)

    #     if data_name in CALIBRATION_REQUIRED and self.calibration is not None:
    #         values = self.apply_calibration_autocorrect(values, motor_names)

    #     # log the number of seconds it took to read the data from the motors
    #     delta_ts_name = get_log_name("delta_timestamp_s", "read", data_name, motor_names)
    #     self.logs[delta_ts_name] = time.perf_counter() - start_time

    #     # log the utc time at which the data was received
    #     ts_utc_name = get_log_name("timestamp_utc", "read", data_name, motor_names)
    #     self.logs[ts_utc_name] = capture_timestamp_utc()

    #     return values

    def write_with_motor_ids(self, motor_models, motor_ids, data_name, values, num_retry=NUM_WRITE_RETRY):
        import scservo_sdk as scs

        if not isinstance(motor_ids, list):
            motor_ids = [motor_ids]
        if not isinstance(values, list):
            values = [values]

        assert_same_address(self.model_ctrl_table, motor_models, data_name)
        addr, bytes = self.model_ctrl_table[motor_models[0]][data_name]
        group = scs.GroupSyncWrite(self.port_handler, self.packet_handler, addr, bytes)
        for idx, value in zip(motor_ids, values, strict=True):
            data = convert_to_bytes(value, bytes)
            group.addParam(idx, data)

        for _ in range(num_retry):
            comm = group.txPacket()
            if comm == scs.COMM_SUCCESS:
                break

        if comm != scs.COMM_SUCCESS:
            raise ConnectionError(
                f"Write failed due to communication error on port {self.port_handler.port_name} for indices {motor_ids}: "
                f"{self.packet_handler.getTxRxResult(comm)}"
            )

    def write(self, data_name, values: int | float | np.ndarray, motor_names: str | list[str] | None = None):
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                f"FeetechMotorsBus({self.port}) is not connected. You need to run `motors_bus.connect()`."
            )

        start_time = time.perf_counter()

        import scservo_sdk as scs

        if motor_names is None:
            motor_names = self.motor_names

        if isinstance(motor_names, str):
            motor_names = [motor_names]

        if isinstance(values, (int, float, np.integer)):
            values = [int(values)] * len(motor_names)

        values = np.array(values)

        motor_ids = []
        models = []
        motor_id2model = {}
        for name in motor_names:
            motor_idx, model = self.motors[name]
            motor_ids.append(motor_idx)
            models.append(model)
            motor_id2model[motor_idx]=model

        # if data_name == "Goal_Position":
        #     print("revert_calibration Goal_Position:", values)
        #     return 1

        values = values.tolist()

        assert_same_address(self.model_ctrl_table, models, data_name)
        addr, bytes = self.model_ctrl_table[model][data_name]
        group_key = get_group_sync_key(data_name, motor_names)

        init_group = data_name not in self.group_readers
        # print("addr:",addr)
        # print("bytes:",bytes)
        # print("self.port_handler.port_name:",self.port_handler.port_name)
        for idx, value in zip(motor_ids, values, strict=True):
            protocol_version = PROTOCOL_VERSION[motor_id2model[idx]]
            self.packet_handler = scs.PacketHandler(protocol_version)
            if bytes == 1:
                comm, error = self.packet_handler.write1ByteTxRx(self.port_handler, idx, addr, value)
            elif bytes == 2:
                comm, error = self.packet_handler.write2ByteTxRx(self.port_handler, idx, addr, value)
            if comm != scs.COMM_SUCCESS or error != 0:
                raise ConnectionError(
                    f"Write failed due to communication error on port {self.port} for idx {idx}: "
                    f"{self.packet_handler.getTxRxResult(comm)}"
                )

        # log the number of seconds it took to write the data to the motors
        delta_ts_name = get_log_name("delta_timestamp_s", "write", data_name, motor_names)
        self.logs[delta_ts_name] = time.perf_counter() - start_time

    def disconnect(self):
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                f"FeetechMotorsBus({self.port}) is not connected. Try running `motors_bus.connect()` first."
            )

        if self.port_handler is not None:
            self.port_handler.closePort()
            self.port_handler = None

        self.packet_handler = None
        self.group_readers = {}
        self.group_writers = {}
        self.is_connected = False

    def __del__(self):
        if getattr(self, "is_connected", False):
            self.disconnect()

from typing import Protocol

class MotorsBus(Protocol):
    def motor_names(self): ...
    def set_calibration(self): ...
    def apply_calibration(self): ...
    def revert_calibration(self): ...
    def read(self): ...
    def write(self): ...


def make_motors_buses_from_configs(motors_bus_configs: dict[str, MotorsBusConfig]) -> list[MotorsBus]:
    motors_buses = {}

    for key, cfg in motors_bus_configs.items():
        if cfg.type == "dynamixel":
            print("dynamixel motor not supported in this version.")
            return None

        elif cfg.type == "feetech":
            motors_buses[key] = FeetechMotorsBus(cfg)

        else:
            raise ValueError(f"The motor type '{cfg.type}' is not valid.")

    return motors_buses


def make_motors_bus(motor_type: str, **kwargs) -> MotorsBus:
    if motor_type == "dynamixel":
        print("dynamixel motor not supported in this version.")
        return None

    elif motor_type == "feetech":
        config = FeetechMotorsBusConfig(**kwargs)
        return FeetechMotorsBus(config)

    else:
        raise ValueError(f"The motor type '{motor_type}' is not valid.")

# ================= FeetechDriver: 兼容 snake_agent.py 的简易关节接口 =================
class FeetechDriver:
    def __init__(self, joint_ids, models, port="/dev/ttyUSB0", baudrate=1000000):
        # 支持混合 SCS/STS，motors 字典格式: {name: (id, model)}
        self.joint_ids = list(joint_ids)
        self.models = list(models)

        # Validate lengths match
        assert len(self.joint_ids) == len(self.models), (
            f"joint_ids length ({len(self.joint_ids)}) must match models length ({len(self.models)})"
        )

        self.motors = {f"m{i}": (jid, model) for i, (jid, model) in enumerate(zip(self.joint_ids, self.models))}
        config = FeetechMotorsBusConfig(port=port, motors=self.motors)
        self.bus = FeetechMotorsBus(config)

    def connect(self):
        self.bus.connect()

    def sync_write(self, joint_ids, values, address, size):
        # 逐个写入
        for jid, val in zip(joint_ids, values):
            # Find the model for this joint_id
            motor_model = self._get_model_for_joint_id(jid)
            self.bus.write_with_motor_ids([motor_model], [jid], self._addr_to_name(address), [int(val)])

    def set_torque_enabled(self, joint_ids, enabled):
        for jid in joint_ids:
            # Find the model for this joint_id
            motor_model = self._get_model_for_joint_id(jid)
            self.bus.write_with_motor_ids([motor_model], [jid], "Torque_Enable", [int(enabled)])

    def read_pos(self):
        # 读取所有关节 Present_Position，使用对应的motor model
        result = []
        for jid in self.joint_ids:
            # if jid < 2:
            #     continue
            motor_model = self._get_model_for_joint_id(jid)
            print("motor_model:", motor_model)
            print("jid:", jid)
            pos = self.bus.read_with_motor_ids([motor_model], [jid], "Present_Position")[0]
            result.append(pos)
        return np.array(result)

    def read_vel(self):
        result = []
        for jid in self.joint_ids:
            motor_model = self._get_model_for_joint_id(jid)
            vel = self.bus.read_with_motor_ids([motor_model], [jid], "Present_Speed")[0]
            result.append(vel)
        return np.array(result)

    def write_desired_pos(self, joint_ids, positions):
        # 逐个写入目标位置
        for jid, pos in zip(joint_ids, positions):
            motor_model = self._get_model_for_joint_id(jid)
            self.bus.write_with_motor_ids([motor_model], [jid], "Goal_Position", [int(pos)])

    def _addr_to_name(self, address):
        # 地址到寄存器名的简单映射（常用）
        addr_map = {
            40: "Torque_Enable",
            42: "Goal_Position",
            56: "Present_Position",
            58: "Present_Speed",
        }
        return addr_map.get(address, address)

    def _get_model_for_joint_id(self, joint_id: int) -> str:
        """Get the model for a specific joint ID"""
        try:
            joint_index = self.joint_ids.index(joint_id)
            return self.models[joint_index]
        except ValueError:
            raise ValueError(f"Joint ID {joint_id} not found in joint_ids list: {self.joint_ids}")

    def get_motor_info(self) -> dict:
        """Get information about all motors"""
        return {
            f"joint_{jid}": {"id": jid, "model": model}
            for jid, model in zip(self.joint_ids, self.models)
        }