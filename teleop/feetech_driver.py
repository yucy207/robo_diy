import atexit
import logging
import math
import time
from typing import Optional, Sequence, Union, Tuple

import numpy as np
import serial

# 这里参考 scs0009 内存表（SCS_SERIES_CONTROL_TABLE）：
# Torque_Enable: (40, 1)
# Goal_Position: (42, 2)
ADDR_TORQUE_ENABLE = 0x40       # 来自内存表：Torque_Enable
ADDR_GOAL_POSITION = 0x42       # 来自内存表：Goal_Position
LEN_GOAL_POSITION = 2           # 2字节数据

# 命令号定义：这里用自定义命令号模拟写入操作
CMD_SET_GOAL_POSITION = 0x03
CMD_SET_TORQUE_ENABLE = 0x04

class FeetechDriver:
    """
    Feetech 驱动，参考 lerobot 的 FeetechMotorsBus 封装以及 scs0009 内存表，
    实现 set_torque_enabled, sync_write, write_desired_pos, write_byte 等接口。
    上层（例如 snake_agent）可直接替换原 DynamixelDriver 使用此驱动。
    """
    def __init__(self,
                 motor_ids: Sequence[int],
                 port: str = "/dev/ttyUSB0",
                 baudrate: int = 115200,
                 timeout: float = 0.1):
        self.motor_ids = list(motor_ids)
        self.port_name = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.serial = None

    def connect(self):
        self.serial = serial.Serial(port=self.port_name,
                                    baudrate=self.baudrate,
                                    timeout=self.timeout)
        logging.info("Connected to Feetech servos on port %s", self.port_name)

    @property
    def is_connected(self) -> bool:
        return self.serial is not None and self.serial.is_open

    def disconnect(self):
        if self.is_connected:
            # 关闭前禁用所有舵机力矩（确保安全）
            self.set_torque_enabled(self.motor_ids, False)
            self.serial.close()
            self.serial = None
            logging.info("Disconnected from Feetech servos on port %s", self.port_name)

    def check_connected(self):
        if not self.is_connected:
            raise OSError("Not connected. Please call connect() first.")

    def send_packet(self, packet: bytes):
        self.check_connected()
        self.serial.write(packet)
        # 为避免发送过快，简单 sleep 一段（可根据实际情况调整）
        time.sleep(0.005)

    def build_packet(self, motor_id: int, command: int, data: bytes) -> bytes:
        """
        根据协议构造数据包：
        包格式: [Header(0xFF,0xFF), ID, LENGTH, COMMAND, DATA..., CHECKSUM]
        checksum 为除 Header 外所有字节之和取反后保留低 8 位。
        """
        header = bytes([0xFF, 0xFF])
        id_byte = bytes([motor_id])
        length = 1 + len(data) + 1  # 包含 command, data, checksum
        length_byte = bytes([length])
        command_byte = bytes([command])
        packet_without_checksum = id_byte + length_byte + command_byte + data
        checksum = (~sum(packet_without_checksum)) & 0xFF
        checksum_byte = bytes([checksum])
        packet = header + packet_without_checksum + checksum_byte
        return packet

    def write_byte(self, motor_ids: Sequence[int], value: int, address: int) -> Sequence[int]:
        """
        写入单字节值到指定地址。
        此处根据 address 判断命令类型：
          - 如果 address==ADDR_TORQUE_ENABLE，则使用 CMD_SET_TORQUE_ENABLE，
            data 为单字节的目标值。
        返回写入失败的 motor id 列表。
        """
        self.check_connected()
        errored_ids = []
        for motor_id in motor_ids:
            if address == ADDR_TORQUE_ENABLE:
                command = CMD_SET_TORQUE_ENABLE
                data = bytes([value])
            else:
                logging.error("Unsupported address %s for write_byte", address)
                errored_ids.append(motor_id)
                continue
            packet = self.build_packet(motor_id, command, data)
            try:
                self.send_packet(packet)
            except Exception as e:
                logging.error("Error writing byte to motor id %s: %s", motor_id, e)
                errored_ids.append(motor_id)
        return errored_ids

    def set_torque_enabled(self,
                           motor_ids: Sequence[int],
                           enabled: bool,
                           retries: int = -1,
                           retry_interval: float = 0.25):
        """
        设置指定 motor_ids 的舵机是否使能力矩。
        """
        remaining_ids = list(motor_ids)
        while remaining_ids:
            remaining_ids = self.write_byte(remaining_ids, int(enabled), ADDR_TORQUE_ENABLE)
            if remaining_ids:
                logging.error("Could not set torque %s for IDs: %s",
                              'enabled' if enabled else 'disabled',
                              str(remaining_ids))
            if retries == 0:
                break
            time.sleep(retry_interval)
            retries -= 1

    def sync_write(self, motor_ids: Sequence[int],
                   values: Sequence[Union[int, float]],
                   address: int,
                   size: int):
        """
        对指定电机进行同步写入。
        Feetech 驱动通常不支持组写，本实现逐个发送包。
        参数 values 通常为目标位置等，若需要转换可在此处理。
        """
        self.check_connected()
        assert len(motor_ids) == len(values), "motor_ids and values length mismatch"
        for motor_id, value in zip(motor_ids, values):
            # 此处假设 value 是数字，将其转换为整数后编码为小端字节序
            data = int(value).to_bytes(size, byteorder='little', signed=False)
            packet = self.build_packet(motor_id, CMD_SET_GOAL_POSITION, data)
            self.send_packet(packet)

    def write_desired_pos(self, motor_ids: Sequence[int],
                          positions: np.ndarray):
        """
        写入舵机的目标位置。
        positions 单位为度（如需要转换，请在此扩展）。
        这里参考 "Goal_Position" 在 scs0009 中的地址和数据长度。
        """
        self.check_connected()
        assert len(motor_ids) == len(positions), "Mismatch in write_desired_pos"
        self.sync_write(motor_ids, positions, ADDR_GOAL_POSITION, LEN_GOAL_POSITION)

    # 其他辅助接口（例如 check_connected_and_raise 等）可按需扩展

    def __enter__(self):
        if not self.is_connected:
            self.connect()
        return self

    def __exit__(self, *args):
        self.disconnect()

    def __del__(self):
        try:
            self.disconnect()
        except Exception:
            pass

def feetech_cleanup_handler():
    logging.info("Cleanup Feetech driver resources.")

atexit.register(feetech_cleanup_handler)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--motors', required=True,
                        help='Comma-separated list of motor IDs, e.g., 1,2,3')
    parser.add_argument('-p', '--port', default='/dev/ttyUSB0', help='Serial port device')
    parser.add_argument('-b', '--baud', type=int, default=115200, help='Baudrate')
    args = parser.parse_args()

    motor_ids = [int(x) for x in args.motors.split(",")]
    driver = FeetechDriver(motor_ids, port=args.port, baudrate=args.baud)
    driver.connect()
    try:
        logging.info("Setting torque enabled for motors %s", motor_ids)
        driver.set_torque_enabled(motor_ids, True)
        # 示例：将所有舵机设置到 90 度
        target_positions = np.full((len(motor_ids),), 90.0)
        driver.write_desired_pos(motor_ids, target_positions)
        logging.info("Sent target positions: %s", target_positions.tolist())
    finally:
        driver.disconnect()