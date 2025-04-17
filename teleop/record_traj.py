import datetime
import time
from dataclasses import dataclass
import pickle
from pathlib import Path
from typing import Dict
import numpy as np
import tyro
from snake_agent import DynamixelRobotConfig
from snake_agent import SnakeAgent

@dataclass  
class Args:
    leader_port: str = "/dev/ttyUSB0"
    folder: str = "snake_data"
    num_joints: int = 5

def print_color(*args, color=None, attrs=(), **kwargs):
    import termcolor

    if len(args) > 0:
        args = tuple(termcolor.colored(arg, color=color, attrs=attrs) for arg in args)
    print(*args, **kwargs)

def save_frame(
        folder: Path,
        timestamp: datetime.datetime,
        action: np.ndarray,
) -> None:
    folder.mkdir(exist_ok=True, parents=True)
    recorded_file = folder / (timestamp.isoformat() + ".pkl")
    with open(recorded_file, "wb") as f:
        pickle.dump(action, f)

def main(args):
    joint_ids = list(range(args.num_joints))
    config = DynamixelRobotConfig(joint_ids=joint_ids, joint_offsets=np.zeros(args.num_joints), joint_signs=np.ones(args.num_joints))
    leader = SnakeAgent(port=args.leader_port, dynamixel_config=config, enable_torque=False)

    start_time = time.time()
    save_path = (
        Path(args.folder).expanduser()
        / datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    )
    while True:
        num = time.time() - start_time
        message = f"\rTime passed: {round(num, 2)}          "
        print_color(
            message,
            color="white",
            attrs=("bold",),
            end="",
            flush=True,
        )
        action = leader.get_act()
        dt = datetime.datetime.now()
        save_frame(save_path, dt, action)
        time.sleep(0.01)

if __name__ == '__main__':
    main(tyro.cli(Args))