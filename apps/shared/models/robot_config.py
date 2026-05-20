from typing import List
from pypot.dynamixel.io import Dxl320IO

# Change this to the port of your robot
PORT = "COM15"

BAUDRATE = 1000000


class RobotConfig:

    def __init__(self, port: str = PORT, baudrate: int = BAUDRATE):
        self.port = port
        self.baudrate = baudrate
        self.config = {
            "controllers": {
                "my_dxl_controller": {
                    "sync_read": False,
                    "attached_motors": ["tower", "body", "head"],
                    "port": self.port,
                    "baudrate": self.baudrate,
                    "protocol": 2,
                }
            },
            "motorgroups": {
                "tower": ["tower_1", "tower_2", "tower_3"],
                "body": ["base"],
                "head": ["ears"],
            },
            "motors": {
                "tower_1": {
                    "orientation": "direct",
                    "type": "XL-320",
                    "id": 1,
                    "angle_limit": [-150.0, 150.0],
                    "offset": 0.0,
                },
                "tower_2": {
                    "orientation": "direct",
                    "type": "XL-320",
                    "id": 2,
                    "angle_limit": [-150.0, 150.0],
                    "offset": 0.0,
                },
                "tower_3": {
                    "orientation": "direct",
                    "type": "XL-320",
                    "id": 3,
                    "angle_limit": [-150.0, 150.0],
                    "offset": 0.0,
                },
                "base": {
                    "orientation": "direct",
                    "type": "XL-320",
                    "id": 4,
                    "angle_limit": [-150.0, 150.0],
                    "offset": 0.0,
                },
                "ears": {
                    "orientation": "direct",
                    "type": "XL-320",
                    "id": 5,
                    "angle_limit": [50, 130.0],
                    "offset": 0.0,
                },
            },
        }
        self._update_config()

        self.motor_names = list(self.config["motors"].keys())
        self.motorgroup_names = list(self.config["motorgroups"].keys())

    def _update_config(self):
        scanned_ids = []
        try:
            dxl = Dxl320IO(self.port, self.baudrate)
            scanned_ids = dxl.scan(range(20))
        except Exception as e:
            print(f"Error scanning motors: {e}")

        if len(scanned_ids) == 0:
            print("No motors found!")
            return

        self.config["motors"] = self._return_motor_config(scanned_ids)
        self.config["motorgroups"] = self._return_motorgroup_config(scanned_ids)

    def _return_motor_config(self, scanned_ids: List[int]) -> dict:
        # Only keep motors whose IDs are in the scanned list
        filtered_motors = {}
        for motor_name, motor_config in self.config["motors"].items():
            if motor_config["id"] in scanned_ids:
                filtered_motors[motor_name] = motor_config
            else:
                print(f"Motor {motor_name} (ID: {motor_config['id']}) not found")

        return filtered_motors

    def _return_motorgroup_config(self, scanned_ids: List[int]) -> dict:
        # Get motor names that are in the scanned IDs list
        available_motors = []
        for motor_name, motor_config in self.config["motors"].items():
            if motor_config["id"] in scanned_ids:
                available_motors.append(motor_name)

        # Filter each motorgroup to only include available motors
        filtered_motorgroups = {}
        for group_name, motors_list in self.config["motorgroups"].items():
            filtered_motorgroups[group_name] = [
                motor for motor in motors_list if motor in available_motors
            ]

        return filtered_motorgroups
