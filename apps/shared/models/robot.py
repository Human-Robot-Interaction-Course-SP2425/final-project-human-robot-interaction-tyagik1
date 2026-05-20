from typing import List
from pypot.robot import Robot, from_config
from pypot.dynamixel.motor import DxlMotor


class BlossomRobot(Robot):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.reset_pos = {
            "tower_1": 50,
            "tower_2": 50,
            "tower_3": 50,
            "base": 0,
            "ears": 100,
        }
        self.range_pos = {
            "tower_1": (-40, 140),
            "tower_2": (-40, 140),
            "tower_3": (-40, 140),
            "base": (-140, 140),
            "ears": (0, 140),
        }

    @classmethod
    def from_config(cls, config: dict):
        base_robot = from_config(config)
        blossom = cls()
        blossom.__dict__.update(base_robot.__dict__)
        blossom.power_up()
        blossom.compliant = False
        blossom.reset_position()
        return blossom

    def reset_position(self):
        positions = {}
        for motor in self.motors:
            positions[motor.name] = self.reset_pos[motor.name]
        self.goto_position(positions, 1, wait=False)

    # All motors are DxlMXMotor
    @property
    def motors(self) -> List[DxlMotor]:
        return self._motors
