import json
import os
import time
from typing import List, Union
from pypot.primitive import Primitive, LoopPrimitive
from pypot.robot import Robot

from .robot import BlossomRobot
from .frame import Frame
from .position import Position
from ..constants import SEQUENCE_DIR, get_blossom_robot


class Sequence(Primitive):
    def __init__(self, robot: BlossomRobot, animation: str, frames: List[Frame]):
        super().__init__(robot)
        self.animation = animation
        self.frames = frames

    @staticmethod
    def _load_frames(sequence_json: dict) -> List[Frame] | None:
        frames: List[Frame] = []

        if "frame_list" not in sequence_json:
            return None

        for frame in sequence_json["frame_list"]:
            positions = []
            for p in frame["positions"]:
                try:
                    positions.append(Position(
                        dof=p["dof"],
                        pos=p["pos"],
                    ))
                except Exception:
                    # Skip invalid positions (e.g., unsupported DOF like 'arms')
                    continue
            
            if positions:  # Only add frame if it has valid positions
                frames.append(Frame(positions=positions, millis=frame["millis"]))

        return frames if frames else None

    @classmethod
    def from_config(cls, sequence_path: str, robot: BlossomRobot):
        with open(sequence_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if "animation" not in data:
            return None

        animation: str = data["animation"]
        frames: List[Frame] | None = cls._load_frames(data)

        if frames is None:
            return None

        return cls(robot, animation, frames)

    @classmethod
    def from_json(cls, sequence_json: dict, robot: BlossomRobot) -> "Sequence":
        if "animation" not in sequence_json:
            return None

        animation: str = sequence_json["animation"]
        frames: List[Frame] | None = cls._load_frames(sequence_json)

        if frames is None:
            return None

        return cls(robot, animation, frames)

    @staticmethod
    def get_all_sequences() -> List["Sequence"]:
        sequences: List["Sequence"] = []
        for file in os.listdir(SEQUENCE_DIR):
            if file.endswith("sequence.json"):
                try:
                    seq = Sequence.from_config(os.path.join(SEQUENCE_DIR, file), get_blossom_robot())
                    if seq is not None:
                        sequences.append(seq)
                except Exception as e:
                    # Skip invalid sequences
                    print(f"Warning: Skipping invalid sequence {file}: {e}")
                    continue
        return sequences

    @staticmethod
    def sequence_str(sequences: Union["Sequence", List["Sequence"]]) -> str:
        if isinstance(sequences, List):
            return "\n".join([sequence.animation for sequence in sequences])

        return sequences.animation

    def setup(self):
        print(f"Setting up sequence: {self.animation}")

    def run(self):
        # Assumption of amp = 1 and post = 0 for simplicity
        start_time = time.time()
        motor_names = [m.name for m in self.robot.motors]

        for frame in self.frames:
            if not self._running:
                break

            current_time = (time.time() - start_time) * 1000
            delay_millis = frame.millis - current_time

            if delay_millis <= 0:
                delay_millis = 200

            delay = delay_millis / 1000

            positions = {}
            for position in frame.positions:
                motor = position.dof
                if motor in motor_names:
                    converted_position = self.adjust_position(position.pos, motor)
                    positions[motor] = converted_position
                    self.robot.goto_position(
                        {motor: converted_position}, delay, wait=False
                    )

            # self.robot.goto_position(positions, delay, wait=False)
            time.sleep(delay)

    def adjust_position(self, position: float, motor: str) -> float:
        converted_position = self._convert_rad_to_angle(position)
        max_position = self.robot.range_pos[motor][1]
        min_position = self.robot.range_pos[motor][0]

        if converted_position > max_position:
            converted_position = max_position
        elif converted_position < min_position:
            converted_position = min_position

        return converted_position

    def _convert_rad_to_angle(self, rad: float) -> float:
        return (float(rad) - 3) * 50  # Used in old code

    def teardown(self):
        print(f"Tearing down sequence: {self.animation}")


class SequenceLoop(LoopPrimitive):
    def __init__(self, robot: Robot, sequence: Sequence):
        last_frame = sequence.frames[-1]
        super().__init__(robot, freq=last_frame.millis / 1000)
        self.sequence = sequence
        self.current_frame_index = 0
        self.start_time = None

    def setup(self):
        print(f"Starting sequence loop: {self.sequence.animation}")
        self.start_time = time.time()
        self.current_frame_index = 0

    def update(self):
        if not self.sequence.frames:
            return

        current_frame = self.sequence.frames[self.current_frame_index]
        next_index = (self.current_frame_index + 1) % len(self.sequence.frames)

        positions = {}
        for position in current_frame.positions:
            motor = position.dof
            if motor in self.robot.motors:
                converted_position = self.sequence.adjust_position(position.pos, motor)
                positions[motor] = converted_position

        next_frame = self.sequence.frames[next_index]
        duration = next_frame.millis / 1000

        self.robot.goto_position(positions, duration)

        self.current_frame_index = next_index

    def teardown(self):
        print(f"Stopping sequence loop: {self.sequence.animation}")
