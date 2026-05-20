import os
from .models.robot import BlossomRobot
from .models.robot_config import RobotConfig

CONFIG = None
BLOSSOM_ROBOT = None

# 🔥 FIX: make path absolute instead of relative
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
SEQUENCE_DIR = os.path.join(BASE_DIR, "sequences")

def get_config():
    global CONFIG
    if CONFIG is None:
        CONFIG = RobotConfig().config
    return CONFIG

def get_blossom_robot():
    global BLOSSOM_ROBOT
    if BLOSSOM_ROBOT is None:
        BLOSSOM_ROBOT = BlossomRobot.from_config(get_config())
    return BLOSSOM_ROBOT