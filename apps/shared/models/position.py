from typing import Literal
from pydantic import BaseModel

type Motor = Literal["tower_1", "tower_2", "tower_3", "base", "ears"]

class Position(BaseModel):
    dof: Motor
    pos: float
