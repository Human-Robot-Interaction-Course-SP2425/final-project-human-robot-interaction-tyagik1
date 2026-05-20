from typing import List
from pydantic import BaseModel
from apps.shared.models.position import Position


class Frame(BaseModel):
    positions: List[Position]
    millis: float
