from pydantic import BaseModel, field_validator
from typing import List
import json

from apps.shared.models.frame import Frame
from apps.shared.models.position import Position


class SequenceRequest(BaseModel):
    animation: str
    frame_list: List[Frame]
    
    @field_validator('frame_list')
    @classmethod
    def validate_frames(cls, v):
        if not v:
            raise ValueError('frame_list cannot be empty')
        
        # Validate each frame has at least one position
        for i, frame in enumerate(v):
            if not frame.positions:
                raise ValueError(f'Frame {i} has no positions')
            
            # Validate position values are in valid range (1-5)
            for pos in frame.positions:
                if pos.pos < 1 or pos.pos > 5:
                    raise ValueError(
                        f'Position value {pos.pos} for {pos.dof} in frame {i} '
                        f'is out of range (1-5)'
                    )
        
        return v
    
    @field_validator('animation')
    @classmethod
    def validate_animation_name(cls, v):
        if not v or not v.strip():
            raise ValueError('animation name cannot be empty')
        return v.strip()