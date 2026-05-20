from typing import Annotated, List, Optional
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages


class ChatBotState(BaseModel):
    messages: Annotated[List[BaseMessage], add_messages]
    sequence_to_play: Optional[str] = Field(default=None, description="The name of the sequence to play")
    should_play_sequence: bool = Field(default=False, description="Whether to play a sequence based on the conversation emotion")


class ShouldPlaySequence(BaseModel):
    should_play_sequence: bool = Field(default=False, description="Whether to play a sequence based on the conversation emotion")


class SequenceSelectorOutput(BaseModel):
    sequence_name: str = Field(default="", description="The name of the sequence to play")