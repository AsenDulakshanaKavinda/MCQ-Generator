

from pydantic import BaseModel, Field
from typing import Annotated
from enum import Enum

class ChatAnswer(BaseModel):
    """
    validate chat answer type and length.
    """
    answer: Annotated[str, Field(min_length=1, max_length=4096*4)]

class PromptType(str, Enum):
    SYSTEM_PROMPT = "system_prompt"
    AI_PROMPT = "ai_prompt"
    HUMAN_PROMPT = "human_prompt"

class UploadResponse(BaseModel):
    session_id: str
    indexed: bool
    nessage: str | None = None

class ChatRequest(BaseModel):
    session_id: str
    message: str

class ChatResponse(BaseModel):
    answer: str  
    







