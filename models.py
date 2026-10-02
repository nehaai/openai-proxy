from typing import Literal
from pydantic import BaseModel, Field, ConfigDict

class FunctionCall(BaseModel):
    name: str
    arguments: str  # spec: a JSON-encoded string, not an object


class ToolCall(BaseModel):
    id: str
    type: Literal["function"] = "function"
    function: FunctionCall


class Message(BaseModel):
    role: Literal["system", "developer", "user", "assistant", "tool"]
    content: str | None = None
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None


class StreamOptions(BaseModel):
    include_usage: bool = False


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    model: str
    messages: list[Message] = Field(min_length=1)
    temperature: float | None = Field(None, ge=0, le=2)
    top_p: float | None = Field(None, ge=0, le=1)
    max_tokens: int | None = None
    max_completion_tokens: int | None = Field(None, gt=0)
    stop: str | list[str] | None = None
    stream: bool = False
    n: int = Field(1, ge=1, le=1)  # n>1 not supported yet
    stream_options: StreamOptions | None = None
    tools: list[dict] | None = None
    tool_choice: str | dict | None = None
    response_format: dict | None = None
    
