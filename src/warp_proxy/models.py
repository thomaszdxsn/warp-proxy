"""OpenAI-compatible API 요청/응답 모델 정의.

Pydantic v2 BaseModel을 사용하며, FastAPI의 ReDoc/Swagger에 자동 반영된다.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

# ---------------------------------------------------------------------------
# 에러 모델
# ---------------------------------------------------------------------------


class APIErrorEnvelope(BaseModel):
    """에러 응답 외부 envelope. {"error": {...}} 형식."""
    error: "APIError"


class APIError(BaseModel):
    """OpenAI-style 에러 본문."""

    message: str
    type: str = "invalid_request_error"
    param: str | None = None
    code: str


# ---------------------------------------------------------------------------
# 요청 모델
# ---------------------------------------------------------------------------


class MessageTextPart(BaseModel):
    """메시지 content 배열의 텍스트 파트."""
    type: Literal["text"]
    text: str


class ChatMessage(BaseModel):
    """단일 대화 메시지. content는 문자열 또는 text 파트 배열."""

    role: str
    content: str | list[MessageTextPart]

    @model_validator(mode="before")
    @classmethod
    def validate_content(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            raise TypeError("message must be an object")
        content = data.get("content")
        if isinstance(content, str):
            return data
        if isinstance(content, list):
            for index, part in enumerate(content):
                if not isinstance(part, dict):
                    raise ValueError(f"messages content part at index {index} must be an object")
                if part.get("type") != "text" or not isinstance(part.get("text"), str):
                    raise ValueError(f"messages content part at index {index} must be a text part")
            return data
        raise ValueError("messages content must be a string or an array of text parts")

    def flattened_content(self) -> str:
        """content가 배열이면 \n으로 이어붙여 단일 문자열로 반환."""
        if isinstance(self.content, str):
            return self.content
        return "\n".join(part.text for part in self.content)


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request."""

    model_config = ConfigDict(extra="forbid")

    model: str = Field(description="Model ID (e.g. `warp-oz-cli`)")
    messages: list[ChatMessage] = Field(description="Conversation messages")
    stream: bool = Field(False, description="Enable SSE streaming")
    temperature: float | None = Field(None, description="Sampling temperature")
    top_p: float | None = Field(None, description="Nucleus sampling")
    max_tokens: int | None = Field(None, description="Max response tokens")
    stop: str | list[str] | None = Field(None, description="Stop sequence(s)")
    user: str | None = Field(None, description="End-user identifier")
    metadata: dict[str, Any] | None = Field(
        None, description="Proxy metadata (e.g. `warp_previous_response_id` for continuation)",
    )

    # Accepted for OpenAI client compatibility — not forwarded to Oz.
    tools: Any | None = Field(None, description="Ignored — compatibility only")
    tool_choice: Any | None = Field(None, description="Ignored — compatibility only")
    functions: Any | None = Field(None, description="Ignored — compatibility only")
    function_call: Any | None = Field(None, description="Ignored — compatibility only")
    response_format: Any | None = Field(None, description="Ignored — compatibility only")
    audio: Any | None = Field(None, description="Ignored — compatibility only")
    parallel_tool_calls: Any | None = Field(None, description="Ignored — compatibility only")
    stream_options: Any | None = Field(None, description="Ignored — compatibility only")
    max_completion_tokens: Any | None = Field(None, description="Ignored — compatibility only")
    store: Any | None = Field(None, description="Ignored — compatibility only")


class ResponsesRequest(BaseModel):
    """OpenAI Responses API 요청(부분 호환)."""

    model_config = ConfigDict(extra="allow")

    model: str
    input: Any
    stream: bool = False
    instructions: Any | None = None
    previous_response_id: str | None = None
    max_output_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    stop: str | list[str] | None = None
    user: str | None = None
    metadata: dict[str, Any] | None = None
    tools: Any | None = None
    tool_choice: Any | None = None
    truncation: str | None = None

# ---------------------------------------------------------------------------
# 요청 모델 (Anthropic-compatible)
# ---------------------------------------------------------------------------


class AnthropicMessage(BaseModel):
    """Anthropic Messages API의 단일 입력 메시지."""

    model_config = ConfigDict(extra="allow")

    role: Literal["user", "assistant"]
    content: Any

    @model_validator(mode="after")
    def validate_content(self) -> "AnthropicMessage":
        if not isinstance(self.content, (str, list)):
            raise ValueError("content must be a string or an array of blocks")
        return self


class AnthropicMessagesRequest(BaseModel):
    """Anthropic-compatible messages request."""

    model_config = ConfigDict(extra="allow")

    model: str
    messages: list[AnthropicMessage]
    max_tokens: int | None = None
    system: Any | None = None
    stream: bool = False
    temperature: float | None = None
    top_p: float | None = None
    stop_sequences: list[str] | None = None
    metadata: dict[str, Any] | None = None
    tools: Any | None = None
    tool_choice: Any | None = None


class AnthropicCountTokensResponse(BaseModel):
    """`/v1/messages/count_tokens` 응답."""

    input_tokens: int


class AnthropicTextBlock(BaseModel):
    """Anthropic 메시지의 텍스트 블록."""

    type: Literal["text"] = "text"
    text: str


class AnthropicUsage(BaseModel):
    """Anthropic usage shape (현재는 proxy 추정/기본값)."""

    input_tokens: int = 0
    output_tokens: int = 0


class AnthropicMessageResponse(BaseModel):
    """Anthropic-compatible message response."""

    id: str
    type: Literal["message"] = "message"
    role: Literal["assistant"] = "assistant"
    model: str
    content: list[AnthropicTextBlock]
    stop_reason: str | None = "end_turn"
    stop_sequence: str | None = None
    usage: AnthropicUsage = Field(default_factory=AnthropicUsage)


class AnthropicError(BaseModel):
    """Anthropic-style error body."""

    type: str = "invalid_request_error"
    message: str


class AnthropicErrorEnvelope(BaseModel):
    """Anthropic-style error envelope."""

    type: Literal["error"] = "error"
    error: AnthropicError


class AnthropicMessageStartEvent(BaseModel):
    type: Literal["message_start"] = "message_start"
    message: AnthropicMessageResponse


class AnthropicContentBlockStartEvent(BaseModel):
    type: Literal["content_block_start"] = "content_block_start"
    index: int = 0
    content_block: AnthropicTextBlock


class AnthropicTextDelta(BaseModel):
    type: Literal["text_delta"] = "text_delta"
    text: str


class AnthropicContentBlockDeltaEvent(BaseModel):
    type: Literal["content_block_delta"] = "content_block_delta"
    index: int = 0
    delta: AnthropicTextDelta


class AnthropicContentBlockStopEvent(BaseModel):
    type: Literal["content_block_stop"] = "content_block_stop"
    index: int = 0


class AnthropicMessageDeltaPayload(BaseModel):
    stop_reason: str | None = "end_turn"
    stop_sequence: str | None = None


class AnthropicMessageDeltaEvent(BaseModel):
    type: Literal["message_delta"] = "message_delta"
    delta: AnthropicMessageDeltaPayload = Field(default_factory=AnthropicMessageDeltaPayload)
    usage: AnthropicUsage = Field(default_factory=AnthropicUsage)


class AnthropicMessageStopEvent(BaseModel):
    type: Literal["message_stop"] = "message_stop"


class AnthropicStreamErrorEvent(BaseModel):
    type: Literal["error"] = "error"
    error: AnthropicError


# ---------------------------------------------------------------------------
# 응답 모델 (non-streaming)
# ---------------------------------------------------------------------------


class ChatCompletionMessage(BaseModel):
    """응답 메시지 본문."""

    role: Literal["assistant"] = "assistant"
    content: str


class ChatCompletionChoice(BaseModel):
    """응답 선택지. warp-proxy는 항상 단일 choice(index=0)만 반환."""
    index: int = 0
    message: ChatCompletionMessage
    finish_reason: Literal["stop"] = "stop"


class Usage(BaseModel):
    """Oz CLI는 토큰 정보를 반환하지 않으므로 항상 0."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    """Non-streaming chat completion response."""

    id: str = Field(description="Unique response ID")
    object: Literal["chat.completion"] = "chat.completion"
    created: int = Field(description="Unix timestamp")
    model: str = Field(description="Model used")
    choices: list[ChatCompletionChoice] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)


# ---------------------------------------------------------------------------
# 응답 모델 (streaming / SSE)
# ---------------------------------------------------------------------------


class ChatCompletionChunkDelta(BaseModel):
    """SSE chunk의 delta 페이로드."""

    role: Literal["assistant"] | None = None
    content: str | None = None


class ChatCompletionChunkChoice(BaseModel):
    """SSE chunk 선택지."""

    index: int = 0
    delta: ChatCompletionChunkDelta = Field(default_factory=ChatCompletionChunkDelta)
    finish_reason: Literal["stop"] | None = None


class ChatCompletionChunkResponse(BaseModel):
    """SSE 스트리밍 응답 chunk."""
    id: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int
    model: str
    choices: list[ChatCompletionChunkChoice] = Field(default_factory=list)


class ModelDescriptor(BaseModel):
    id: str = Field(description="Model ID")
    object: Literal["model"] = "model"
    owned_by: str = "warp-proxy"


class ModelsResponse(BaseModel):
    """/v1/models 응답."""

    object: Literal["list"] = "list"
    data: list[ModelDescriptor]


# ---------------------------------------------------------------------------
# 운영 모델 (/admin/status)
# ---------------------------------------------------------------------------


class ModelAvailability(BaseModel):
    """모델 alias의 가용성 상태."""

    id: str
    available: bool
    reason: str | None = None


class CwdStatus(BaseModel):
    """작업 디렉토리 설정 상태."""
    configured: bool
    value: str | None = None


class VersionProbeStatus(BaseModel):
    """Warp CLI 버전 프로브 결과."""

    checked: bool
    supported: bool | None = None
    detected_version: str | None = None
    allow_unverified_warp_cli: bool
    verified_warp_versions: list[str]
    error_code: str | None = None
    error_message: str | None = None


class AdminStatusResponse(BaseModel):
    """Operator health-check response."""

    service: Literal["warp-proxy"] = "warp-proxy"
    version: str = Field(description="warp-proxy version")
    auth_mode: Literal["session", "api_key"] = Field(description="Current auth mode")
    cwd: CwdStatus = Field(description="Working directory status")
    models: list[ModelAvailability] = Field(description="Model availability")
    version_probe: VersionProbeStatus = Field(description="Warp CLI version check")


APIErrorEnvelope.model_rebuild()
