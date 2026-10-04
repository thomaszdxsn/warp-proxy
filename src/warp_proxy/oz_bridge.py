"""Oz CLI 브리지.

OpenAI-compatible 요청을 `oz agent run` CLI 호출로 변환하고,
CLI NDJSON 출력을 OpenAI-style 응답으로 정규화한다.
"""

from __future__ import annotations

import asyncio
import json
import re
import subprocess
import time
import uuid
from contextlib import suppress
from dataclasses import dataclass
from typing import Any, AsyncIterator, Protocol

from .config import Settings
from .conversation_store import ConversationRecord, ConversationStore, ConversationStoreError
from .models import (
    APIError,
    APIErrorEnvelope,
    AdminStatusResponse,
    ChatCompletionChunkChoice,
    ChatCompletionChunkDelta,
    ChatCompletionChunkResponse,
    ChatCompletionChoice,
    ChatCompletionMessage,
    ChatCompletionRequest,
    ChatCompletionResponse,
    CwdStatus,
    ModelAvailability,
    ModelDescriptor,
    ModelsResponse,
    Usage,
    VersionProbeStatus,
)

# ---------------------------------------------------------------------------
# 상수
# ---------------------------------------------------------------------------

DEFAULT_MODEL_ALIAS = "warp-oz-cli"  # stable public alias
DEFAULT_OWNER = "warp-proxy"

# /v1/models 기본 노출 목록. WARP_PROXY_LIST_ALL_MODELS=true 시 전체 노출.
CURATED_MODEL_IDS: tuple[str, ...] = (
    "auto",
    "auto-efficient",
    "auto-genius",
    "auto-open",
    "claude-4-5-opus-thinking",
    "claude-4-5-sonnet",
    "claude-4-5-sonnet-thinking",
    "claude-4-6-opus-high",
    "claude-4-6-opus-max",
    "claude-4-6-sonnet-high",
    "claude-4-6-sonnet-max",
    "claude-4-7-opus-max",
    "claude-4-7-opus-xhigh",
    "claude-4-8-opus-high",
    "claude-4-8-opus-max",
    "claude-4-8-opus-xhigh",
    "claude-4-8-opus-xhigh-fast",
    "claude-5-opus-max",
    "claude-5-opus-xhigh",
    "claude-5-sonnet-max",
    "claude-5-sonnet-xhigh",
    "claude-5-fable-max",
    "claude-5-fable-xhigh",
    "claude-5-1-fable-max",
    "claude-5-1-fable-xhigh",
    "claude-5-1-fable-high",
    "deepseek-v4-pro-fireworks",
    "gemini-3.1-pro",
    "gpt-5-3-codex-high",
    "gpt-5-3-codex-low",
    "gpt-5-3-codex-medium",
    "gpt-5-3-codex-xhigh",
    "gpt-5-4-high",
    "gpt-5-4-low",
    "gpt-5-4-medium",
    "gpt-5-4-xhigh",
    "gpt-5-5-high",
    "gpt-5-5-xhigh",
    "gpt-5-6-luna-xhigh",
    "gpt-5-6-sol-xhigh",
    "gpt-5-6-terra-xhigh",
    "gpt-6-astra-high",
    "gpt-6-astra-max",
    "gpt-6-astra-xhigh",
    "grok-4-6-high",
    "grok-4-6-xhigh",
)
# 수신은 하지만 Oz로 전달하지 않는 필드 (OpenAI SDK 호환용)
SUPPORTED_UNSUPPORTED_FIELDS = (
    "tools",
    "tool_choice",
    "functions",
    "function_call",
    "response_format",
    "audio",
    "parallel_tool_calls",
    "stream_options",
    "max_completion_tokens",
    "store",
)
# oz dump-debug-info 출력에서 버전을 추출하는 정규식
WARP_VERSION_PATTERN = re.compile(r'Warp version:\s+Some\("(?P<version>[^"]+)"\)')

# 백엔드 실패 원인을 분류하는 패턴
_AUTH_FAILURE_PATTERN = re.compile(
    r"(login|logged in|api key|authentication|credentials|unauthorized|forbidden|session)",
    re.IGNORECASE,
)
_CONVERSATION_EXPIRED_PATTERN = re.compile(
    r"conversation.*(expired|not found|unknown|does not exist|invalid)",
    re.IGNORECASE,
)
# asyncio StreamReader의 기본 한 줄 한계(64 KiB)보다 긴 NDJSON 이벤트를 허용한다.
# 도구 출력·긴 코드 블록이 한 줄로 몰리는 경우 실제로 초과한다.
_STREAM_READER_LIMIT_BYTES = 16 * 1024 * 1024



# ---------------------------------------------------------------------------
# 예외 및 데이터 클래스
# ---------------------------------------------------------------------------


class ProxyError(Exception):
    """프록시 수준의 에러. HTTP status + OpenAI-style error body를 포함한다."""
    def __init__(self, *, status_code: int, code: str, message: str, param: str | None = None, error_type: str = "invalid_request_error") -> None:
        super().__init__(message)
        self.status_code = status_code
        self.error = APIError(message=message, type=error_type, param=param, code=code)


@dataclass(slots=True)
class CommandResult:
    """CLI 실행 결과."""

    args: list[str]
    returncode: int
    stdout: str
    stderr: str


@dataclass(slots=True)
class VersionStatus:
    """Warp CLI 버전 프로브 결과."""
    checked: bool
    supported: bool
    version: str | None
    error_code: str | None = None
    error_message: str | None = None


@dataclass(slots=True)
class ResolvedModel:
    """모델 리졸브 결과. public alias → 백엔드 커맨드 + Oz 모델 ID."""

    public_model: str       # 클라이언트가 보낸 모델명
    backend_command: str    # "run"
    oz_model_id: str | None = None  # namespaced 요청 시 실제 Oz 모델 ID


@dataclass(slots=True)
class PreparedExecution:
    """실행 준비 완료된 요청. CLI args, 응답 ID, continuation 정보 포함."""

    request: ChatCompletionRequest
    response_id: str
    created: int
    model: ResolvedModel
    args: list[str]
    prior_response_id: str | None
    prior_record: ConversationRecord | None


@dataclass(slots=True)
class PreparedStream:
    """스트리밍 준비 결과. 첫 텍스트를 prime한 후 나머지 이벤트 이터레이터를 들고 있다."""

    prepared: PreparedExecution
    event_iter: AsyncIterator[ParsedEvent]
    first_text: str
    conversation_id: str | None


@dataclass(slots=True)
class ParsedEvent:
    """NDJSON 이벤트 한 줄을 파싱한 결과."""

    kind: str  # "agent", "system", 또는 원본 type 값
    text: str | None = None
    conversation_id: str | None = None
    payload: dict[str, Any] | None = None


class CommandRunner(Protocol):
    """테스트 시 모킹할 수 있는 CLI 실행 인터페이스."""

    def run(self, *, args: list[str], timeout_seconds: float) -> CommandResult:
        ...


class SubprocessCommandRunner:
    """실제 subprocess로 CLI를 실행하는 기본 구현체."""
    def run(self, *, args: list[str], timeout_seconds: float) -> CommandResult:
        completed = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
        return CommandResult(
            args=args,
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
        )


class OzBridge:
    """Oz CLI와 OpenAI-compatible API 간의 브리지.

    요청 검증, 모델 리졸브, CLI 실행, NDJSON 파싱, 응답 정규화를 담당한다.
    """

    def __init__(self, settings: Settings, runner: CommandRunner | None = None) -> None:
        self.settings = settings
        self.runner = runner or SubprocessCommandRunner()
        self._version_status: VersionStatus | None = None
        self._catalog: tuple[str, ...] | None = None
        self._conversation_store = ConversationStore(settings.conversation_store_path)
        self._semaphore = asyncio.Semaphore(settings.max_concurrent_requests)

    async def list_models(self) -> ModelsResponse:
        """모델 목록을 반환한다. discovery가 필요하면 to_thread로 실행."""
        return await asyncio.to_thread(self._list_models_sync)

    def _list_models_sync(self) -> ModelsResponse:
        models = [
            ModelDescriptor(id=DEFAULT_MODEL_ALIAS, owned_by=DEFAULT_OWNER),
        ]
        if self.settings.list_all_models:
            discovered = self._discover_models_best_effort()
            if discovered:
                models.extend(ModelDescriptor(id=_namespaced_model_id(DEFAULT_MODEL_ALIAS, model_id), owned_by=DEFAULT_OWNER) for model_id in discovered)
        else:
            models.extend(
                ModelDescriptor(id=_namespaced_model_id(DEFAULT_MODEL_ALIAS, model_id), owned_by=DEFAULT_OWNER)
                for model_id in CURATED_MODEL_IDS
            )
        return ModelsResponse(data=models)

    async def get_admin_status(self) -> AdminStatusResponse:
        """운영 상태를 반환한다. 버전 프로브가 필요하면 to_thread로 실행."""
        return await asyncio.to_thread(self._get_admin_status_sync)

    def _get_admin_status_sync(self) -> AdminStatusResponse:
        version_status = self._get_or_probe_version_status()
        return AdminStatusResponse(
            version=self.settings.app_version,
            auth_mode=self.settings.auth_mode,
            cwd=CwdStatus(configured=bool(self.settings.cwd), value=self.settings.cwd),
            models=self._model_availability(),
            version_probe=VersionProbeStatus(
                checked=version_status.checked,
                supported=version_status.supported,
                detected_version=version_status.version,
                allow_unverified_warp_cli=self.settings.allow_unverified_warp_cli,
                verified_warp_versions=list(self.settings.verified_warp_versions),
                error_code=version_status.error_code,
                error_message=version_status.error_message,
            ),
        )

    async def create_chat_completion(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        """Non-streaming 요청을 처리한다. 세마포어로 동시 실행 수를 제한한다."""
        async with self._semaphore:
            return await asyncio.to_thread(self._create_chat_completion_sync, request)

    def _create_chat_completion_sync(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        prepared = self._prepare_execution(request)
        return self._execute_local_chat_completion(prepared)

    async def prepare_stream_chat_completion(self, request: ChatCompletionRequest) -> PreparedStream:
        """스트리밍 요청을 준비한다. 세마포어로 동시 실행 수를 제한한다."""
        await self._semaphore.acquire()
        try:
            prepared = await asyncio.to_thread(self._prepare_execution, request)
            event_iter = self._stream_local_backend_events(prepared)
            first_text, conversation_id = await self._prime_stream(event_iter, prepared)
            return PreparedStream(prepared=prepared, event_iter=event_iter, first_text=first_text, conversation_id=conversation_id)
        except BaseException:
            self._semaphore.release()
            raise

    async def stream_chat_completion_sse(self, stream: PreparedStream) -> AsyncIterator[str]:
        prepared = stream.prepared
        event_iter = stream.event_iter
        conversation_id = stream.conversation_id
        sent_any_chunk = False
        try:
            yield _sse_data(
                ChatCompletionChunkResponse(
                    id=prepared.response_id,
                    created=prepared.created,
                    model=prepared.request.model,
                    choices=[ChatCompletionChunkChoice(delta=ChatCompletionChunkDelta(role="assistant"))],
                ).model_dump_json(exclude_none=True)
            )
            sent_any_chunk = True
            if stream.first_text:
                yield _sse_data(
                    ChatCompletionChunkResponse(
                        id=prepared.response_id,
                        created=prepared.created,
                        model=prepared.request.model,
                        choices=[ChatCompletionChunkChoice(delta=ChatCompletionChunkDelta(content=stream.first_text))],
                    ).model_dump_json(exclude_none=True)
                )
            async for event in event_iter:
                if event.conversation_id:
                    conversation_id = event.conversation_id
                if event.kind == "agent" and event.text:
                    yield _sse_data(
                        ChatCompletionChunkResponse(
                            id=prepared.response_id,
                            created=prepared.created,
                            model=prepared.request.model,
                            choices=[ChatCompletionChunkChoice(delta=ChatCompletionChunkDelta(content=event.text))],
                        ).model_dump_json(exclude_none=True)
                    )
            conversation_id = conversation_id or (prepared.prior_record.conversation_id if prepared.prior_record else None)
            if conversation_id:
                self._persist_mapping(prepared.response_id, conversation_id, prepared.model.backend_command)
            yield _sse_data(
                ChatCompletionChunkResponse(
                    id=prepared.response_id,
                    created=prepared.created,
                    model=prepared.request.model,
                    choices=[ChatCompletionChunkChoice(delta=ChatCompletionChunkDelta(), finish_reason="stop")],
                ).model_dump_json(exclude_none=True)
            )
            yield "data: [DONE]\n\n"
        except ProxyError as exc:
            if not sent_any_chunk:
                raise
            yield _sse_data(APIErrorEnvelope(error=exc.error).model_dump_json())
        finally:
            self._semaphore.release()

    async def stream_chat_completion_text_deltas(self, stream: PreparedStream) -> AsyncIterator[str]:
        """스트림에서 assistant 텍스트 delta만 순서대로 반환한다.

        이 메서드는 프로토콜별 직렬화(OpenAI/Anthropic)를 분리하기 위한 공통 경로다.
        conversation 매핑 저장과 semaphore 해제까지 책임진다.
        """
        prepared = stream.prepared
        event_iter = stream.event_iter
        conversation_id = stream.conversation_id
        try:
            if stream.first_text:
                yield stream.first_text
            async for event in event_iter:
                if event.conversation_id:
                    conversation_id = event.conversation_id
                if event.kind == "agent" and event.text:
                    yield event.text
            conversation_id = conversation_id or (prepared.prior_record.conversation_id if prepared.prior_record else None)
            if conversation_id:
                self._persist_mapping(prepared.response_id, conversation_id, prepared.model.backend_command)
        finally:
            self._semaphore.release()

    def estimate_input_tokens(self, request: ChatCompletionRequest) -> int:
        """입력 텍스트 기반의 best-effort 토큰 추정치."""
        prompt = flatten_messages(request).strip()
        if not prompt:
            return 0
        return len(prompt.split())

    def _execute_local_chat_completion(self, prepared: PreparedExecution) -> ChatCompletionResponse:
        result = self._run_sync(prepared.args, timeout=self.settings.agent_run_timeout_seconds)
        if result.returncode != 0:
            raise self._map_backend_failure(result, prior_response_id=prepared.prior_response_id)
        events = parse_ndjson_events(result.stdout)
        content, conversation_id = aggregate_events(events)
        conversation_id = conversation_id or (prepared.prior_record.conversation_id if prepared.prior_record else None)
        if conversation_id:
            self._persist_mapping(prepared.response_id, conversation_id, prepared.model.backend_command)
        return ChatCompletionResponse(
            id=prepared.response_id,
            created=prepared.created,
            model=prepared.request.model,
            choices=[ChatCompletionChoice(message=ChatCompletionMessage(content=content))],
            usage=Usage(),
        )

    def _prepare_execution(self, request: ChatCompletionRequest) -> PreparedExecution:
        self._validate_request(request)
        resolved_model = self._resolve_model(request.model)
        self._ensure_supported_cli_version()
        prior_response_id, prior_record = self._resolve_continuation(request, resolved_model)
        prompt = flatten_messages(request)
        args = self._build_command(
            prompt=prompt,
            oz_model_id=resolved_model.oz_model_id,
            conversation_id=prior_record.conversation_id if prior_record else None,
        )
        return PreparedExecution(
            request=request,
            response_id=f"chatcmpl_{uuid.uuid4().hex}",
            created=int(time.time()),
            model=resolved_model,
            args=args,
            prior_response_id=prior_response_id,
            prior_record=prior_record,
        )

    def _validate_request(self, request: ChatCompletionRequest) -> None:
        if not self.settings.ignore_unsupported_fields:
            for field_name in SUPPORTED_UNSUPPORTED_FIELDS:
                if getattr(request, field_name) is not None:
                    raise ProxyError(status_code=400, code="unsupported_field", message=f"{field_name} is not supported in this proxy.", param=field_name)
        if not request.messages:
            raise ProxyError(status_code=400, code="invalid_message_content", message="messages must contain at least one message.", param="messages")

    def _resolve_model(self, public_model: str) -> ResolvedModel:
        if public_model == DEFAULT_MODEL_ALIAS:
            return ResolvedModel(public_model=public_model, backend_command="run")
        if public_model.startswith(f"{DEFAULT_MODEL_ALIAS}/"):
            oz_model_id = public_model.split("/", 1)[1]
            self._ensure_namespaced_model_available(DEFAULT_MODEL_ALIAS, oz_model_id)
            return ResolvedModel(public_model=public_model, backend_command="run", oz_model_id=oz_model_id)
        raise ProxyError(status_code=400, code="unsupported_model", message=f"Model '{public_model}' is not supported.", param="model")

    def _discover_models_best_effort(self) -> tuple[str, ...] | None:
        if self._catalog is not None:
            return self._catalog
        try:
            self._catalog = self._discover_model_catalog(required=False)
        except ProxyError:
            return None
        return self._catalog

    def _ensure_namespaced_model_available(self, namespace: str, oz_model_id: str) -> None:
        if not oz_model_id:
            raise ProxyError(status_code=400, code="unsupported_model", message="Namespaced model id must not be empty.", param="model")
        if self._catalog is None:
            self._catalog = self._discover_model_catalog(required=True)
        if oz_model_id not in self._catalog:
            self._catalog = self._discover_model_catalog(required=True, refresh=True)
            if oz_model_id not in self._catalog:
                raise ProxyError(status_code=400, code="unsupported_model", message=f"Model '{namespace}/{oz_model_id}' is not supported.", param="model")

    def _discover_model_catalog(self, *, required: bool, refresh: bool = False) -> tuple[str, ...]:
        try:
            result = self._run_sync(self._build_model_list_command(), timeout=min(self.settings.command_timeout_seconds, 30.0))
        except ProxyError:
            if required:
                raise
            return self._catalog or ()
        if result.returncode != 0:
            if required:
                raise ProxyError(status_code=503, code="model_catalog_unavailable", message=result.stderr.strip() or "Could not load Oz model catalog.")
            return self._catalog or ()
        try:
            payload = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            if required:
                raise ProxyError(status_code=503, code="model_catalog_unavailable", message="Oz model catalog returned malformed JSON.") from exc
            return self._catalog or ()
        catalog = _parse_model_catalog(payload)
        if not catalog and required:
            raise ProxyError(status_code=503, code="model_catalog_unavailable", message="Oz model catalog did not contain any usable model ids.")
        return catalog

    def _build_model_list_command(self) -> list[str]:
        args = ["oz", "model", "list", "--output-format", "json"]
        if self.settings.auth_mode == "api_key" and self.settings.warp_api_key:
            args.extend(["--api-key", self.settings.warp_api_key])
        return args

    def _resolve_continuation(self, request: ChatCompletionRequest, resolved_model: ResolvedModel) -> tuple[str | None, ConversationRecord | None]:
        metadata = request.metadata or {}
        previous = metadata.get("warp_previous_response_id")
        if previous is None:
            return (None, None)
        if not isinstance(previous, str) or not previous.strip():
            raise ProxyError(status_code=400, code="invalid_conversation_reference", message="metadata.warp_previous_response_id must be a non-empty string.", param="metadata.warp_previous_response_id")
        response_id = previous.strip()
        try:
            record = self._conversation_store.get(response_id)
        except ConversationStoreError as exc:
            raise self._map_store_error(exc, param="metadata.warp_previous_response_id")
        if record is None:
            raise ProxyError(status_code=400, code="invalid_conversation_reference", message="Conversation reference was not found.", param="metadata.warp_previous_response_id")
        if record.backend != resolved_model.backend_command:
            raise ProxyError(status_code=400, code="invalid_conversation_reference", message="Conversation reference does not match the requested backend.", param="metadata.warp_previous_response_id")
        try:
            self._conversation_store.touch(response_id)
        except ConversationStoreError as exc:
            raise self._map_store_error(exc, param="metadata.warp_previous_response_id")
        return (response_id, record)

    def _build_command(self, *, prompt: str, oz_model_id: str | None, conversation_id: str | None) -> list[str]:
        args = ["oz", "agent", "run", "--output-format", "json"]
        if self.settings.cwd:
            args.extend(["--cwd", self.settings.cwd])
        if self.settings.environment:
            args.extend(["--environment", self.settings.environment])
        if self.settings.skill:
            args.extend(["--skill", self.settings.skill])
        for mcp_spec in self.settings.mcp:
            args.extend(["--mcp", mcp_spec])
        if oz_model_id:
            args.extend(["--model", oz_model_id])
        if conversation_id:
            args.extend(["--conversation", conversation_id])
        if self.settings.auth_mode == "api_key" and self.settings.warp_api_key:
            args.extend(["--api-key", self.settings.warp_api_key])
        args.extend(["--prompt", prompt])
        return args

    def _persist_mapping(self, response_id: str, conversation_id: str, backend: str) -> None:
        try:
            self._conversation_store.put(response_id, conversation_id=conversation_id, backend=backend)
        except ConversationStoreError as exc:
            raise self._map_store_error(exc)

    def _map_store_error(self, exc: ConversationStoreError, *, param: str | None = None) -> ProxyError:
        status_map = {
            "invalid_conversation_reference": 400,
            "conversation_expired": 409,
            "conversation_store_corrupt": 500,
            "conversation_store_unavailable": 500,
        }
        return ProxyError(status_code=status_map.get(exc.code, 500), code=exc.code, message=exc.message, param=param)

    def _model_availability(self) -> list[ModelAvailability]:
        return [
            ModelAvailability(id=DEFAULT_MODEL_ALIAS, available=True, reason=None),
        ]

    def _get_or_probe_version_status(self) -> VersionStatus:
        if self._version_status is None:
            self._version_status = self._probe_cli_version_status()
        return self._version_status

    def _ensure_supported_cli_version(self) -> None:
        status = self._get_or_probe_version_status()
        if status.supported:
            return
        if status.error_code == "backend_timeout":
            raise ProxyError(status_code=504, code="backend_timeout", message=status.error_message or "Timed out while probing Warp CLI version.")
        raise ProxyError(status_code=503, code=status.error_code or "unsupported_cli_version", message=status.error_message or "Unsupported Warp CLI version.")

    def _probe_cli_version_status(self) -> VersionStatus:
        try:
            result = self._run_sync(["oz", "dump-debug-info"], timeout=min(self.settings.command_timeout_seconds, 15.0))
        except ProxyError as exc:
            if exc.error.code == "backend_timeout":
                return VersionStatus(checked=True, supported=False, version=None, error_code="backend_timeout", error_message=exc.error.message)
            return VersionStatus(checked=True, supported=False, version=None, error_code="unsupported_cli_version", error_message=exc.error.message)
        version = parse_warp_version(result.stdout)
        if result.returncode != 0 or version is None:
            return VersionStatus(checked=True, supported=False, version=version, error_code="unsupported_cli_version", error_message="Detected unsupported Warp CLI version 'unknown'. Revalidate against an allowed version or set ALLOW_UNVERIFIED_WARP_CLI=true to override.")
        if self.settings.allow_unverified_warp_cli:
            return VersionStatus(checked=True, supported=True, version=version)
        if version in self.settings.verified_warp_versions:
            return VersionStatus(checked=True, supported=True, version=version)
        return VersionStatus(checked=True, supported=False, version=version, error_code="unsupported_cli_version", error_message=f"Detected unsupported Warp CLI version '{version}'. Revalidate against an allowed version or set ALLOW_UNVERIFIED_WARP_CLI=true to override.")

    def _run_sync(self, args: list[str], *, timeout: float | None = None) -> CommandResult:
        timeout_seconds = timeout if timeout is not None else self.settings.command_timeout_seconds
        try:
            return self.runner.run(args=args, timeout_seconds=timeout_seconds)
        except subprocess.TimeoutExpired as exc:
            raise ProxyError(status_code=504, code="backend_timeout", message=f"Oz CLI timed out after {exc.timeout} seconds.") from exc

    async def _prime_stream(self, event_iter: AsyncIterator[ParsedEvent], prepared: PreparedExecution) -> tuple[str, str | None]:
        conversation_id = prepared.prior_record.conversation_id if prepared.prior_record else None
        async for event in event_iter:
            if event.conversation_id:
                conversation_id = event.conversation_id
            if event.kind == "agent" and event.text is not None:
                return event.text, conversation_id
        raise ProxyError(status_code=502, code="malformed_backend_output", message="Oz CLI stream ended before any assistant text was emitted.")

    async def _stream_local_backend_events(self, prepared: PreparedExecution) -> AsyncIterator[ParsedEvent]:
        process = await asyncio.create_subprocess_exec(
            *prepared.args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            limit=_STREAM_READER_LIMIT_BYTES,
        )
        stderr_chunks: list[str] = []

        async def read_stderr() -> None:
            assert process.stderr is not None
            while True:
                chunk = await process.stderr.read(4096)
                if not chunk:
                    return
                stderr_chunks.append(chunk.decode())

        stderr_task = asyncio.create_task(read_stderr())
        try:
            assert process.stdout is not None
            while True:
                try:
                    line = await asyncio.wait_for(process.stdout.readline(), timeout=self.settings.agent_run_timeout_seconds)
                except asyncio.TimeoutError as exc:
                    process.kill()
                    with suppress(ProcessLookupError):
                        await process.wait()
                    raise ProxyError(status_code=504, code="backend_timeout", message="Oz CLI stream timed out.") from exc
                except ValueError as exc:
                    process.kill()
                    with suppress(ProcessLookupError):
                        await process.wait()
                    raise ProxyError(
                        status_code=502,
                        code="backend_output_too_large",
                        message=f"Oz CLI emitted a single output line longer than {_STREAM_READER_LIMIT_BYTES} bytes.",
                    ) from exc
                if not line:
                    break
                raw = line.decode().strip()
                if not raw:
                    continue
                yield parse_event_line(raw)
            returncode = await process.wait()
            await stderr_task
            if returncode != 0:
                raise self._map_backend_failure(CommandResult(args=prepared.args, returncode=returncode, stdout="", stderr="".join(stderr_chunks)), prior_response_id=prepared.prior_response_id)
        finally:
            if not stderr_task.done():
                stderr_task.cancel()
                with suppress(asyncio.CancelledError):
                    await stderr_task
            if process.returncode is None:
                with suppress(ProcessLookupError):
                    process.kill()
                with suppress(Exception):
                    await process.wait()

    def _map_backend_failure(self, result: CommandResult, *, prior_response_id: str | None = None) -> ProxyError:
        combined_output = "\n".join(part for part in (result.stdout.strip(), result.stderr.strip()) if part).strip()
        if prior_response_id and _CONVERSATION_EXPIRED_PATTERN.search(combined_output):
            with suppress(ConversationStoreError):
                self._conversation_store.delete(prior_response_id)
            return ProxyError(status_code=409, code="conversation_expired", message="The referenced Oz conversation is expired or no longer available.", param="metadata.warp_previous_response_id")
        if self.settings.auth_mode == "session" and _AUTH_FAILURE_PATTERN.search(combined_output):
            return ProxyError(status_code=503, code="cli_session_required", message="Oz CLI session is missing or expired. Re-authenticate locally with Warp/Oz and retry.")
        return ProxyError(status_code=502, code="backend_execution_failed", message=combined_output or "Oz CLI execution failed.")


# ---------------------------------------------------------------------------
# 순수 함수 (OzBridge 외부)
# ---------------------------------------------------------------------------


def flatten_messages(request: ChatCompletionRequest) -> str:
    """messages 배열을 [role]\ntext 형식의 단일 프롬프트 문자열로 병합한다."""
    lines: list[str] = []
    for message in request.messages:
        text = message.flattened_content().strip()
        lines.append(f"[{message.role}]\n{text}" if text else f"[{message.role}]")
    lines.append("\nPlease answer the latest user request directly and clearly.")
    return "\n\n".join(lines)


def parse_warp_version(stdout: str) -> str | None:
    """oz dump-debug-info 출력에서 Warp 버전 문자열을 추출한다."""
    match = WARP_VERSION_PATTERN.search(stdout)
    if not match:
        return None
    return match.group("version")


def parse_event_line(raw_line: str) -> ParsedEvent:
    """NDJSON 한 줄을 ParsedEvent로 변환한다."""
    try:
        payload = json.loads(raw_line)
    except json.JSONDecodeError as exc:
        raise ProxyError(status_code=502, code="malformed_backend_output", message="Oz CLI returned malformed JSON output.") from exc
    if not isinstance(payload, dict):
        raise ProxyError(status_code=502, code="malformed_backend_output", message="Oz CLI returned an unexpected non-object event.")
    conversation_id = payload.get("conversation_id") if isinstance(payload.get("conversation_id"), str) else None
    if payload.get("type") == "agent" and isinstance(payload.get("text"), str):
        return ParsedEvent(kind="agent", text=payload["text"], conversation_id=conversation_id, payload=payload)
    if conversation_id:
        return ParsedEvent(kind="system", conversation_id=conversation_id, payload=payload)
    return ParsedEvent(kind=str(payload.get("type") or "other"), payload=payload)


def parse_ndjson_events(stdout: str) -> list[ParsedEvent]:
    """NDJSON 출력 전체를 파싱한다."""
    events: list[ParsedEvent] = []
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        events.append(parse_event_line(line))
    return events


def aggregate_events(events: list[ParsedEvent]) -> tuple[str, str | None]:
    """이벤트 목록에서 agent 텍스트를 병합하고, conversation_id를 추출한다."""
    if not events:
        raise ProxyError(status_code=502, code="malformed_backend_output", message="Oz CLI returned empty output.")
    conversation_id: str | None = None
    agent_chunks: list[str] = []
    for event in events:
        if event.conversation_id:
            conversation_id = event.conversation_id
        if event.kind == "agent" and event.text is not None:
            agent_chunks.append(event.text)
    content = "".join(agent_chunks).strip()
    if not content:
        raise ProxyError(status_code=502, code="malformed_backend_output", message="Oz CLI JSON output did not include any agent text.")
    return content, conversation_id


def parse_json_output(stdout: str) -> str:
    """NDJSON stdout에서 assistant 텍스트만 추출하는 편의 함수."""
    content, _ = aggregate_events(parse_ndjson_events(stdout))
    return content


def _parse_model_catalog(payload: Any) -> tuple[str, ...]:
    if not isinstance(payload, list):
        return ()
    ids: list[str] = []
    for item in payload:
        if isinstance(item, dict) and isinstance(item.get("id"), str) and item["id"].strip():
            ids.append(item["id"].strip())
        elif isinstance(item, str) and item.strip():
            ids.append(item.strip())
    return tuple(dict.fromkeys(ids).keys())


def _namespaced_model_id(prefix: str, oz_model_id: str) -> str:
    return f"{prefix}/{oz_model_id}"


def _sse_data(payload: str) -> str:
    return f"data: {payload}\n\n"
