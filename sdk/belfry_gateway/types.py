"""Response objects from the inference gateway."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass
class BelfryDecision:
    request_id: str = ""
    requested_model: str = ""
    routed_model: str = ""
    input_action: str = "allow"
    output_action: str = "allow"
    blocked: bool = False
    block_stage: Optional[str] = None
    findings: list = field(default_factory=list)
    routing_reason: Optional[str] = None
    approval_id: Optional[str] = None
    request_hash: Optional[str] = None
    raw: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Optional[dict]) -> "BelfryDecision":
        data = data or {}
        return cls(
            request_id=str(data.get("request_id") or ""),
            requested_model=str(data.get("requested_model") or ""),
            routed_model=str(data.get("routed_model") or ""),
            input_action=str(data.get("input_action") or "allow"),
            output_action=str(data.get("output_action") or "allow"),
            blocked=bool(data.get("blocked")),
            block_stage=data.get("block_stage"),
            findings=list(data.get("findings") or []),
            routing_reason=data.get("routing_reason"),
            approval_id=data.get("approval_id"),
            request_hash=data.get("request_hash"),
            raw=data,
        )


@dataclass
class ChatCompletion:
    id: str
    model: str
    content: str
    finish_reason: str
    usage: Usage
    belfry: BelfryDecision
    raw: dict = field(default_factory=dict)

    @property
    def blocked(self) -> bool:
        return self.belfry.blocked or self.finish_reason == "content_filter"

    @classmethod
    def from_dict(cls, data: dict) -> "ChatCompletion":
        choices = data.get("choices") or []
        message: dict = {}
        finish = "stop"
        if choices:
            message = choices[0].get("message") or {}
            finish = choices[0].get("finish_reason") or "stop"
        usage_raw = data.get("usage") or {}
        return cls(
            id=str(data.get("id") or ""),
            model=str(data.get("model") or ""),
            content=str(message.get("content") or ""),
            finish_reason=str(finish),
            usage=Usage(
                prompt_tokens=int(usage_raw.get("prompt_tokens") or 0),
                completion_tokens=int(usage_raw.get("completion_tokens") or 0),
                total_tokens=int(usage_raw.get("total_tokens") or 0),
            ),
            belfry=BelfryDecision.from_dict(data.get("belfry")),
            raw=data,
        )


@dataclass
class ChatChunk:
    id: str
    model: str
    delta: str
    finish_reason: Optional[str]
    raw: dict = field(default_factory=dict)

    @property
    def blocked(self) -> bool:
        return self.finish_reason == "content_filter"

    @classmethod
    def from_dict(cls, data: dict) -> "ChatChunk":
        choices = data.get("choices") or []
        delta = ""
        finish = None
        if choices:
            delta = str((choices[0].get("delta") or {}).get("content") or "")
            finish = choices[0].get("finish_reason")
        return cls(
            id=str(data.get("id") or ""),
            model=str(data.get("model") or ""),
            delta=delta,
            finish_reason=finish,
            raw=data,
        )
