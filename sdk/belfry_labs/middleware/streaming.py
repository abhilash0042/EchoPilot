"""
Streaming Guard — intercept LLM streaming responses and enforce safety decisions.

Wraps async generators to check each chunk against the Belfry runtime
safety engine and yield/raise on BLOCK actions.
"""

import asyncio
import logging
from typing import AsyncIterator, Any, Callable, Optional, Dict
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class StreamChunk:
    """A single chunk from a streaming LLM response."""
    content: str
    finish_reason: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class StreamingBlockedError(Exception):
    """Raised when streaming content is blocked by Belfry safety engine."""
    def __init__(self, message: str, decision: Dict[str, Any] = None):
        super().__init__(message)
        self.decision = decision or {}


class StreamingGuard:
    """
    Wraps an async streaming LLM response generator to enforce safety decisions.

    Usage:
        from belfry_labs.middleware.streaming import StreamingGuard
        from belfry_labs import AsyncBelfryLabsClient

        client = AsyncBelfryLabsClient(api_key="...")
        guard = StreamingGuard(client, project_id="proj_123")

        # Wrap OpenAI streaming response
        stream = await openai_client.chat.completions.create(..., stream=True)
        async for chunk in guard.wrap(stream, extract_content=lambda c: c.choices[0].delta.content or ""):
            print(chunk.content, end="", flush=True)
    """

    def __init__(
        self,
        belfry_client,
        project_id: Optional[str] = None,
        buffer_size: int = 200,
        check_interval: int = 50,
        on_block: Optional[Callable[[Dict], None]] = None,
        fail_open: bool = False,
    ):
        """
        Args:
            belfry_client: AsyncBelfryLabsClient instance
            project_id: Project ID for context
            buffer_size: Number of chars to buffer before safety check
            check_interval: Check every N chars
            on_block: Callback called with the block decision dict
            fail_open: If True, continue streaming on safety API errors
        """
        self.client = belfry_client
        self.project_id = project_id
        self.buffer_size = buffer_size
        self.check_interval = check_interval
        self.on_block = on_block
        self.fail_open = fail_open

    async def wrap(
        self,
        stream: AsyncIterator[Any],
        extract_content: Callable[[Any], str] = lambda x: str(x),
        extract_finish: Callable[[Any], Optional[str]] = lambda x: None,
    ) -> AsyncIterator[StreamChunk]:
        """
        Wrap a streaming iterator with safety checks.

        Args:
            stream: The async iterator from the LLM provider
            extract_content: Function to extract text from each chunk
            extract_finish: Function to extract finish_reason from each chunk

        Yields:
            StreamChunk objects

        Raises:
            StreamingBlockedError: If content is blocked
        """
        buffer = ""
        chars_since_check = 0

        async for raw_chunk in stream:
            content = extract_content(raw_chunk) or ""
            finish_reason = extract_finish(raw_chunk)

            buffer += content
            chars_since_check += len(content)

            # Check buffer periodically or at end of stream
            if chars_since_check >= self.check_interval or finish_reason:
                if buffer.strip():
                    try:
                        decision = await self.client.runtime(
                            buffer,
                            project_id=self.project_id,
                            context={"check_type": "streaming_output"},
                        )
                        action = (
                            decision.get("action", "allow")
                            if isinstance(decision, dict)
                            else getattr(decision, "action", "allow")
                        )
                        action = action.value if hasattr(action, "value") else str(action)

                        if action == "block":
                            if self.on_block:
                                self.on_block(
                                    decision
                                    if isinstance(decision, dict)
                                    else (decision.to_dict() if hasattr(decision, "to_dict") else {})
                                )
                            raise StreamingBlockedError(
                                "Streaming content blocked by Belfry safety engine",
                                decision=decision if isinstance(decision, dict) else {},
                            )
                        elif action == "redact":
                            yield StreamChunk(
                                content="[REDACTED]",
                                finish_reason=finish_reason,
                                metadata={"redacted": True, "original_length": len(buffer)},
                            )
                            buffer = ""
                            chars_since_check = 0
                            continue

                    except StreamingBlockedError:
                        raise
                    except Exception as e:
                        logger.warning(f"Belfry streaming check failed: {e}")
                        if not self.fail_open:
                            raise StreamingBlockedError(
                                f"Belfry safety check unavailable: {e}"
                            )

                yield StreamChunk(
                    content=buffer,
                    finish_reason=finish_reason,
                    metadata={"buffered_chars": len(buffer)},
                )
                buffer = ""
                chars_since_check = 0

        # Yield any remaining buffer
        if buffer:
            yield StreamChunk(content=buffer)

    async def wrap_openai(self, stream) -> AsyncIterator[StreamChunk]:
        """Convenience wrapper for OpenAI streaming responses."""
        return self.wrap(
            stream,
            extract_content=lambda c: (c.choices[0].delta.content or "") if c.choices else "",
            extract_finish=lambda c: (c.choices[0].finish_reason) if c.choices else None,
        )

    async def wrap_anthropic(self, stream) -> AsyncIterator[StreamChunk]:
        """Convenience wrapper for Anthropic streaming responses."""
        return self.wrap(
            stream,
            extract_content=lambda c: c.delta.text if hasattr(c, "delta") and hasattr(c.delta, "text") else "",
            extract_finish=lambda c: c.type if hasattr(c, "type") and c.type == "message_stop" else None,
        )
