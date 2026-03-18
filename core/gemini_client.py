"""Async VLM client using OpenAI-compatible API with structured JSON output.

Wraps the OpenAI client for nameplate detection, providing:
- Structured JSON output via response_format + Pydantic schema in prompt
- Exponential backoff retry on transient errors
- Per-call logging with sample context
"""

import asyncio
import json
import logging
from typing import Any, Optional

import httpx
from openai import AsyncOpenAI

from core.config_loader import GeminiConfig
from core.token_tracker import TokenTracker
from models.vlm_models import NameplateDetectionResult

logger = logging.getLogger(__name__)

# JSON schema description appended to system prompt for structured output
_JSON_SCHEMA_INSTRUCTION = """

请严格按照以下JSON格式输出结果，不要输出任何其他内容：
{
    "artifact_description": "对帧中可见文物/展品的简要描述",
    "has_legible_nameplate": true/false,
    "reasoning_process": "关于铭牌可见性和可读性的逐步推理",
    "best_frame_filename": "铭牌文字最清晰的帧文件名（无铭牌则为null）",
    "ocr_text": "铭牌上的完整转录文字（无铭牌则为null）"
}"""


class GeminiClient:
    """Async wrapper using OpenAI-compatible API for nameplate detection."""

    def __init__(self, config: GeminiConfig, token_tracker: Optional[TokenTracker] = None):
        http_client = httpx.AsyncClient(verify=False, timeout=float(config.timeout))
        self._client = AsyncOpenAI(
            api_key=config.api_key,
            base_url=config.base_url or None,
            http_client=http_client,
        )
        self._model = config.model
        self._temperature = config.temperature
        self._max_retries = config.max_retries
        self._retry_base_delay = config.retry_base_delay
        self._token_tracker = token_tracker

    async def detect_nameplate(
        self,
        messages: list[dict[str, Any]],
        *,
        sample_id: str,
        response_time: float,
    ) -> NameplateDetectionResult:
        """Call VLM with structured JSON output for nameplate detection.

        Args:
            messages: OpenAI-format messages with text and image_url content.
            sample_id: Sample identifier for logging context.
            response_time: Response timestamp for logging context.

        Returns:
            Parsed NameplateDetectionResult.

        Raises:
            Exception: After all retries are exhausted.
        """
        log_prefix = f"[{sample_id}][resp@{response_time}s]"
        last_error: Exception | None = None

        for attempt in range(1, self._max_retries + 1):
            try:
                logger.debug(
                    f"{log_prefix} VLM API call attempt {attempt}/{self._max_retries}"
                )

                response = await self._client.chat.completions.create(
                    model=self._model,
                    messages=messages,
                    temperature=self._temperature,
                    response_format={"type": "json_object"},
                )

                # Extract token usage
                input_tokens = 0
                output_tokens = 0
                if response.usage:
                    input_tokens = response.usage.prompt_tokens or 0
                    output_tokens = response.usage.completion_tokens or 0

                if self._token_tracker:
                    await self._token_tracker.record(
                        "gemini_vlm",
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        sample_id=sample_id,
                    )

                # Parse structured output
                raw_content = response.choices[0].message.content or "{}"
                result = NameplateDetectionResult.model_validate_json(raw_content)

                logger.info(
                    f"{log_prefix} Detection complete: "
                    f"has_nameplate={result.has_legible_nameplate}, "
                    f"tokens={input_tokens}+{output_tokens}"
                )
                return result

            except Exception as e:
                last_error = e
                error_type = type(e).__name__

                # Check for non-retryable errors
                if _is_non_retryable(e):
                    logger.error(
                        f"{log_prefix} Non-retryable error ({error_type}): {e}"
                    )
                    raise

                if attempt < self._max_retries:
                    wait_time = min(
                        self._retry_base_delay * (2 ** (attempt - 1)), 60.0
                    )
                    logger.warning(
                        f"{log_prefix} Attempt {attempt} failed ({error_type}): {e}. "
                        f"Retrying in {wait_time:.1f}s..."
                    )
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(
                        f"{log_prefix} All {self._max_retries} attempts failed. "
                        f"Last error ({error_type}): {e}"
                    )

        raise last_error  # type: ignore[misc]

    @staticmethod
    def get_json_schema_instruction() -> str:
        """Return the JSON schema instruction to append to system prompts."""
        return _JSON_SCHEMA_INSTRUCTION


def _is_non_retryable(error: Exception) -> bool:
    """Check if an error should not be retried."""
    error_str = str(error).lower()
    non_retryable_keywords = ["authentication", "permission", "invalid_api_key", "401", "403"]
    return any(kw in error_str for kw in non_retryable_keywords)
