"""Async Gemini API client using google-genai SDK with structured output.

Wraps the google.genai Client for nameplate detection, providing:
- Structured JSON output via response_schema
- Exponential backoff retry on transient errors
- Per-call logging with sample context
"""

import asyncio
import logging
from typing import Any, Optional

from google import genai
from google.genai import types

from core.config_loader import GeminiConfig
from core.token_tracker import TokenTracker
from models.vlm_models import NameplateDetectionResult

logger = logging.getLogger(__name__)


class GeminiClient:
    """Async wrapper around google-genai for nameplate detection."""

    def __init__(self, config: GeminiConfig, token_tracker: Optional[TokenTracker] = None):
        self._client = genai.Client(api_key=config.api_key)
        self._model = config.model
        self._temperature = config.temperature
        self._max_retries = config.max_retries
        self._retry_base_delay = config.retry_base_delay
        self._timeout = config.timeout
        self._token_tracker = token_tracker

    async def detect_nameplate(
        self,
        contents: list[Any],
        *,
        sample_id: str,
        response_time: float,
    ) -> NameplateDetectionResult:
        """Call Gemini with structured output for nameplate detection.

        Args:
            contents: Interleaved list of text parts and image parts.
            sample_id: Sample identifier for logging context.
            response_time: Response timestamp for logging context.

        Returns:
            Parsed NameplateDetectionResult from Gemini's structured output.

        Raises:
            Exception: After all retries are exhausted.
        """
        log_prefix = f"[{sample_id}][resp@{response_time}s]"
        last_error: Exception | None = None

        for attempt in range(1, self._max_retries + 1):
            try:
                logger.debug(
                    f"{log_prefix} Gemini API call attempt {attempt}/{self._max_retries}"
                )

                response = await self._client.aio.models.generate_content(
                    model=self._model,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=NameplateDetectionResult,
                        temperature=self._temperature,
                    ),
                )

                # Extract token usage from Gemini response
                input_tokens = 0
                output_tokens = 0
                if hasattr(response, "usage_metadata") and response.usage_metadata:
                    um = response.usage_metadata
                    input_tokens = getattr(um, "prompt_token_count", 0) or 0
                    output_tokens = getattr(um, "candidates_token_count", 0) or 0

                if self._token_tracker:
                    await self._token_tracker.record(
                        "gemini_vlm",
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        sample_id=sample_id,
                    )

                result = NameplateDetectionResult.model_validate_json(response.text)
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


def _is_non_retryable(error: Exception) -> bool:
    """Check if an error should not be retried."""
    error_type = type(error).__name__
    # google-genai client errors (auth, bad request) are not retryable
    non_retryable_types = {"ClientError", "AuthenticationError", "PermissionDeniedError"}
    return error_type in non_retryable_types
