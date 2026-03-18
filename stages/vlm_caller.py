"""Step B: Nameplate detection via Gemini VLM.

Builds interleaved text+image content and calls the Gemini client.
Input format: [system_prompt, "filename: x.jpg", image, "filename: y.jpg", image, ..., user_instruction]
"""

import asyncio
import logging
from pathlib import Path

from PIL import Image
from google.genai import types

from core.gemini_client import GeminiClient
from models.pipeline_models import FrameGroup, VLMResult
from prompts.nameplate_detection import SYSTEM_PROMPT, USER_INSTRUCTION

logger = logging.getLogger(__name__)


class VLMCaller:
    """Calls Gemini for nameplate detection on frame groups."""

    def __init__(self, client: GeminiClient):
        self._client = client

    async def call(self, fg: FrameGroup) -> VLMResult:
        """Process a single FrameGroup through Gemini.

        On API failure, returns VLMResult with error field set instead of raising.
        """
        log_prefix = f"[{fg.sample_id}][qa{fg.qa_index}][resp{fg.response_index}@{fg.response_time}s]"

        try:
            contents = await self._build_contents(fg)
            result = await self._client.detect_nameplate(
                contents,
                sample_id=fg.sample_id,
                response_time=fg.response_time,
            )
            return VLMResult(frame_group=fg, detection=result)

        except Exception as e:
            logger.error(f"{log_prefix} VLM call failed: {e}")
            return VLMResult(frame_group=fg, error=str(e))

    async def _build_contents(self, fg: FrameGroup) -> list:
        """Build interleaved text + image content list for Gemini.

        Format: [system_prompt, "文件名: x.jpg", image, ..., user_instruction]
        """
        parts: list = [SYSTEM_PROMPT]

        for fname, fpath in zip(fg.frame_filenames, fg.frame_paths):
            parts.append(f"文件名: {fname}")
            # Load image in a thread to avoid blocking the event loop
            img = await asyncio.to_thread(Image.open, fpath)
            parts.append(img)

        parts.append(USER_INSTRUCTION)
        return parts
