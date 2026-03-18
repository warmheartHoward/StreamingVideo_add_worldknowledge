"""Step B: Nameplate detection via VLM (OpenAI-compatible API).

Builds OpenAI-format messages with text + base64 image content,
then calls the VLM client for structured nameplate detection.
"""

import asyncio
import base64
import logging
import mimetypes
from pathlib import Path

from core.gemini_client import GeminiClient
from models.pipeline_models import FrameGroup, VLMResult
from prompts.nameplate_detection import SYSTEM_PROMPT, USER_INSTRUCTION

logger = logging.getLogger(__name__)


def _image_to_data_url(image_path: str) -> str:
    """Convert an image file to a base64 data URL."""
    mime_type, _ = mimetypes.guess_type(image_path)
    if mime_type is None:
        mime_type = "image/jpeg"
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime_type};base64,{b64}"


class VLMCaller:
    """Calls VLM for nameplate detection on frame groups."""

    def __init__(self, client: GeminiClient):
        self._client = client

    async def call(self, fg: FrameGroup) -> VLMResult:
        """Process a single FrameGroup through VLM.

        On API failure, returns VLMResult with error field set instead of raising.
        """
        log_prefix = f"[{fg.sample_id}][qa{fg.qa_index}][resp{fg.response_index}@{fg.response_time}s]"

        try:
            messages = await self._build_messages(fg)
            result = await self._client.detect_nameplate(
                messages,
                sample_id=fg.sample_id,
                response_time=fg.response_time,
            )
            return VLMResult(frame_group=fg, detection=result)

        except Exception as e:
            logger.error(f"{log_prefix} VLM call failed: {e}")
            return VLMResult(frame_group=fg, error=str(e))

    async def _build_messages(self, fg: FrameGroup) -> list[dict]:
        """Build OpenAI-format messages with text + base64 images.

        Format:
            system: SYSTEM_PROMPT + JSON schema instruction
            user: [text: "文件名: x.jpg", image_url: base64, ..., text: USER_INSTRUCTION]
        """
        system_content = SYSTEM_PROMPT + GeminiClient.get_json_schema_instruction()

        # Build user content parts: interleaved text labels + images
        user_parts: list[dict] = []
        for fname, fpath in zip(fg.frame_filenames, fg.frame_paths):
            user_parts.append({"type": "text", "text": f"文件名: {fname}"})
            data_url = await asyncio.to_thread(_image_to_data_url, fpath)
            user_parts.append({
                "type": "image_url",
                "image_url": {"url": data_url},
            })

        user_parts.append({"type": "text", "text": USER_INSTRUCTION})

        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_parts},
        ]
