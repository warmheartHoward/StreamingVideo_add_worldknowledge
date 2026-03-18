"""Pydantic model for Gemini VLM structured output.

This model is passed directly to google-genai SDK as response_schema,
enforcing structured JSON output from the model.
"""

from typing import Optional

from pydantic import BaseModel, Field


class NameplateDetectionResult(BaseModel):
    """Structured output schema for Gemini nameplate detection.

    Passed to google-genai as:
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=NameplateDetectionResult,
        )
    """

    artifact_description: str = Field(
        description="Brief description of the artifact/exhibit visible in the frames"
    )
    has_legible_nameplate: bool = Field(
        description="Whether a legible nameplate/label is clearly visible in any frame"
    )
    reasoning_process: str = Field(
        description="Step-by-step reasoning about nameplate visibility and legibility"
    )
    best_frame_filename: Optional[str] = Field(
        default=None,
        description="Filename of the frame where the nameplate text is most legible (None if no nameplate)",
    )
    ocr_text: Optional[str] = Field(
        default=None,
        description="Full transcribed text from the nameplate (None if no nameplate)",
    )
