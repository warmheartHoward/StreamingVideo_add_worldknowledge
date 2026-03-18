"""JSONL output schema.

Each line in the output JSONL file represents one response time point,
with content filled in by the pipeline.
"""

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class NameplateAnnotation(BaseModel):
    """Nameplate detection metadata attached to each output record."""

    has_legible_nameplate: bool
    artifact_description: str = ""
    reasoning_process: str = ""
    best_frame_filename: Optional[str] = None
    ocr_text: Optional[str] = None
    world_knowledge: Optional[dict] = None


class OutputLogits(BaseModel):
    """Logits with alias support for serialization."""

    model_config = ConfigDict(populate_by_name=True)

    first_response: float = Field(alias="</first_response>", default=0.0)
    second_response: float = Field(alias="</second_response>", default=0.0)
    silence: float = Field(alias="</silence>", default=0.0)
    standby: float = Field(alias="</standby>", default=0.0)


class OutputResponse(BaseModel):
    """Response entry in the output, with content filled."""

    content: str
    st_time: float = 0.0
    end_time: float = 0.0
    time: float = 0.0
    logits: OutputLogits


class OutputQuestion(BaseModel):
    """Question entry in the output."""

    content: str
    time: float


class OutputRecord(BaseModel):
    """A single line in the output JSONL file.

    Represents one processed response time point with:
    - Original question/response metadata
    - Filled-in response content
    - Nameplate detection annotations
    """

    sample_id: str
    video_path: str
    qa_index: int
    response_index: int
    question: OutputQuestion
    response: OutputResponse
    nameplate: NameplateAnnotation
