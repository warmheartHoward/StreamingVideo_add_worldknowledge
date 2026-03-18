"""Internal data carriers for the pipeline stages.

Data flows: FrameGroup (Step A) -> VLMResult (Step B) -> RoutedResult (Step C) -> EnrichedResult (Step D)
"""

from typing import Optional

from pydantic import BaseModel

from .vlm_models import NameplateDetectionResult


class FrameGroup(BaseModel):
    """Step A output: a response timestamp paired with its nearby frames.

    One FrameGroup is created per data[i].response[j] in gt.json.
    """

    sample_id: str
    sample_dir: str
    video_path: str
    qa_index: int
    response_index: int
    question_content: str
    question_time: float
    response_st_time: float
    response_end_time: float
    response_time: float
    original_logits: dict
    frame_paths: list[str]
    frame_filenames: list[str]


class VLMResult(BaseModel):
    """Step B output: FrameGroup enriched with VLM detection result."""

    frame_group: FrameGroup
    detection: Optional[NameplateDetectionResult] = None
    error: Optional[str] = None


class RoutedResult(BaseModel):
    """Step C output: routing decision applied.

    - is_positive=True: nameplate detected, generated_content will be filled by Step D
    - is_positive=False: no nameplate or error, generated_content is refusal text
    """

    vlm_result: VLMResult
    is_positive: bool
    generated_content: str


class EnrichedResult(BaseModel):
    """Step D output: world knowledge attached (if positive sample)."""

    routed_result: RoutedResult
    world_knowledge: Optional[dict] = None
