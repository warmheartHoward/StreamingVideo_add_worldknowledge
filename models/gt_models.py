"""Pydantic models for parsing the input gt.json files.

The gt.json format matches v2_project output structure:
- Top level is a list[GTDocument] (usually 1 element)
- data[] contains QA pairs, where response[].content is empty in our input
"""

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class Logits(BaseModel):
    """Response logits with special token aliases."""

    model_config = ConfigDict(populate_by_name=True)

    first_response: float = Field(alias="</first_response>", default=0.0)
    second_response: float = Field(alias="</second_response>", default=0.0)
    silence: float = Field(alias="</silence>", default=0.0)
    standby: float = Field(alias="</standby>", default=0.0)


class ResponseEntry(BaseModel):
    """A single response in a QA pair. In our input, content is empty."""

    content: str = ""
    st_time: str = ""
    end_time: str = ""
    time: float
    logits: Logits


class Question(BaseModel):
    """Question with content text and trigger timestamp."""

    content: str
    time: float


class QAEntry(BaseModel):
    """A question-answer pair containing one question and multiple responses."""

    question: Question
    response: list[ResponseEntry]


class DataProduction(BaseModel):
    version: str = ""
    fps: float = 0.0


class QualityControl(BaseModel):
    version: str = ""
    result: str = ""


class MetaInfo(BaseModel):
    """Sample metadata from gt.json."""

    id: str
    dataset: str = ""
    language: str = "zh"
    video_path: str = ""
    duration: float = 0.0
    data_type: str = ""
    domain: str = ""
    task_type: str = ""
    task_type_source: str = ""
    data_production: list[DataProduction] = []
    quality_control: list[QualityControl] = []


class GTDocument(BaseModel):
    """Root document model for a single gt.json entry.

    The actual gt.json file is a JSON array, typically containing 1 GTDocument.
    """

    meta_info: MetaInfo
    video_path: str
    frame_path: list[str]
    extracted_fps: float
    data: list[QAEntry]
