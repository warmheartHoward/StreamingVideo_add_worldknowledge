"""Step E: Assembly and JSONL persistence.

Assembles EnrichedResult into OutputRecord and appends to the output JSONL file.
Uses asyncio.Lock to protect concurrent writes.
"""

import asyncio
import logging
from pathlib import Path

import aiofiles

from models.output_models import (
    NameplateAnnotation,
    OutputLogits,
    OutputQuestion,
    OutputRecord,
    OutputResponse,
)
from models.pipeline_models import EnrichedResult

logger = logging.getLogger(__name__)


class JSONLWriter:
    """Assembles and writes OutputRecords to a JSONL file."""

    def __init__(self, output_path: str):
        self._output_path = output_path
        self._file = None
        self._lock = asyncio.Lock()
        self._count = 0

    async def _ensure_open(self) -> None:
        """Lazily open the output file on first write."""
        if self._file is None:
            # Ensure parent directory exists
            Path(self._output_path).parent.mkdir(parents=True, exist_ok=True)
            self._file = await aiofiles.open(
                self._output_path, mode="a", encoding="utf-8"
            )
            logger.info(f"Opened output file: {self._output_path}")

    def assemble(self, enriched: EnrichedResult) -> OutputRecord:
        """Convert EnrichedResult to OutputRecord.

        For positive samples: content is filled from world knowledge or OCR text.
        For negative samples: content is the refusal text from routing.
        """
        routed = enriched.routed_result
        vlm = routed.vlm_result
        fg = vlm.frame_group
        detection = vlm.detection

        # Determine response content
        if routed.is_positive:
            content = self._extract_content_from_world_knowledge(
                enriched.world_knowledge, detection, routed.generated_content
            )
        else:
            content = routed.generated_content

        # Build logits from original data
        logits = OutputLogits(**fg.original_logits)

        return OutputRecord(
            sample_id=fg.sample_id,
            video_path=fg.video_path,
            qa_index=fg.qa_index,
            response_index=fg.response_index,
            question=OutputQuestion(
                content=fg.question_content,
                time=fg.question_time,
            ),
            response=OutputResponse(
                content=content,
                st_time=fg.response_st_time,
                end_time=fg.response_end_time,
                time=fg.response_time,
                logits=logits,
            ),
            nameplate=NameplateAnnotation(
                has_legible_nameplate=(
                    detection.has_legible_nameplate if detection else False
                ),
                artifact_description=(
                    detection.artifact_description if detection else ""
                ),
                reasoning_process=(
                    detection.reasoning_process if detection else ""
                ),
                best_frame_filename=(
                    detection.best_frame_filename if detection else None
                ),
                ocr_text=detection.ocr_text if detection else None,
                world_knowledge=enriched.world_knowledge,
            ),
        )

    @staticmethod
    def _extract_content_from_world_knowledge(
        wk: dict | None,
        detection,
        fallback: str,
    ) -> str:
        """Extract response content from world knowledge result.

        Priority:
        1. Labeler report's curatorial_conclusion (from real labeler)
        2. Labeler report's training_caption (markdown, if conclusion empty)
        3. OCR text + artifact description (fallback for mock mode)
        4. The routing-provided fallback text
        """
        if not wk:
            return fallback

        # Real labeler output: check for report.curatorial_conclusion
        report = wk.get("report", {})
        if isinstance(report, dict):
            conclusion = report.get("curatorial_conclusion", "")
            if conclusion:
                return conclusion

        # Training caption fallback
        caption = wk.get("training_caption", "")
        if caption:
            return caption

        # Mock mode: "world_knowledge" == "MOCK_KNOWLEDGE"
        if wk.get("world_knowledge") == "MOCK_KNOWLEDGE":
            parts = []
            if detection and detection.ocr_text:
                parts.append(detection.ocr_text)
            if detection and detection.artifact_description:
                parts.append(detection.artifact_description)
            if parts:
                return "\n".join(parts)

        return fallback

    async def write(self, record: OutputRecord) -> None:
        """Append a single OutputRecord as one JSONL line (thread-safe)."""
        async with self._lock:
            await self._ensure_open()
            line = record.model_dump_json(by_alias=True)
            await self._file.write(line + "\n")  # type: ignore[union-attr]
            self._count += 1

            if self._count % 100 == 0:
                logger.info(f"Written {self._count} records to JSONL")

    async def close(self) -> None:
        """Close the output file."""
        if self._file:
            await self._file.close()  # type: ignore[union-attr]
            logger.info(
                f"Closed output file. Total records written: {self._count}"
            )
