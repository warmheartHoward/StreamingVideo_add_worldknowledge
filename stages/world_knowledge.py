"""Step D: World knowledge generation interface.

Provides an abstract base class and a mock implementation.
The actual implementation will be swapped in later by connecting
to an existing knowledge pipeline.
"""

import logging
from abc import ABC, abstractmethod

from models.pipeline_models import EnrichedResult, RoutedResult

logger = logging.getLogger(__name__)


class WorldKnowledgeBase(ABC):
    """Abstract interface for world knowledge generation.

    Implementations should take the best frame image path and OCR text,
    then return structured knowledge about the artifact.
    """

    @abstractmethod
    async def generate(self, image_path: str, ocr_text: str) -> dict:
        """Generate world knowledge for an artifact.

        Args:
            image_path: Absolute path to the best frame image.
            ocr_text: OCR text extracted from the nameplate.

        Returns:
            Dictionary containing world knowledge fields.
        """
        ...


class MockWorldKnowledge(WorldKnowledgeBase):
    """Mock implementation that returns placeholder data.

    Replace with actual implementation by subclassing WorldKnowledgeBase.
    """

    async def generate(self, image_path: str, ocr_text: str) -> dict:
        logger.debug(
            f"MockWorldKnowledge: image={image_path}, ocr_text={ocr_text[:50] if ocr_text else 'None'}..."
        )
        return {"world_knowledge": "MOCK_KNOWLEDGE"}


async def enrich_with_world_knowledge(
    routed: RoutedResult,
    provider: WorldKnowledgeBase,
) -> EnrichedResult:
    """Helper to call world knowledge provider and wrap result.

    Extracts best_frame_path and ocr_text from the VLM result,
    calls the provider, and returns an EnrichedResult.
    """
    detection = routed.vlm_result.detection
    fg = routed.vlm_result.frame_group

    # Resolve best frame path
    best_frame_path = ""
    if detection and detection.best_frame_filename:
        for fname, fpath in zip(fg.frame_filenames, fg.frame_paths):
            if fname == detection.best_frame_filename:
                best_frame_path = fpath
                break
        # Fallback: construct path from sample_dir
        if not best_frame_path:
            from pathlib import Path
            best_frame_path = str(
                Path(fg.sample_dir) / "frames" / detection.best_frame_filename
            )

    ocr_text = detection.ocr_text if detection else ""

    wk_data = await provider.generate(best_frame_path, ocr_text or "")

    return EnrichedResult(
        routed_result=routed,
        world_knowledge=wk_data,
    )
