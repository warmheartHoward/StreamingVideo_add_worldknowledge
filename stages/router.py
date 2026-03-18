"""Step C: Route based on nameplate detection result.

- Positive (nameplate found): pass to world knowledge step
- Negative (no nameplate or VLM error): fill with refusal text
"""

import logging

from core.config_loader import AppConfig
from models.pipeline_models import RoutedResult, VLMResult

logger = logging.getLogger(__name__)


class Router:
    """Routes VLM results to either world knowledge or refusal."""

    def __init__(self, config: AppConfig):
        self._refusal_text = config.pipeline.refusal_text

    def route(self, vlm_result: VLMResult) -> RoutedResult:
        """Apply routing logic based on detection result.

        Returns RoutedResult with:
        - is_positive=True if nameplate detected (generated_content empty, to be filled by Step D)
        - is_positive=False if no nameplate or error (generated_content = refusal text)
        """
        fg = vlm_result.frame_group
        log_prefix = f"[{fg.sample_id}][qa{fg.qa_index}][resp{fg.response_index}]"

        # Error case: VLM call failed entirely
        if vlm_result.error:
            logger.info(f"{log_prefix} Routed to REFUSAL (VLM error)")
            return RoutedResult(
                vlm_result=vlm_result,
                is_positive=False,
                generated_content=self._refusal_text,
            )

        # Check detection result
        detection = vlm_result.detection
        if detection and detection.has_legible_nameplate:
            logger.info(
                f"{log_prefix} Routed to WORLD_KNOWLEDGE "
                f"(best_frame={detection.best_frame_filename})"
            )
            return RoutedResult(
                vlm_result=vlm_result,
                is_positive=True,
                generated_content="",  # Will be filled by Step D
            )

        logger.info(f"{log_prefix} Routed to REFUSAL (no legible nameplate)")
        return RoutedResult(
            vlm_result=vlm_result,
            is_positive=False,
            generated_content=self._refusal_text,
        )
